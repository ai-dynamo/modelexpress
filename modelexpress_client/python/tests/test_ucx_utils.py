# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os

from modelexpress import ucx_utils


def test_nic_access_requires_exposed_uverbs_device(monkeypatch):
    monkeypatch.setattr(
        ucx_utils.os,
        "listdir",
        lambda path: ["uverbs4"] if path.endswith("infiniband_verbs") else [],
    )
    monkeypatch.setattr(
        ucx_utils.os.path,
        "exists",
        lambda path: path == "/dev/infiniband/uverbs4",
    )

    assert ucx_utils._nic_has_accessible_verbs_device("mlx5_4")


def test_nic_access_rejects_host_only_uverbs_device(monkeypatch):
    monkeypatch.setattr(
        ucx_utils.os,
        "listdir",
        lambda path: ["uverbs0"] if path.endswith("infiniband_verbs") else [],
    )
    monkeypatch.setattr(ucx_utils.os.path, "exists", lambda path: False)

    assert not ucx_utils._nic_has_accessible_verbs_device("mlx5_0")


def _install_fake_topology(monkeypatch, gpus, nics, visible=None):
    """Drive probe_nic_pin_for_device from an in-memory topology.

    ``gpus`` maps host GPU index -> (bdf, numa, pci_path) and describes the
    whole machine, which is what the PCI listing reports;
    ``nics`` is the list of (name, numa, rate_gbps, pci_path) tuples that
    _list_compute_ib_nics would return for the devices actually exposed to the
    container.

    ``visible`` names the subset of those host indices that CUDA exposes to this
    process, defaulting to all of them. Passing a single index models a worker
    launched under its own ``CUDA_VISIBLE_DEVICES`` mask: CUDA renumbers what is
    left from 0, so the visible index a rank drives is a position in this list,
    not a host index. Keeping the two distinguishable is the entire point of the
    parameter - a harness that conflates them cannot reproduce the convergence.
    """
    host_order = list(gpus) if visible is None else list(visible)
    monkeypatch.setattr(
        ucx_utils, "_list_compute_ib_nics", lambda min_rate_gbps=None: nics
    )
    monkeypatch.setattr(ucx_utils.torch.cuda, "device_count", lambda: len(host_order))
    monkeypatch.setattr(
        ucx_utils,
        "_gpu_pci_bdf",
        lambda gi: gpus[host_order[gi]][0] if gi < len(host_order) else None,
    )
    monkeypatch.setattr(
        ucx_utils, "_host_gpu_bdfs", lambda: sorted(spec[0] for spec in gpus.values())
    )
    bdf_to_gpu = {spec[0]: spec for spec in gpus.values()}
    monkeypatch.setattr(
        ucx_utils,
        "_pci_path_components",
        lambda bdf: bdf_to_gpu[bdf][2] if bdf in bdf_to_gpu else [],
    )
    monkeypatch.setattr(
        ucx_utils,
        "_read_int_file",
        lambda path: next(
            (spec[1] for spec in bdf_to_gpu.values() if spec[0] in path), None
        ),
    )


# Measured on cluster node hx78c: 4 GPUs all on NUMA 1, but the RDMA device
# plugin handed the pod three rails rooted on NUMA 0 and one on NUMA 1.
#
# The paths matter as much as the NUMA numbers, and are the real measured ones.
# Only GPU3 shares a component with mlx5_11; GPU0/1/2 share nothing with any
# rail and so score 0 against all four, including the same-socket one. That is
# not a simplification of the fixture - it is what the metric does on this
# hardware, because _pci_common_depth drops the root complex. The tests below
# depend on it, so test_the_fixture_reproduces_the_depth_collapse asserts it.
_MISAFFINE_GPUS = {
    0: ("0000:9a:00.0", 1, ["0000:97:01.0", "0000:98:00.0", "0000:9a:00.0"]),
    1: ("0000:aa:00.0", 1, ["0000:a7:01.0", "0000:a8:00.0", "0000:aa:00.0"]),
    2: ("0000:ba:00.0", 1, ["0000:b7:01.0", "0000:b8:00.0", "0000:ba:00.0"]),
    3: ("0000:ca:00.0", 1, ["0000:c7:01.0", "0000:c8:00.0", "0000:ca:00.0"]),
}
_MISAFFINE_NICS = [
    ("mlx5_0", 0, 400.0, ["0000:15:01.0", "0000:19:00.0"]),
    ("mlx5_1", 0, 400.0, ["0000:26:01.0", "0000:2a:00.0"]),
    ("mlx5_11", 1, 400.0, ["0000:c7:01.0", "0000:cb:00.0"]),
    ("mlx5_2", 0, 400.0, ["0000:37:01.0", "0000:3b:00.0"]),
]


def test_the_fixture_reproduces_the_depth_collapse(monkeypatch):
    """Guard the property the two tests below silently rely on.

    Every assertion about this topology turns on GPU0/1/2 scoring 0 against
    ``mlx5_11`` - tied with the cross-socket rails - so that only the
    cross-socket term can tell them apart. If someone "improves" these fixture
    paths to give the same-socket rail a shared component, the ranking tests
    keep passing while no longer testing the tie that the real pod presents.

    Asserted directly against ``_pci_common_depth`` rather than inferred from a
    pin, so a failure here says the inputs drifted, not that the ranking broke.
    """
    nic_paths = {name: path for name, _numa, _rate, path in _MISAFFINE_NICS}

    for gpu in (0, 1, 2):
        gpu_path = _MISAFFINE_GPUS[gpu][2]
        depths = {
            name: ucx_utils._pci_common_depth(gpu_path, path)
            for name, path in nic_paths.items()
        }
        assert set(depths.values()) == {0}, (
            f"GPU{gpu} must tie at depth 0 against every rail, got {depths}"
        )

    assert (
        ucx_utils._pci_common_depth(_MISAFFINE_GPUS[3][2], nic_paths["mlx5_11"]) > 0
    ), "GPU3 is the one GPU with real PCIe affinity; without it the tie is total"


def test_partial_allocation_spreads_rather_than_sharing_one_rail(monkeypatch):
    """Under-provisioned pod: four distinct rails, not four GPUs on one.

    This test previously asserted the exact opposite - that all four GPUs pin to
    the single same-socket rail ``mlx5_11`` - on the reasoning that scattering
    them onto NUMA-0 rails would send every byte across UPI and cost 4.4x. The
    measurement says the reverse, and the 4.4x belongs to the other arm:

    - four readers sharing ``mlx5_11``: 1.5 GB/s each, 6.1 GB/s aggregate
    - four readers on four distinct rails: 6.75 GB/s each, 27.0 GB/s aggregate

    Sharing a rail cost 4.45x. Crossing a socket cost nothing measurable - the
    one reader on its affine rail got 6.747 GB/s against 6.745 for the three
    cross-socket ones. So distinctness must outrank NUMA locality, and the old
    expectation encoded the pathological configuration as the desired one.

    GPU3 keeps ``mlx5_11`` because it is the only GPU with real PCIe affinity to
    it, which is what the best-affinity-first visit order is for; in plain index
    order GPU0 takes it on a zero-depth tie and GPU3 is pushed off.
    """
    _install_fake_topology(monkeypatch, _MISAFFINE_GPUS, _MISAFFINE_NICS)

    chosen = {gpu: ucx_utils.probe_nic_pin_for_device(gpu) for gpu in _MISAFFINE_GPUS}

    assert len(set(chosen.values())) == 4, (
        f"every GPU must get its own rail; got {chosen}"
    )
    assert chosen[3] == "mlx5_11:1", (
        "the one GPU with PCIe affinity to the same-socket rail should keep it"
    )


def test_rank_local_cuda_visibility_still_spreads(monkeypatch):
    """Four one-GPU workers must still land on four rails.

    The test above runs the probe four times inside one process that can see all
    four GPUs, which is the arrangement that makes the shared assignment counter
    work by accident: every call rebuilds the same four-GPU map. Production does
    not look like that. NeMo-RL and prime-RL give each worker its own
    ``CUDA_VISIBLE_DEVICES``, so each process sees one device, calls it index 0,
    and - when the map was built from ``torch.cuda.device_count()`` - restarted
    the tally at zero and picked the same best rail as its three peers. Four
    workers, one rail: exactly the collapse this module exists to prevent,
    produced by the code meant to prevent it.

    So the loop below never lets a rank see more than its own GPU, and the ranks
    never talk. Distinctness here can only come from each of them deriving the
    same host-wide map independently, which is the property under test.
    """
    chosen = {}
    for host_index in _MISAFFINE_GPUS:
        _install_fake_topology(
            monkeypatch, _MISAFFINE_GPUS, _MISAFFINE_NICS, visible=[host_index]
        )
        # Index 0 because that is what CUDA renumbers the masked device to, and
        # what the worker passes; the rank has no way to name its host index.
        chosen[host_index] = ucx_utils.probe_nic_pin_for_device(0)

    assert len(set(chosen.values())) == 4, (
        f"each rank-local worker must still get its own rail; got {chosen}"
    )
    assert chosen[3] == "mlx5_11:1", (
        "the affinity-aware placement must survive the masking too, not just "
        "the distinctness"
    )


def test_rank_local_assignment_matches_the_full_visibility_one(monkeypatch):
    """Masking changes what a rank can see, and must change nothing it decides.

    Stronger than distinctness: it pins that the two paths agree tuple-for-tuple,
    so a future change cannot satisfy the test above by spreading rails some
    other way while quietly giving masked and unmasked ranks different answers.
    """
    _install_fake_topology(monkeypatch, _MISAFFINE_GPUS, _MISAFFINE_NICS)
    unmasked = {gpu: ucx_utils.probe_nic_pin_for_device(gpu) for gpu in _MISAFFINE_GPUS}

    masked = {}
    for host_index in _MISAFFINE_GPUS:
        _install_fake_topology(
            monkeypatch, _MISAFFINE_GPUS, _MISAFFINE_NICS, visible=[host_index]
        )
        masked[host_index] = ucx_utils.probe_nic_pin_for_device(0)

    assert masked == unmasked, (
        f"masked ranks disagree with the full-visibility map: {masked} != {unmasked}"
    )


def test_unreadable_pci_listing_falls_back_to_visible_devices(monkeypatch):
    """No host listing must degrade, not crash.

    ``_host_gpu_bdfs`` returns [] on any unreadable ``/sys``, and on such a host
    the probe cannot spread across GPUs it cannot enumerate. It should still pin
    the rank to a sensible rail from what it can see - the pre-existing
    behaviour - because a pin on a possibly-shared rail beats no pin at all.
    """
    _install_fake_topology(monkeypatch, _MISAFFINE_GPUS, _MISAFFINE_NICS)
    monkeypatch.setattr(ucx_utils, "_host_gpu_bdfs", list)

    assert ucx_utils.probe_nic_pin_for_device(3) == "mlx5_11:1"


def test_a_lone_same_socket_rail_is_not_handed_to_everyone(monkeypatch):
    """The specific regression, stated as the property rather than a mapping.

    ``mlx5_11`` is the only rail on the GPUs' NUMA node. Any ranking that puts
    NUMA locality above load balancing gives it to all four GPUs in turn, since
    a same-socket rail beats every free cross-socket one on every pass. That is
    precisely the configuration measured at 1.5 GB/s per reader, and it is what
    the intermediate ``(-score, cross_socket, count, name)`` key produced.

    The key this replaced in main, ``(-score, count, name)``, fails this too but
    less badly - two GPUs on the lone rail rather than four - because with every
    pair at depth 0 the count term does get to break the tie. Worth knowing when
    reading the measured numbers: they came from UCX choosing for itself with no
    pin active, not from this probe running and getting it wrong.
    """
    _install_fake_topology(monkeypatch, _MISAFFINE_GPUS, _MISAFFINE_NICS)

    on_the_lone_rail = [
        gpu
        for gpu in _MISAFFINE_GPUS
        if ucx_utils.probe_nic_pin_for_device(gpu) == "mlx5_11:1"
    ]

    assert len(on_the_lone_rail) == 1, (
        f"only one GPU may pin to the lone same-socket rail, got {on_the_lone_rail}"
    )


# Same cluster, node th9sn: the allocation the scheduler is supposed to produce,
# with one PCIe-affine rail per GPU spread across both sockets.
_AFFINE_GPUS = {
    0: ("0000:18:00.0", 0, ["0000:15:01.0", "0000:16:00.0", "0000:18:00.0"]),
    1: ("0000:29:00.0", 0, ["0000:26:01.0", "0000:27:00.0", "0000:29:00.0"]),
    2: ("0000:3a:00.0", 0, ["0000:37:01.0", "0000:38:00.0", "0000:3a:00.0"]),
    3: ("0000:4b:00.0", 0, ["0000:48:01.0", "0000:49:00.0", "0000:4b:00.0"]),
    4: ("0000:9a:00.0", 1, ["0000:97:01.0", "0000:98:00.0", "0000:9a:00.0"]),
    5: ("0000:aa:00.0", 1, ["0000:a7:01.0", "0000:a8:00.0", "0000:aa:00.0"]),
    6: ("0000:ba:00.0", 1, ["0000:b7:01.0", "0000:b8:00.0", "0000:ba:00.0"]),
    7: ("0000:ca:00.0", 1, ["0000:c7:01.0", "0000:c8:00.0", "0000:ca:00.0"]),
}
_AFFINE_NICS = [
    ("mlx5_0", 0, 400.0, ["0000:15:01.0", "0000:19:00.0"]),
    ("mlx5_1", 0, 400.0, ["0000:26:01.0", "0000:2a:00.0"]),
    ("mlx5_10", 1, 400.0, ["0000:b7:01.0", "0000:bc:00.0"]),
    ("mlx5_11", 1, 400.0, ["0000:c7:01.0", "0000:cb:00.0"]),
    ("mlx5_2", 0, 400.0, ["0000:37:01.0", "0000:3b:00.0"]),
    ("mlx5_3", 0, 400.0, ["0000:48:01.0", "0000:4c:00.0"]),
    ("mlx5_4", 1, 400.0, ["0000:97:01.0", "0000:9b:00.0"]),
    ("mlx5_5", 1, 400.0, ["0000:a7:01.0", "0000:ab:00.0"]),
]


def test_affine_allocation_keeps_one_to_one_mapping(monkeypatch):
    """The healthy topology still gets a distinct PCIe-local NIC per GPU.

    The other half of the ranking change. Leading with distinctness must not cost
    anything where affinity is fully satisfiable: with one PIX rail per GPU the
    answer is unchanged, each GPU on its own closest rail. If this mapping ever
    shifts, distinctness has started overriding real PCIe locality rather than
    just breaking ties it cannot resolve.
    """
    _install_fake_topology(monkeypatch, _AFFINE_GPUS, _AFFINE_NICS)

    chosen = {gpu: ucx_utils.probe_nic_pin_for_device(gpu) for gpu in _AFFINE_GPUS}

    assert chosen == {
        0: "mlx5_0:1",
        1: "mlx5_1:1",
        2: "mlx5_2:1",
        3: "mlx5_3:1",
        4: "mlx5_4:1",
        5: "mlx5_5:1",
        6: "mlx5_10:1",
        7: "mlx5_11:1",
    }
    assert len(set(chosen.values())) == len(chosen)


def test_more_gpus_than_rails_balances_reuse(monkeypatch):
    """When sharing is unavoidable it must be spread evenly, not stacked.

    Eight GPUs and two rails: the answer is four apiece. A ranking that leads
    with anything other than the assignment count can pile GPUs onto whichever
    rail scores best and leave the other idle - the same failure as the lone-rail
    case, just without a same-socket rail to blame it on.
    """
    nics = [
        ("mlx5_0", 0, 400.0, ["0000:15:01.0", "0000:19:00.0"]),
        ("mlx5_11", 1, 400.0, ["0000:c7:01.0", "0000:cb:00.0"]),
    ]
    _install_fake_topology(monkeypatch, _AFFINE_GPUS, nics)

    counts: dict[str, int] = {}
    for gpu in _AFFINE_GPUS:
        pinned = ucx_utils.probe_nic_pin_for_device(gpu)
        counts[pinned] = counts.get(pinned, 0) + 1

    assert sorted(counts.values()) == [4, 4], f"uneven reuse: {counts}"


def test_unknown_numa_does_not_override_pcie_affinity(monkeypatch):
    """A NIC reporting numa_node -1 is not penalised as cross-socket.

    Kernels report -1 when NUMA is unknown, which is absence of information
    rather than evidence of a bad path, so PCIe depth must still decide.
    """
    gpus = {0: ("0000:9a:00.0", 1, ["0000:97:01.0", "0000:9a:00.0"])}
    nics = [
        ("mlx5_4", -1, 400.0, ["0000:97:01.0", "0000:9b:00.0"]),
        ("mlx5_11", 1, 400.0, ["0000:c7:01.0", "0000:cb:00.0"]),
    ]
    _install_fake_topology(monkeypatch, gpus, nics)

    assert ucx_utils.probe_nic_pin_for_device(0) == "mlx5_4:1"


def test_host_gpu_listing_ignores_cuda_visibility(monkeypatch, tmp_path):
    """The listing must report GPUs, all of them, and nothing else.

    Built against a real directory tree rather than a mocked reader, because the
    bug this guards is a filter that is too loose or too tight, and a mock of
    the filter's own inputs cannot show that. The tree holds one VGA-class and
    one 3D-controller-class NVIDIA GPU (both spellings appear on real hosts), a
    Mellanox NIC that must not be mistaken for a GPU, and a non-GPU NVIDIA
    device, which is the entry a vendor-only check would wrongly pick up.
    """
    devices = tmp_path / "devices"
    entries = {
        "0000:9a:00.0": ("0x10de", "0x030000"),  # NVIDIA VGA controller
        "0000:aa:00.0": ("0x10de", "0x030200"),  # NVIDIA 3D controller
        "0000:c7:00.0": ("0x15b3", "0x020700"),  # Mellanox NIC
        "0000:05:00.0": ("0x10de", "0x0c0330"),  # NVIDIA USB controller
    }
    for bdf, (vendor, pci_class) in entries.items():
        (devices / bdf).mkdir(parents=True)
        (devices / bdf / "vendor").write_text(f"{vendor}\n")
        (devices / bdf / "class").write_text(f"{pci_class}\n")

    real_listdir, real_open = os.listdir, open
    monkeypatch.setattr(
        ucx_utils.os,
        "listdir",
        lambda path: (
            real_listdir(devices)
            if path == "/sys/bus/pci/devices"
            else real_listdir(path)
        ),
    )
    monkeypatch.setattr(
        ucx_utils,
        "_read_str_file",
        lambda path: (
            real_open(str(devices) + path[len("/sys/bus/pci/devices") :]).read().strip()
            if path.startswith("/sys/bus/pci/devices")
            else None
        ),
    )

    assert ucx_utils._host_gpu_bdfs() == ["0000:9a:00.0", "0000:aa:00.0"]


def test_host_gpu_listing_is_empty_when_sysfs_is_unreadable(monkeypatch):
    """An unreadable /sys is a degraded host, not a crashing one."""

    def _boom(_path):
        raise OSError("no /sys here")

    monkeypatch.setattr(ucx_utils.os, "listdir", _boom)

    assert ucx_utils._host_gpu_bdfs() == []
