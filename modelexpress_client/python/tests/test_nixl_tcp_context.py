# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
from unittest.mock import MagicMock

import pytest

from modelexpress import nixl_transfer as nt


@pytest.mark.parametrize('original', [None, 'rc,cuda'])
@pytest.mark.parametrize('fail', [False, True])
def test_tcp_configuration_is_scoped(monkeypatch, original, fail):
    monkeypatch.setattr(nt, 'NIXL_AVAILABLE', True)
    monkeypatch.setattr(nt, 'nixl_agent_config', MagicMock())
    monkeypatch.setenv('MX_NIXL_BACKEND', 'UCX')
    monkeypatch.setenv('NIXL_UCX_TLS', 'rc,cuda')
    for key in ('UCX_TLS', 'UCX_NET_DEVICES'):
        if original is None:
            monkeypatch.delenv(key, raising=False)
        else:
            monkeypatch.setenv(key, original)
    pin = MagicMock()
    monkeypatch.setattr(nt.ucx_utils, 'apply_nic_pin_for_device', pin)
    seen = []

    def create(*args):
        seen.append((os.environ.get('UCX_TLS'), os.environ.get('UCX_NET_DEVICES')))
        if fail:
            raise RuntimeError('creation failed')
        return MagicMock()

    monkeypatch.setattr(nt, 'NixlAgent', create)
    manager = nt.NixlTransferManager('tcp', 0, listen_port=0,
                                    accelerator_backend=MagicMock(), ucx_mode='tcp')
    try:
        if fail:
            with pytest.raises(RuntimeError, match='creation failed'):
                manager.initialize()
        else:
            manager.initialize()
        assert seen == [('tcp,cuda_copy', 'all')]
        pin.assert_not_called()
        assert os.environ.get('UCX_TLS') == original
        assert os.environ.get('UCX_NET_DEVICES') == original
    finally:
        manager.shutdown()


def test_tcp_rejects_other_backends(monkeypatch):
    monkeypatch.setenv('MX_NIXL_BACKEND', 'LIBFABRIC')
    with pytest.raises(ValueError, match='UCX'):
        nt.NixlTransferManager('tcp', 0, ucx_mode='tcp')


def test_tcp_arena_uses_per_tensor_registration(monkeypatch):
    monkeypatch.setenv('MX_NIXL_BACKEND', 'UCX')
    manager = nt.NixlTransferManager('tcp', 0, ucx_mode='tcp')
    manager._agent = MagicMock()
    manager.register_tensors = MagicMock(return_value=b'metadata')
    tensors = {'weight': object()}
    assert manager.register_arena(MagicMock(), tensors) == b'metadata'
    manager.register_tensors.assert_called_once_with(tensors, force_per_tensor=True)


def test_auto_creation_waits_for_tcp_environment(monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Event, Lock

    monkeypatch.setattr(nt, 'NIXL_AVAILABLE', True)
    monkeypatch.setattr(nt, 'nixl_agent_config', MagicMock())
    monkeypatch.setattr(nt.ucx_utils, 'apply_nic_pin_for_device', MagicMock())
    monkeypatch.setenv('MX_NIXL_BACKEND', 'UCX')
    monkeypatch.setenv('NIXL_UCX_TLS', 'rc,cuda')
    monkeypatch.delenv('UCX_NET_DEVICES', raising=False)
    monkeypatch.setenv("UCX_TLS", "rc,cuda")
    entered, release, auto_waiting = Event(), Event(), Event()
    seen = []

    class ObservedLock:
        def __init__(self):
            self.lock = Lock()

        def __enter__(self):
            if entered.is_set():
                auto_waiting.set()
            self.lock.acquire()

        def __exit__(self, *args):
            self.lock.release()

    monkeypatch.setattr(nt, '_UCX_INIT_LOCK', ObservedLock())

    def create(name, config):
        seen.append((name, os.environ.get('UCX_TLS'), os.environ.get('UCX_NET_DEVICES')))
        if name == 'tcp':
            entered.set()
            assert release.wait(5)
        return MagicMock()

    monkeypatch.setattr(nt, 'NixlAgent', create)
    tcp = nt.NixlTransferManager('tcp', 0, accelerator_backend=MagicMock(), ucx_mode='tcp')
    auto = nt.NixlTransferManager('auto', 0, accelerator_backend=MagicMock())

    try:
        with ThreadPoolExecutor(2) as pool:
            first = pool.submit(tcp.initialize)
            try:
                assert entered.wait(5)
                second = pool.submit(auto.initialize)
                assert auto_waiting.wait(5)
                assert not second.done()
                assert os.environ["UCX_TLS"] == "tcp,cuda_copy"
                assert os.environ["UCX_NET_DEVICES"] == "all"
            finally:
                release.set()
            first.result(timeout=5)
            second.result(timeout=5)
        assert seen == [('tcp', 'tcp,cuda_copy', 'all'), ('auto', 'rc,cuda', None)]
    finally:
        tcp.shutdown()
        auto.shutdown()
