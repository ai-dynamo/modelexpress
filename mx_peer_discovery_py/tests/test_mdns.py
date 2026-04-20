# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for mx_peer_discovery.mdns."""

import string

from mx_peer_discovery.mdns import (
    DEFAULT_SERVICE_TYPE,
    Config,
    MdnsDiscovery,
    _decode_properties,
    _instance_label_from_fullname,
    _random_instance_name,
)


def test_default_service_type_constant():
    assert DEFAULT_SERVICE_TYPE == "_mx-peer._tcp.local."


def test_random_instance_name_shape():
    name = _random_instance_name()
    assert len(name) == 32
    alphabet = set(string.ascii_lowercase + string.digits)
    assert all(c in alphabet for c in name)


def test_random_instance_name_varies():
    # Two draws from a 36^32 space. Collision probability is vanishingly
    # small; if this ever trips, the RNG is broken.
    a = _random_instance_name()
    b = _random_instance_name()
    assert a != b


def test_instance_label_extracts_from_fullname():
    label = _instance_label_from_fullname(
        "myinstance._mx-peer._tcp.local.",
        "_mx-peer._tcp.local.",
    )
    assert label == "myinstance"


def test_instance_label_returns_none_on_wrong_suffix():
    label = _instance_label_from_fullname(
        "myinstance._other._tcp.local.",
        "_mx-peer._tcp.local.",
    )
    assert label is None


def test_decode_properties_basic():
    props = {b"key1": b"value1", b"key2": b"value2"}
    assert _decode_properties(props) == {"key1": "value1", "key2": "value2"}


def test_decode_properties_none_value_becomes_empty():
    props = {b"flag": None}
    assert _decode_properties(props) == {"flag": ""}


def test_decode_properties_invalid_utf8_replaces():
    props = {b"key": b"\xff\xfe"}
    decoded = _decode_properties(props)
    assert "key" in decoded
    # Should not raise; the replacement char indicates the decode happened.
    assert "\ufffd" in decoded["key"]


def test_config_defaults():
    config = Config(
        hostname="host.local.",
        ip="127.0.0.1",
        port=4001,
        on_resolved=lambda *args: None,
    )
    assert config.service_type == DEFAULT_SERVICE_TYPE
    assert config.instance_name is None
    assert config.txt == {}


def test_discovery_instance_name_auto_generated():
    config = Config(
        hostname="host.local.",
        ip="127.0.0.1",
        port=4001,
        on_resolved=lambda *args: None,
    )
    discovery = MdnsDiscovery(config)
    assert len(discovery.instance_name) == 32


def test_discovery_instance_name_explicit():
    config = Config(
        hostname="host.local.",
        ip="127.0.0.1",
        port=4001,
        on_resolved=lambda *args: None,
        instance_name="my-explicit-name",
    )
    discovery = MdnsDiscovery(config)
    assert discovery.instance_name == "my-explicit-name"
