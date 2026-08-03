from __future__ import annotations

import pytest

from quantbacktest.config_io import ConfigLoadError, load_yaml_mapping


def test_yaml12_loader_keeps_legacy_boolean_words_as_strings() -> None:
    payload = load_yaml_mapping("debug:\n  mode: off\nvalues: [on, yes, no]\n")

    assert payload["debug"]["mode"] == "off"
    assert payload["values"] == ["on", "yes", "no"]


def test_yaml12_loader_keeps_true_false_as_booleans() -> None:
    payload = load_yaml_mapping("ml:\n  enabled: true\nrisk:\n  enabled: false\n")

    assert payload["ml"]["enabled"] is True
    assert payload["risk"]["enabled"] is False


def test_yaml_loader_requires_mapping_root() -> None:
    with pytest.raises(ConfigLoadError, match="根节点必须是对象"):
        load_yaml_mapping("- not\n- a mapping\n")
