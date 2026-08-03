from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import yaml


class ConfigLoadError(ValueError):
    """YAML 配置无法被安全地读取或不符合根节点约定。"""


class _Yaml12SafeLoader(yaml.SafeLoader):
    """保留 PyYAML SafeLoader 的安全性，同时采用 YAML 1.2 布尔语义。"""


_Yaml12SafeLoader.yaml_implicit_resolvers = {
    key: [
        resolver
        for resolver in resolvers
        if resolver[0] != "tag:yaml.org,2002:bool"
    ]
    for key, resolvers in yaml.SafeLoader.yaml_implicit_resolvers.items()
}
_Yaml12SafeLoader.add_implicit_resolver(
    "tag:yaml.org,2002:bool",
    re.compile(r"^(?:true|True|TRUE|false|False|FALSE)$"),
    list("tTfF"),
)


def load_yaml_mapping(content: str, *, source: str = "YAML 配置") -> dict[str, Any]:
    """读取 YAML 对象；off/on/yes/no 按字符串处理，true/false 才是布尔值。"""
    try:
        payload = yaml.load(content, Loader=_Yaml12SafeLoader)
    except yaml.YAMLError as exc:
        raise ConfigLoadError(f"无法读取 {source}：{exc}") from exc
    if not isinstance(payload, dict):
        raise ConfigLoadError(f"{source} 的根节点必须是对象")
    return payload


def load_yaml_file(path: str | Path) -> dict[str, Any]:
    """以 UTF-8 读取配置文件，并在错误中保留具体文件位置。"""
    config_path = Path(path)
    try:
        content = config_path.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        raise ConfigLoadError(f"无法读取 YAML 配置 {config_path}：文件必须使用 UTF-8 编码") from exc
    except OSError as exc:
        raise ConfigLoadError(f"无法读取 YAML 配置 {config_path}：{exc}") from exc
    return load_yaml_mapping(content, source=f"YAML 配置 {config_path}")
