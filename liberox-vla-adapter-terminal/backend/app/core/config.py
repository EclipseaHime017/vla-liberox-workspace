"""Strict configuration for the local-only Web UI."""

from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

import yaml


DEFAULT_UI_CONFIG = Path(__file__).resolve().parents[4] / "configs" / "ui_config.yaml"


class UniqueKeyLoader(yaml.SafeLoader):
    pass


def _unique_mapping(loader: UniqueKeyLoader, node: yaml.MappingNode, deep: bool = False):
    result = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in result:
            raise yaml.constructor.ConstructorError(
                "while constructing a mapping",
                node.start_mark,
                f"duplicate key {key!r}",
                key_node.start_mark,
            )
        result[key] = loader.construct_object(value_node, deep=deep)
    return result


UniqueKeyLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    _unique_mapping,
)


@dataclass(frozen=True)
class AdditionalTaskConfig:
    level: str
    task_name: str


@dataclass(frozen=True)
class UIConfig:
    host: str
    port: int
    dataset_root: Path
    project_id: str
    legacy_scan_roots: tuple[Path, ...]
    preview_width: int
    preview_height: int
    preview_fps: int
    jpeg_quality: int
    manual_translation_gain: float
    manual_rotation_gain: float
    additional_tasks: tuple[AdditionalTaskConfig, ...]

    @property
    def project_root(self) -> Path:
        return self.dataset_root / "projects" / self.project_id

    @property
    def output_root(self) -> Path:
        """Managed run root; kept as a computed compatibility name."""
        return self.project_root / "runs"

    @property
    def catalog_path(self) -> Path:
        return self.dataset_root / "catalog.sqlite3"

    @property
    def scan_roots(self) -> tuple[Path, ...]:
        return (*self.legacy_scan_roots, self.output_root)


def _resolve(path: str, base: Path) -> Path:
    value = Path(path).expanduser()
    if not value.is_absolute():
        value = base / value
    return value.resolve()


def _strict_int(raw: dict[str, Any], key: str) -> int:
    value = raw[key]
    if type(value) is not int:
        raise TypeError(f"UI key {key!r} must be an integer")
    return value


def _strict_number(raw: dict[str, Any], key: str) -> float:
    value = raw[key]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"UI key {key!r} must be a number")
    return float(value)


def _strict_string(raw: dict[str, Any], key: str) -> str:
    value = raw[key]
    if not isinstance(value, str) or not value.strip():
        raise TypeError(f"UI key {key!r} must be a non-empty string")
    return value


def load_ui_config(path: Path = DEFAULT_UI_CONFIG) -> UIConfig:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"UI configuration not found: {path}")
    raw = yaml.load(path.read_text(encoding="utf-8"), Loader=UniqueKeyLoader)
    if not isinstance(raw, dict):
        raise TypeError("UI configuration root must be a mapping")
    expected = {field.name for field in fields(UIConfig)}
    missing = sorted(expected - set(raw))
    unknown = sorted(set(raw) - expected)
    if missing:
        raise ValueError(f"Missing UI configuration keys: {missing}")
    if unknown:
        raise ValueError(f"Unknown UI configuration keys: {unknown}")
    scan_roots = raw["legacy_scan_roots"]
    if not isinstance(scan_roots, list):
        raise TypeError("UI key 'legacy_scan_roots' must be a list")
    if any(not isinstance(value, str) or not value.strip() for value in scan_roots):
        raise TypeError("Every legacy_scan_roots entry must be a non-empty string")
    additional_tasks = raw["additional_tasks"]
    if not isinstance(additional_tasks, list):
        raise TypeError("UI key 'additional_tasks' must be a list")
    parsed_tasks: list[AdditionalTaskConfig] = []
    seen_tasks: set[tuple[str, str]] = set()
    for index, value in enumerate(additional_tasks):
        if not isinstance(value, dict):
            raise TypeError(f"additional_tasks[{index}] must be a mapping")
        expected_task_keys = {"level", "task_name"}
        if set(value) != expected_task_keys:
            raise ValueError(
                f"additional_tasks[{index}] keys must be {sorted(expected_task_keys)}"
            )
        level = _strict_string(value, "level")
        task_name = _strict_string(value, "task_name")
        if level not in {"LEVEL1", "LEVEL2", "LEVEL3", "LEVEL4"}:
            raise ValueError(f"additional_tasks[{index}].level must be LEVEL1..LEVEL4")
        if (
            Path(task_name).name != task_name
            or task_name.endswith(".bddl")
            or ".." in task_name
            or "\\" in task_name
        ):
            raise ValueError(f"additional_tasks[{index}].task_name is unsafe")
        identity = (level, task_name)
        if identity in seen_tasks:
            raise ValueError(f"Duplicate additional task: {level}/{task_name}")
        seen_tasks.add(identity)
        parsed_tasks.append(AdditionalTaskConfig(level=level, task_name=task_name))
    base = path.parent
    config = UIConfig(
        host=_strict_string(raw, "host"),
        port=_strict_int(raw, "port"),
        dataset_root=_resolve(_strict_string(raw, "dataset_root"), base),
        project_id=_strict_string(raw, "project_id"),
        legacy_scan_roots=tuple(_resolve(value, base) for value in scan_roots),
        preview_width=_strict_int(raw, "preview_width"),
        preview_height=_strict_int(raw, "preview_height"),
        preview_fps=_strict_int(raw, "preview_fps"),
        jpeg_quality=_strict_int(raw, "jpeg_quality"),
        manual_translation_gain=_strict_number(raw, "manual_translation_gain"),
        manual_rotation_gain=_strict_number(raw, "manual_rotation_gain"),
        additional_tasks=tuple(parsed_tasks),
    )
    if config.host != "127.0.0.1":
        raise ValueError("The first UI version is local-only; host must be 127.0.0.1")
    if not 1 <= config.port <= 65535:
        raise ValueError("port must be in [1, 65535]")
    if (
        not config.project_id.replace("_", "").replace("-", "").isalnum()
        or Path(config.project_id).name != config.project_id
    ):
        raise ValueError("project_id may contain only letters, numbers, '-' and '_'")
    if not 64 <= config.preview_width <= 2048 or not 64 <= config.preview_height <= 2048:
        raise ValueError("preview dimensions must be in [64, 2048]")
    if not 1 <= config.preview_fps <= 60:
        raise ValueError("preview_fps must be in [1, 60]")
    if not 1 <= config.jpeg_quality <= 100:
        raise ValueError("jpeg_quality must be in [1, 100]")
    for key, value in (
        ("manual_translation_gain", config.manual_translation_gain),
        ("manual_rotation_gain", config.manual_rotation_gain),
    ):
        if not 0.05 <= value <= 1.0:
            raise ValueError(f"{key} must be in [0.05, 1.0]")
    return config
