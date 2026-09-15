"""Validated, configurable LIBERO-X task catalog."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re
import threading
from types import SimpleNamespace
from typing import Any, Protocol, Sequence

import eval_pickplace_direct as direct


class TaskCatalog(Protocol):
    default_task_id: str
    def metadata(self, task_id: str) -> dict[str, Any]: ...
    def list_tasks(self) -> list[dict[str, Any]]: ...
    def paths(self, task_id: str) -> tuple[Path, Path]: ...
    def initial_state(self, task_id: str, index: int = 0) -> Any: ...
    def initial_state_count(self, task_id: str) -> int: ...


@dataclass(frozen=True)
class TaskEntry:
    task_id: str
    level: str
    task_name: str
    prompt: str
    bddl_path: Path
    init_path: Path
    prompt_variant: str | None = None
    family_id: str = ""
    family_label: str = ""

    def metadata(self, init_state_count: int) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "level": self.level,
            "task_name": self.task_name,
            "prompt": self.prompt,
            "family_id": self.family_id or self.task_name,
            "family_label": self.family_label or self.prompt,
            "prompt_variant": self.prompt_variant,
            "scene_level": "LEVEL4" if self.level == "LEVEL5" else self.level,
            "available": True,
            "init_state_count": init_state_count,
            "init_state_index_min": 0,
            "init_state_index_max": init_state_count - 1,
        }


class ConfiguredTaskCatalog:
    """Resolve BDDL and init-state assets without owning a simulator."""

    def __init__(
        self,
        runtime: SimpleNamespace,
        liberox_root: Path,
        eval_config: direct.EvalConfig,
        additional_tasks: Sequence[Any],
        task_families: Sequence[Any] = (),
    ):
        self.runtime = runtime
        families = {name: family for family in task_families for name in family.task_names}
        identities = [(eval_config.level, eval_config.task_name, None, False)]
        identities.extend((task.level, task.task_name, variant, task.optional)
                          for task in additional_tasks for variant in (task.prompt_variants or (None,)))
        if len({identity[:3] for identity in identities}) != len(identities):
            raise ValueError("UI task catalog contains duplicate level/task_name entries")
        self._entries: dict[str, TaskEntry] = {}
        self._unavailable: dict[str, dict[str, Any]] = {}
        self._states: dict[Path, Any] = {}
        self._state_lock = threading.RLock()
        self._task_ids: list[str] = []
        prompts: dict[str, dict[tuple[str, str], str]] = {}
        scene_prompts: dict[Path, str] = {}
        for level, task_name, variant, optional in identities:
            family = families.get(task_name)
            family_id = family.family_id if family else task_name
            family_label = family.label if family else ""
            task_id = self.make_task_id(level, task_name, variant)
            self._task_ids.append(task_id)
            try:
                scene_level = "LEVEL4" if level == "LEVEL5" else level
                bddl_path, init_path = direct.resolve_task(liberox_root, scene_level, task_name)
                if bddl_path not in scene_prompts:
                    scene_prompts[bddl_path] = str(runtime.parse_bddl_file(str(bddl_path))["language"])
                prompt = scene_prompts[bddl_path]
                if variant:
                    if variant not in prompts:
                        prompts[variant] = self._level5_prompts(liberox_root, variant)
                    match = re.search(r"__T(\d+)(?:__A(\d+))?", task_name)
                    if match is None:
                        raise ValueError(f"Cannot resolve LEVEL5 task key: {task_name}")
                    key = (match[1].zfill(3), f"A{match[2]}" if match[2] else "")
                    if key not in prompts[variant]:
                        raise ValueError(f"Missing {variant} prompt for {key}")
                    prompt = prompts[variant][key]
                if not prompt.strip():
                    raise ValueError(f"Task has an empty prompt: {task_id}")
                self._entries[task_id] = TaskEntry(
                    task_id, level, task_name, prompt, bddl_path, init_path, variant,
                    family_id, family_label,
                )
                # Validate small CPU init arrays once, before serving requests.
                # Model weights and MuJoCo environments stay lazy.
                self._load_states(task_id)
            except (OSError, ValueError) as exc:
                if not optional:
                    raise
                self._entries.pop(task_id, None)
                self._unavailable[task_id] = dict(task_id=task_id, level=level,
                    task_name=task_name, prompt=task_name, prompt_variant=variant,
                    family_id=family_id, family_label=family_label or task_name,
                    available=False, unavailable_reason=str(exc),
                    init_state_count=0, init_state_index_min=0, init_state_index_max=-1)
        self.default_task_id = self.make_task_id(eval_config.level, eval_config.task_name)
        choices: set[tuple[str, str, str]] = set()
        for entry in self._entries.values():
            choice = (entry.family_id, entry.level, entry.prompt)
            if choice in choices:
                raise ValueError(f"Ambiguous task family/level/prompt: {choice}")
            choices.add(choice)

    @staticmethod
    def _level5_prompts(root: Path, variant: str) -> dict[tuple[str, str], str]:
        path = root / "libero" / "libero_x" / "LEVEL5" / f"{variant}.jsonl"
        result = {}
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            if not isinstance(record, dict) or not isinstance(record.get("task_id"), (str, int)):
                raise ValueError(f"Invalid LEVEL5 prompt record: {path.name}")
            if record.get("variant") is not None and not isinstance(record["variant"], str):
                raise ValueError(f"Invalid LEVEL5 task variant: {path.name}")
            key = (str(record["task_id"]).zfill(3), record.get("variant") or "")
            if key in result or not isinstance(record.get("task_desc"), str) or not record["task_desc"].strip():
                raise ValueError(f"Duplicate/invalid LEVEL5 prompt: {path.name} {key}")
            result[key] = record["task_desc"]
        return result

    @staticmethod
    def make_task_id(level: str, task_name: str, variant: str | None = None) -> str:
        return f"{level}::{task_name}" + (f"::{variant}" if variant else "")

    def resolve_id(self, level: str | None, task_name: str | None) -> str | None:
        if not level or not task_name:
            return None
        task_id = self.make_task_id(str(level), str(task_name))
        return task_id if task_id in self._entries else None

    def entry(self, task_id: str) -> TaskEntry:
        if task_id in self._unavailable:
            raise ValueError(f"Task unavailable: {self._unavailable[task_id]['unavailable_reason']}")
        try:
            return self._entries[task_id]
        except KeyError as exc:
            raise ValueError(f"Unknown UI task_id: {task_id}") from exc

    def _load_states(self, task_id: str) -> Any:
        entry = self.entry(task_id)
        # Language variants share one state array; no simulator is constructed.
        with self._state_lock:
            if entry.init_path not in self._states:
                self._states[entry.init_path] = direct.load_initial_states(self.runtime, entry.init_path)
            return self._states[entry.init_path]

    def initial_state_count(self, task_id: str) -> int:
        return len(self._load_states(task_id))

    def initial_state(self, task_id: str, index: int = 0) -> Any:
        states = self._load_states(task_id)
        if isinstance(index, bool) or not isinstance(index, int):
            raise ValueError("init_state_index must be an integer")
        if not 0 <= index < len(states):
            raise ValueError(
                f"init_state_index for {task_id} must be in [0, {len(states) - 1}], "
                f"got {index}"
            )
        return states[index]

    def paths(self, task_id: str) -> tuple[Path, Path]:
        entry = self.entry(task_id)
        return entry.bddl_path, entry.init_path

    def metadata(self, task_id: str) -> dict[str, Any]:
        return self.entry(task_id).metadata(self.initial_state_count(task_id))

    def list_tasks(self) -> list[dict[str, Any]]:
        return [dict(self._unavailable[task_id]) if task_id in self._unavailable
                else self.metadata(task_id) for task_id in self._task_ids]
