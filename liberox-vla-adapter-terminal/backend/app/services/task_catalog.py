"""Validated, configurable LIBERO-X task catalog."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
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

    def metadata(self, init_state_count: int) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "level": self.level,
            "task_name": self.task_name,
            "prompt": self.prompt,
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
    ):
        self.runtime = runtime
        identities = [(eval_config.level, eval_config.task_name)]
        identities.extend((task.level, task.task_name) for task in additional_tasks)
        if len(set(identities)) != len(identities):
            raise ValueError("UI task catalog contains duplicate level/task_name entries")
        self._entries: dict[str, TaskEntry] = {}
        self._states: dict[str, Any] = {}
        for level, task_name in identities:
            task_id = self.make_task_id(level, task_name)
            bddl_path, init_path = direct.resolve_task(liberox_root, level, task_name)
            prompt = str(runtime.parse_bddl_file(str(bddl_path))["language"])
            if not prompt.strip():
                raise ValueError(f"Task has an empty BDDL prompt: {bddl_path}")
            self._entries[task_id] = TaskEntry(
                task_id, level, task_name, prompt, bddl_path, init_path
            )
        self.default_task_id = self.make_task_id(eval_config.level, eval_config.task_name)

    @staticmethod
    def make_task_id(level: str, task_name: str) -> str:
        return f"{level}::{task_name}"

    def resolve_id(self, level: str | None, task_name: str | None) -> str | None:
        if not level or not task_name:
            return None
        task_id = self.make_task_id(str(level), str(task_name))
        return task_id if task_id in self._entries else None

    def entry(self, task_id: str) -> TaskEntry:
        try:
            return self._entries[task_id]
        except KeyError as exc:
            raise ValueError(f"Unknown UI task_id: {task_id}") from exc

    def _load_states(self, task_id: str) -> Any:
        entry = self.entry(task_id)
        if task_id not in self._states:
            self._states[task_id] = direct.load_initial_states(self.runtime, entry.init_path)
        return self._states[task_id]

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
        return [self.metadata(task_id) for task_id in self._entries]
