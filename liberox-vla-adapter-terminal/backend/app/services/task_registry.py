"""Declarative tasks and immutable scene snapshots; no simulator ownership."""
from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import threading

import yaml

from ..core.config import UniqueKeyLoader
from .task_catalog import ConfiguredTaskCatalog, TaskEntry


def _string(value, name):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _read(path):
    raw = yaml.load(path.read_text(encoding="utf-8"), Loader=UniqueKeyLoader)
    if not isinstance(raw, dict) or raw.get("schema_version") != 1:
        raise ValueError(f"Invalid task configuration: {path}")
    return raw


def _path(value, base):
    path = Path(_string(value, "resource path")).expanduser()
    return (path if path.is_absolute() else base / path).resolve()


def _sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class RegistryTaskEntry(TaskEntry):
    instructions: tuple = ()
    bddl_sha256: str = ""
    init_sha256: str = ""
    task_revision_sha256: str = ""

    def metadata(self, init_state_count):
        return {**super().metadata(init_state_count), "goal_prompt": self.prompt,
                "instructions": [dict(item) for item in self.instructions],
                "bddl_sha256": self.bddl_sha256, "init_sha256": self.init_sha256,
                "task_revision_sha256": self.task_revision_sha256}


def _identity(entry):
    return dict(task_id=entry.task_id, level=entry.level, task_name=entry.task_name,
                goal_prompt=entry.prompt, prompt_variant=entry.prompt_variant,
                family_id=entry.family_id, family_label=entry.family_label,
                instructions=list(entry.instructions), bddl_sha256=entry.bddl_sha256,
                init_sha256=entry.init_sha256)


def _definition(meta):
    goal = _string(meta.get("goal_prompt", meta.get("prompt")), "goal_prompt")
    instructions = meta.get("instructions", [{"id": "full", "text": goal}])
    if not isinstance(instructions, list) or not instructions:
        raise ValueError("instructions must be a non-empty list")
    for instruction in instructions:
        if not isinstance(instruction, dict) or set(instruction) != {"id", "text"}:
            raise ValueError("instructions require id and text")
        _string(instruction["id"], "instruction.id")
        _string(instruction["text"], "instruction.text")
    if len({item["id"] for item in instructions}) != len(instructions):
        raise ValueError("Duplicate instruction id")
    if meta.get("level") not in {f"LEVEL{i}" for i in range(1, 6)}:
        raise ValueError("Task level must be LEVEL1..LEVEL5")
    name = _string(meta.get("task_name"), "task_name")
    if Path(name).name != name or ".." in name or "\\" in name:
        raise ValueError("Unsafe task_name")
    _string(meta.get("task_id"), "task_id")
    _string(meta.get("family_id"), "family_id")
    _string(meta.get("family_label"), "family_label")
    return goal, instructions, name


def _entry(meta, bddl, init):
    goal, instructions, name = _definition(meta)
    entry = RegistryTaskEntry(_string(meta.get("task_id"), "task_id"), meta["level"], name,
        goal, bddl, init, meta.get("prompt_variant"),
        _string(meta.get("family_id"), "family_id"), _string(meta.get("family_label"), "family_label"),
        tuple(dict(item) for item in instructions), _sha(bddl), _sha(init))
    digest = hashlib.sha256(json.dumps(_identity(entry), sort_keys=True,
        separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()
    return replace(entry, task_revision_sha256=digest)


def _initialize(catalog, runtime):
    catalog.runtime = runtime
    catalog._entries, catalog._unavailable, catalog._states = {}, {}, {}
    catalog._state_lock = threading.RLock()
    catalog._task_ids = []


class RegistryTaskCatalog(ConfiguredTaskCatalog):
    def __init__(self, runtime, registry_path):
        _initialize(self, runtime)
        path = Path(registry_path).resolve()
        registry = _read(path)
        if set(registry) != {"schema_version", "default_task_id", "families"}:
            raise ValueError("Task registry requires schema_version/default_task_id/families")
        if not isinstance(registry["families"], dict) or not registry["families"]:
            raise ValueError("Task registry has no families")
        choices = set()
        for family_id, source in registry["families"].items():
            family_path = _path(source, path.parent)
            family = _read(family_path)
            if (set(family) != {"schema_version", "family_id", "label", "variants"}
                    or family["family_id"] != family_id or not isinstance(family["variants"], dict)
                    or not family["variants"]):
                raise ValueError(f"Invalid task family: {family_path}")
            for value in family["variants"].values():
                required = {"task_id", "level", "task_name", "goal_prompt", "bddl", "init"}
                if (not isinstance(value, dict) or not required <= value.keys()
                        or value.keys() - required - {"optional", "prompt_variant", "instructions"}):
                    raise ValueError(f"Invalid task variant: {family_path}")
                task_id = _string(value["task_id"], "task_id")
                if task_id in self._task_ids:
                    raise ValueError(f"Duplicate task_id: {task_id}")
                if type(value.get("optional", False)) is not bool:
                    raise ValueError("optional must be boolean")
                choice = (family_id, value["level"], value["goal_prompt"])
                if choice in choices:
                    raise ValueError(f"Ambiguous task family/level/prompt: {choice}")
                choices.add(choice)
                self._task_ids.append(task_id)
                meta = {**value, "family_id": family_id, "family_label": family["label"]}
                _definition(meta)
                try:
                    bddl, init = (_path(value[key], family_path.parent) for key in ("bddl", "init"))
                    entry = _entry(meta, bddl, init)
                    runtime.parse_bddl_file(str(bddl))
                    self._entries[task_id] = entry
                    self._load_states(task_id)
                except Exception as exc:
                    self._entries.pop(task_id, None)
                    if not value.get("optional", False):
                        raise
                    self._unavailable[task_id] = dict(task_id=task_id, level=value["level"],
                        task_name=value["task_name"], prompt=value["goal_prompt"],
                        family_id=family_id, family_label=family["label"], available=False,
                        unavailable_reason=str(exc), init_state_count=0,
                        init_state_index_min=0, init_state_index_max=-1)
        self.default_task_id = _string(registry["default_task_id"], "default_task_id")
        self.entry(self.default_task_id)


class TaskSnapshot(ConfiguredTaskCatalog):
    @classmethod
    def load(cls, runtime, manifest_path):
        path = Path(manifest_path)
        if path.is_dir():
            path = path / "task.json"
        if path.is_symlink():
            raise ValueError("Task snapshot manifest must not be a symlink")
        manifest = json.loads(path.read_text(encoding="utf-8"))
        if manifest.get("schema_version") != 1:
            raise ValueError("Invalid task snapshot version")
        paths = []
        for key in ("bddl", "init"):
            relative = Path(manifest["files"][key])
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError("Task snapshot resource escapes its directory")
            resource = path.parent / relative
            if resource.is_symlink():
                raise ValueError("Task snapshot resource must not be a symlink")
            paths.append(resource.resolve())
        entry = _entry(manifest["task"], *paths)
        if entry.task_revision_sha256 != manifest["task_revision_sha256"]:
            raise ValueError("Task snapshot hash mismatch")
        result = cls.__new__(cls)
        _initialize(result, runtime)
        result.manifest_path, result.manifest = path.resolve(), manifest
        result.default_task_id = entry.task_id
        result._entries[entry.task_id] = entry
        result._task_ids.append(entry.task_id)
        return result


def freeze_task(catalog, task_id, directory):
    meta = catalog.metadata(task_id)
    bddl, init = catalog.paths(task_id)
    current = _entry(meta, bddl, init)
    expected = getattr(catalog.entry(task_id), "task_revision_sha256", "")
    if expected and expected != current.task_revision_sha256:
        raise ValueError("Task resources changed; reload the task registry")
    directory = Path(directory)
    if directory.exists():
        existing = TaskSnapshot.load(catalog.runtime, directory)
        if existing.manifest["task_revision_sha256"] != current.task_revision_sha256:
            raise ValueError("Conflicting task snapshot")
        return existing
    directory.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".task-", dir=directory.parent))
    try:
        shutil.copyfile(bddl, temporary / "task.bddl")
        shutil.copyfile(init, temporary / "initial_states.init")
        frozen = _entry(meta, temporary / "task.bddl", temporary / "initial_states.init")
        if frozen.task_revision_sha256 != current.task_revision_sha256:
            raise ValueError("Task resources changed while freezing")
        manifest = dict(schema_version=1, task_revision_sha256=frozen.task_revision_sha256,
            task=_identity(frozen), files={"bddl": "task.bddl", "init": "initial_states.init"})
        (temporary / "task.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
        staged = TaskSnapshot.load(catalog.runtime, temporary / "task.json")
        staged.initial_state_count(task_id)
        temporary.rename(directory)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return TaskSnapshot.load(catalog.runtime, directory / "task.json")
