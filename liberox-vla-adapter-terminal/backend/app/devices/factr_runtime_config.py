"""Shared official runtime paths for GUI and CLI, not a second control law."""
from dataclasses import asdict, dataclass
from pathlib import Path
import os

@dataclass(frozen=True)
class FactrRuntimeOptions:
    upstream_root: str
    runtime_python: str
    calibration_file: str

def parse_runtime_options(raw, config_dir: Path) -> FactrRuntimeOptions:
    if raw is not None and not isinstance(raw, dict):
        raise TypeError("runtime must be a mapping")
    values = dict(raw or {})
    defaults = {
        "upstream_root": "../third_party/FACTR_Teleop",
        "runtime_python": "../third_party/factr-runtime/bin/python",
        "calibration_file": "../runs/factr_calibration/calibration.json",
    }
    if set(values) - set(defaults):
        raise ValueError(f"Unknown runtime keys: {sorted(set(values)-set(defaults))}. "
                         "Custom gravity/safety tuning was replaced by the pinned official grav_comp_demo.yaml.")
    result = {}
    for key, default in defaults.items():
        value = values.get(key, default)
        if not isinstance(value, str) or not value.strip():
            raise TypeError(f"runtime.{key} must be a path")
        path = Path(value).expanduser()
        # Do not resolve the Python symlink: doing so bypasses the virtualenv.
        result[key] = os.path.abspath(path if path.is_absolute() else config_dir/path)
    return FactrRuntimeOptions(**result)

def serialized_options(raw, config_dir):
    return asdict(parse_runtime_options(raw, config_dir))
