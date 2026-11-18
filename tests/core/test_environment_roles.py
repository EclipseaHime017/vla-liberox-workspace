"""envs/roles.yaml and third_party/lock.yaml: every role resolves, and each installs only what it needs."""
from __future__ import annotations

import hashlib
import importlib.util
import re

import pytest
import yaml

from core.paths import ROOT, THIRD_PARTY
from core.upstream import LOCK_PATH, upstream_lock

_spec = importlib.util.spec_from_file_location("envs_setup", ROOT / "envs/setup.py")
setup = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(setup)
CONFIG = yaml.safe_load((ROOT / "envs/roles.yaml").read_text(encoding="utf-8"))
LOCK = yaml.safe_load(LOCK_PATH.read_text(encoding="utf-8"))
CHECKOUTS = {pin["checkout"] for pin in LOCK.values() if isinstance(pin, dict)}


def plan(*roles, families=("vla_adapter",)):
    units = setup.select_units(CONFIG, list(roles), list(families))
    return units, setup.plan_steps(CONFIG, LOCK, units)


def environments(steps):
    return {environment for _, environment, _, _ in steps}


def clones(steps):
    return {value for _, _, kind, value in steps if kind == "clone"}


@pytest.mark.parametrize("family", sorted(CONFIG["families"]))
@pytest.mark.parametrize("role", sorted(CONFIG["roles"]))
def test_every_role_resolves_to_declared_environments_files_and_upstreams(role, family):
    units, steps = plan(role, families=[family])
    assert units and steps
    for _, environment, kind, value in steps:
        assert environment in CONFIG["environments"]
        if kind in {"clone", "patch", "hf"}:
            assert value in LOCK
        if kind == "pip":
            for flag, argument in zip(value, value[1:]):
                if flag in {"-r", "-c"}:
                    assert (ROOT / argument).is_file(), argument
            for argument in value:
                if argument.startswith("third_party/"):
                    assert argument.split("/")[1].split("[")[0] in CHECKOUTS, argument


@pytest.mark.parametrize("role", ["train", "hpc", "infer"])
@pytest.mark.parametrize("family,environment,upstream",
                         [("vla_adapter", "vla-liberox", "vla_adapter"), ("pi05", "pi05", "openpi")])
def test_training_roles_skip_platform_simulator_teleop_and_reward_models(role, family, environment, upstream):
    units, steps = plan(role, families=[family])
    assert environments(steps) == {environment}
    assert clones(steps) == {upstream}
    assert not {"sim", "studio", "test", "web", "libero-x", "spacemouse", "factr", "rynnvalue", "robometer"} & set(units)


def test_eval_adds_only_the_simulator_to_the_family_environment():
    units, steps = plan("eval", families=["pi05"])
    assert environments(steps) == {"pi05"}
    assert clones(steps) == {"openpi", "libero_x"}
    assert not [unit for unit in units if unit.endswith("-weights")]


def test_reward_roles_use_only_their_own_environment_and_model():
    assert environments(plan("reward-rynnvalue")[1]) == {"rynnvalue-reward"}
    assert clones(plan("reward-rynnvalue")[1]) == {"rynnvalue"}
    assert environments(plan("reward-robometer")[1]) == {"robometer-reward"}
    assert clones(plan("reward-robometer")[1]) == {"robometer"}


def test_reward_environments_are_started_from_the_root_and_never_install_the_repository():
    from_root = {name for name, spec in CONFIG["environments"].items() if not setup.installs_repository(spec)}
    assert from_root == {"rynnvalue-reward", "robometer-reward"}
    for unit, spec in CONFIG["units"].items():
        for step in spec["steps"]:
            (kind, value), = step.items()
            if spec["environment"] in from_root:
                assert not (kind == "pip" and "." in value), (unit, step)
    with pytest.raises(SystemExit, match="installed or from-root"):
        setup.installs_repository({"repository": "copied"})


def test_verify_imports_isolated_where_installed_and_from_the_root_elsewhere(monkeypatch, capsys):
    calls = []
    monkeypatch.setenv("CONDA_EXE", "conda")
    monkeypatch.setattr(setup.subprocess, "run", lambda command, **kwargs: calls.append((command, kwargs)))
    installer = setup.Installer(CONFIG, LOCK, dry_run=False)
    installer.ready.update(CONFIG["environments"])
    installer.run_step("vla-liberox", "verify", "packages")
    installer.run_step("rynnvalue-reward", "verify", "packages")
    (installed, installed_kwargs), (from_root, from_root_kwargs) = calls
    assert installed[installed.index("python") + 1] == "-I"
    assert from_root[from_root.index("python") + 1] == "-E"
    assert installed_kwargs["cwd"] == from_root_kwargs["cwd"] == ROOT
    assert "repository root (not installed)" in capsys.readouterr().out


def test_dev_downloads_no_weights_and_studio_adds_pi05_only_when_asked():
    assert not [unit for unit in plan("dev")[0] if unit.endswith("-weights")]
    assert environments(plan("studio")[1]) == {"vla-liberox"}
    assert environments(plan("studio", families=["vla_adapter", "pi05"])[1]) == {"vla-liberox", "pi05"}


def test_dry_run_prints_the_plan_and_runs_nothing(monkeypatch, capsys):
    monkeypatch.setattr(setup.subprocess, "run", lambda *_, **__: pytest.fail("dry run executed a command"))
    assert setup.main(["--role", "studio", "--role", "teleop", "--dry-run"]) == 0
    output = capsys.readouterr().out
    assert "FACTR_Teleop" in output and "Manual steps left" in output and "Dry run only" in output


def test_unknown_roles_and_families_are_rejected():
    with pytest.raises(SystemExit, match="Unknown role"):
        setup.select_units(CONFIG, ["gpu"], [])
    with pytest.raises(SystemExit, match="Unknown family"):
        setup.select_units(CONFIG, ["train"], ["openvla"])


def test_lock_pins_every_upstream_and_the_recorded_patch():
    upstreams = {name: pin for name, pin in LOCK.items() if name != "schema_version"}
    assert set(upstreams) == {"libero_x", "vla_adapter", "openpi", "rynnvalue", "robometer", "factr"}
    for pin in upstreams.values():
        assert pin["repository"].startswith("https://") and re.fullmatch(r"[0-9a-f]{40}", pin["commit"])
    patch = upstream_lock("vla_adapter")["patch"]
    assert hashlib.sha256((THIRD_PARTY / patch["path"]).read_bytes()).hexdigest() == patch["sha256"]


def test_stdlib_openpi_pin_matches_the_lock():
    from core.pi05_assets import OPENPI_COMMIT, OPENPI_REPOSITORY
    pin = upstream_lock("openpi")
    assert (OPENPI_REPOSITORY, OPENPI_COMMIT, THIRD_PARTY / pin["checkout"]) == (
        pin["repository"], pin["commit"], THIRD_PARTY / "OpenPI")


def test_robometer_constraints_repeat_the_locked_runtime():
    runtime = upstream_lock("robometer")["runtime"]
    lines = (ROOT / "envs/robometer-constraints.txt").read_text(encoding="utf-8").splitlines()
    pinned = dict(line.split("==") for line in lines if "==" in line and not line.startswith("#"))
    assert pinned == {name: str(runtime[name]) for name in pinned}
