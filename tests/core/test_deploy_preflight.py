"""skills/deploy/scripts/preflight.py: read-only checks that report what a deployment still needs."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
import sys

import pytest

from core.paths import ROOT

_spec = importlib.util.spec_from_file_location("deploy_preflight", ROOT / "skills/deploy/scripts/preflight.py")
preflight = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(preflight)
READ_ONLY_GIT = {"rev-parse", "status", "rev-list"}


def git(path, *args):
    return subprocess.run(["git", "-C", str(path), *args], check=True, capture_output=True, text=True).stdout.strip()


@pytest.fixture
def checkout(tmp_path):
    root = tmp_path / "Upstream"
    root.mkdir()
    git(root, "init", "-q")
    (root / "module.py").write_text("x = 1\n")
    git(root, "add", ".")
    git(root, "-c", "user.name=Tests", "-c", "user.email=tests@example.invalid", "commit", "-qm", "base")
    return tmp_path, {"repository": "https://example.invalid/upstream.git", "checkout": "Upstream",
                      "commit": git(root, "rev-parse", "HEAD")}


def test_pinned_clean_checkout_passes_and_bytecode_is_ignored(checkout):
    third_party, pin = checkout
    (third_party / "Upstream/__pycache__").mkdir()
    (third_party / "Upstream/__pycache__/module.cpython-310.pyc").write_bytes(b"compiled")
    assert preflight.check_upstream("upstream", pin, third_party)["status"] == "ok"


def test_modified_checkout_warns_and_wrong_or_missing_checkout_fails(checkout):
    third_party, pin = checkout
    (third_party / "Upstream/module.py").write_text("x = 2\n")
    assert preflight.check_upstream("upstream", pin, third_party)["status"] == "warn"
    assert preflight.check_upstream("upstream", {**pin, "commit": "0" * 40}, third_party)["status"] == "fail"
    assert preflight.check_upstream("upstream", {**pin, "checkout": "Missing"}, third_party)["status"] == "fail"


def test_locked_official_files_are_hashed(checkout):
    third_party, pin = checkout
    digest = hashlib.sha256(b"x = 1\n").hexdigest()
    assert preflight.check_upstream("upstream", {**pin, "files": {"module.py": digest}}, third_party)["status"] == "ok"
    assert preflight.check_upstream("upstream", {**pin, "files": {"module.py": "0" * 64}}, third_party)["status"] == "fail"


def test_roles_select_only_their_environments_and_upstreams():
    setup = preflight.load_setup()
    config = setup.load_yaml(setup.ROLES)
    assert preflight.required(config, setup, ["train"], ["pi05"]) == (["pi05"], ["openpi"], False)
    environments, upstreams, web = preflight.required(config, setup, ["dev"], [])
    assert environments == ["vla-liberox"] and set(upstreams) == {"vla_adapter", "libero_x"} and web


def probe(monkeypatch, *, installed, stale=None, packages_from=None):
    """Fake one environment probe; returns the python flag and working directory it ran with."""
    seen = {}
    origins = {name: None if packages_from is None else str(packages_from / name / "__init__.py")
               for name in preflight.PACKAGES}

    def run(command, **kwargs):
        seen.update(flag=command[command.index("python") + 1], cwd=kwargs.get("cwd"))
        report = {"python": "3.10.0", "packages": origins, "stale": stale or {}, "installed": installed}
        return subprocess.CompletedProcess(command, 0, json.dumps(report) + "\n", "")

    monkeypatch.setattr(preflight.subprocess, "run", run)
    return seen


def test_installed_environment_is_probed_isolated_and_must_have_the_repository(monkeypatch, tmp_path):
    seen = probe(monkeypatch, installed="0.8.5", packages_from=tmp_path)
    assert preflight.check_environment("conda", "vla-liberox", tmp_path)["status"] == "ok"
    assert seen == {"flag": "-I", "cwd": tmp_path}
    probe(monkeypatch, installed=None)
    missing = preflight.check_environment("conda", "vla-liberox", tmp_path)
    assert missing["status"] == "warn" and "pip install --no-deps -e ." in missing["hint"]


def test_from_root_environment_must_import_from_the_root_without_installing(monkeypatch, tmp_path):
    seen = probe(monkeypatch, installed=None, packages_from=tmp_path)
    assert preflight.check_environment("conda", "rynnvalue-reward", tmp_path, installed=False)["status"] == "ok"
    assert seen == {"flag": "-E", "cwd": tmp_path}
    probe(monkeypatch, installed="0.8.5", packages_from=tmp_path)
    extra = preflight.check_environment("conda", "rynnvalue-reward", tmp_path, installed=False)
    assert extra["status"] == "warn" and extra["hint"].endswith("pip uninstall -y vla-liberox")
    probe(monkeypatch, installed=None, packages_from=tmp_path, stale={"vla-adapter-robometer": "0.8.0"})
    stale = preflight.check_environment("conda", "robometer-reward", tmp_path, installed=False)
    assert stale["status"] == "warn" and "install" not in stale["hint"].replace("uninstall", "")
    probe(monkeypatch, installed=None, packages_from=tmp_path / "elsewhere")
    assert preflight.check_environment("conda", "rynnvalue-reward", tmp_path, installed=False)["status"] == "fail"


def test_probe_ignores_build_leftovers_in_the_working_directory(tmp_path):
    leftover = tmp_path / "vla_rynn_iql.egg-info"
    leftover.mkdir()
    (leftover / "PKG-INFO").write_text("Metadata-Version: 2.1\nName: vla-rynn-iql\nVersion: 0.8.0\n")
    output = subprocess.run([sys.executable, "-E", "-c", preflight.PROBE], cwd=tmp_path,
                            check=True, capture_output=True, text=True).stdout
    assert "vla-rynn-iql" not in json.loads(output.splitlines()[-1])["stale"]


def test_missing_conda_fails_and_only_read_only_git_commands_run(monkeypatch, capsys):
    real_run = subprocess.run

    def guarded(command, *args, **kwargs):
        assert command[0] == "git", command
        action = command[3]
        assert action in READ_ONLY_GIT or (action == "apply" and "--check" in command), command
        return real_run(command, *args, **kwargs)

    monkeypatch.delenv("CONDA_EXE", raising=False)
    monkeypatch.setattr(preflight.shutil, "which", lambda _: None)
    monkeypatch.setattr(preflight.subprocess, "run", guarded)
    assert preflight.main(["--role", "dev", "--json"]) == 1
    report = json.loads(capsys.readouterr().out)
    assert {"area": "environments", "item": "conda", "status": "fail"}.items() <= next(
        item for item in report["results"] if item["item"] == "conda").items()
    assert report["results"][0]["item"] == "version"
