import json
import os
from pathlib import Path
import signal
import subprocess
import sys

import pytest
import yaml

from hpc import pbs
from vla_rynn_iql import server_resources
from vla_rynn_iql.config import PROJECT_ROOT
from vla_rynn_iql.server_config import load_server_config
from vla_rynn_iql.server_pipeline import training_settings
from vla_rynn_iql.server_resources import TrainingBusy, TrainingLease, register_worker
from test_server_inputs import global_runs
from test_server_pipeline import configuration


GPUS = tuple(f"GPU-00000000-0000-0000-0000-{index:012d}" for index in range(8))


def test_default_paths_and_cli_are_independent_of_the_working_directory(tmp_path):
    path, raw = pbs.load_pbs_config(PROJECT_ROOT / "hpc/configs/pbs_pipeline.yaml")
    assert Path(raw["output_root"]) == PROJECT_ROOT / "outputs/pbs"
    for entry in raw["experiments"]:
        assert (path.parent / entry["config"]).resolve() == PROJECT_ROOT / "configs/server_pipeline.yaml"
        assert (path.parent / entry["runs_root"]).resolve() == PROJECT_ROOT.parent / "dataset-root/imports/task-export"
    result = subprocess.run([sys.executable, str(PROJECT_ROOT / "hpc/train_pbs.py"), "--help"],
                            cwd=tmp_path, check=True, capture_output=True, text=True)
    assert "prepare" in result.stdout and "submit" in result.stdout


@pytest.fixture
def pbs_request(configured, tmp_path, monkeypatch):
    task, _, _ = global_runs(configured)
    server = configuration(tmp_path, task, configured)
    raw = yaml.safe_load((PROJECT_ROOT / "hpc/configs/pbs_pipeline.yaml").read_text())
    raw.update(project="personal-test", output_root=str(tmp_path / "pbs"), gpus=8)
    for index, entry in enumerate(raw["experiments"]):
        entry.update(config=str(server.path), runs_root=str(server.runs_root))
        entry["gpu_ids"] = list(range(index * 4, index * 4 + 4))
    path = tmp_path / "pbs.yaml"
    path.write_text(yaml.safe_dump(raw))
    monkeypatch.setattr(pbs.shutil, "which", lambda _: "/opt/conda/bin/conda")
    return path, raw, task


def test_parallel_request_freezes_independent_model_method_reward_and_scale(pbs_request):
    path, raw, task = pbs_request
    raw["experiments"][1].update(model="vla-adapter-pro-liberox-base", method="bc", reward_source="sparse")
    path.write_text(yaml.safe_dump(raw))
    directory, manifest = pbs.stage_submission(path)
    assert not (directory / "submission.json").exists()
    for plan, expected in zip(manifest["experiments"], raw["experiments"]):
        server = load_server_config(Path(plan["config"]))
        settings = training_settings(server, task)
        assert settings["model"]["base_id"] == expected["model"]
        assert settings["training"]["method"] == expected["method"]
        assert settings["training"]["micro_batch_size"] == 64
        assert settings["training"]["train_steps"] == 10000
        assert settings["training"]["gradient_accumulation_steps"] == 1
        assert server.reward_source == expected["reward_source"]
        assert tuple(expected["gpu_ids"]) == server.distributed.gpu_ids
    path.write_text("invalid: now")
    assert load_server_config(Path(manifest["experiments"][0]["config"]))
    script = (directory / "job.pbs").read_text()
    assert "select=1:ngpus=8" in script and "#PBS -q normal@pbs101" in script
    assert str(PROJECT_ROOT / "hpc/train_pbs.py") in script
    assert "train_queue" not in script and "CUDA_VISIBLE_DEVICES=" not in script
    subprocess.run(["bash", "-n", str(directory / "job.pbs")], check=True)


@pytest.mark.parametrize("change", [
    {"queue": "aiq5"}, {"walltime": "25:00:00"}, {"walltime": "00:00:10"},
    {"gpus": True}, {"project": "x\n#PBS -q internal"}, {"extra": True},
])
def test_invalid_pbs_settings_are_rejected(pbs_request, change):
    path, raw, _ = pbs_request
    raw.update(change)
    path.write_text(yaml.safe_dump(raw))
    with pytest.raises((ValueError, TypeError)):
        pbs.load_pbs_config(path)


@pytest.mark.parametrize("ids", [[0, 1, 2, 3], [4, 5, 6, 8], [4, 4], []])
def test_gpu_groups_cannot_overlap_escape_or_leave_idle_devices(pbs_request, ids):
    path, raw, _ = pbs_request
    raw["experiments"][1]["gpu_ids"] = ids
    path.write_text(yaml.safe_dump(raw))
    with pytest.raises(ValueError):
        pbs.load_pbs_config(path)


def test_pbs_configs_retain_server_reward_override_restrictions(pbs_request):
    path, raw, _ = pbs_request
    raw["experiments"][0]["overrides"]["reward"] = {"shaping_weight": .1}
    path.write_text(yaml.safe_dump(raw))
    with pytest.raises(ValueError, match="reward overrides"):
        pbs.stage_submission(path)


@pytest.mark.parametrize("source,target", [("bc", "iql"), ("iql", "bc")])
def test_method_switch_preserves_direct_configuration_defaults(pbs_request, source, target):
    path, raw, _ = pbs_request
    server_path = Path(raw["experiments"][0]["config"])
    server = yaml.safe_load(server_path.read_text())
    server["training_config"] = str(PROJECT_ROOT / f"configs/training/{source}.yaml")
    server["overrides"]["training"]["method"] = source
    server_path.write_text(yaml.safe_dump(server))
    raw["experiments"][0]["method"] = target
    path.write_text(yaml.safe_dump(raw))
    directory, manifest = pbs.stage_submission(path)
    frozen = load_server_config(Path(manifest["experiments"][0]["config"]))
    cfg = pbs.load_train_config(frozen.training_config).raw
    expected = pbs.load_train_config(PROJECT_ROOT / f"configs/training/{target}.yaml").raw
    assert cfg["training"]["method"] == target
    for section in ("iql", "bc"):
        assert cfg[section] == expected[section]


def allocation(tmp_path, mask):
    nodes = tmp_path / "nodes"
    nodes.write_text("gpu-node\ngpu-node\n")
    return {"PBS_JOBID": "123.pbs102", "PBS_NODEFILE": str(nodes), "CUDA_VISIBLE_DEVICES": mask}


def test_allocation_uses_only_assigned_devices_and_canonical_uuids(tmp_path):
    query = "\n".join(GPUS)
    env = allocation(tmp_path, ",".join(GPUS[4:]))
    assert pbs.allocated_devices(4, environ=env, query=query) == GPUS[4:]
    for mask in ("", "0,1,2,3", ",".join([GPUS[0]] * 4), "-1", "GPU-short", "MIG-invalid"):
        env["CUDA_VISIBLE_DEVICES"] = mask
        with pytest.raises(ValueError):
            pbs.allocated_devices(4, environ=env, query=query)
    env = allocation(tmp_path, "0,1,2,3")
    Path(env["PBS_NODEFILE"]).write_text("node-a\nnode-b\n")
    with pytest.raises(ValueError, match="single-node"):
        pbs.allocated_devices(4, environ=env, query=query)
    with pytest.raises(ValueError, match="inside a PBS"):
        pbs.allocated_devices(4, environ={})


def test_scoped_leases_are_disjoint_and_legacy_exclusion_is_preserved(tmp_path, monkeypatch):
    monkeypatch.setattr(server_resources, "LEASE_ROOT", tmp_path / "leases")
    with TrainingLease():
        with pytest.raises(TrainingBusy):
            TrainingLease(GPUS[:4])
    with TrainingLease(GPUS[:4]) as first, TrainingLease(GPUS[4:]) as second:
        assert first.token != second.token and len(first.fds) == 5
        with pytest.raises(TrainingBusy):
            TrainingLease(GPUS[3:5])
        with pytest.raises(TrainingBusy):
            TrainingLease()
        register_worker(first.token)
        register_worker(second.token)
        first.retire()
        with pytest.raises(TrainingBusy, match="lease changed"):
            register_worker(first.token)
        register_worker(second.token)
        assert len(first.workers()) == 1 and len(second.workers()) == 2
        # Test workers are this process; erase their fake registration before closing.
        with server_resources._members() as state:
            for record in state["scopes"].values():
                record["workers"] = []
    with TrainingLease():
        pass
    with pytest.raises(ValueError, match="UUID"):
        TrainingLease(("0", "1"))


@pytest.mark.parametrize("register", [False, True])
def test_scoped_orphan_or_inherited_locks_only_block_the_same_gpus(tmp_path, monkeypatch, register):
    monkeypatch.setattr(server_resources, "LEASE_ROOT", tmp_path / "leases")
    lease = TrainingLease(GPUS[:4])
    code = """
import sys
from pathlib import Path
from vla_rynn_iql import server_resources as resources
resources.LEASE_ROOT=Path(sys.argv[1])
if sys.argv[3]=='yes':resources.register_worker(sys.argv[2])
print('ready',flush=True)
sys.stdin.read()
"""
    child = subprocess.Popen([sys.executable, "-B", "-c", code, str(server_resources.LEASE_ROOT), lease.token,
                              "yes" if register else "no"],
                             pass_fds=() if register else lease.fds,
                             env={**os.environ, "PYTHONPATH": str(PROJECT_ROOT / "src") + os.pathsep +
                                  os.environ.get("PYTHONPATH", "")},
                             stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
    try:
        assert child.stdout.readline().strip() == "ready"
        lease.close()
        with pytest.raises(TrainingBusy):
            TrainingLease(GPUS[:4])
        with TrainingLease(GPUS[4:]):
            pass
        with pytest.raises(TrainingBusy):
            TrainingLease()
        child.communicate("", timeout=10)
        with TrainingLease(GPUS[:4]):
            pass
    finally:
        lease.close()
        if child.poll() is None:
            child.kill()
        child.wait(timeout=10)


@pytest.mark.parametrize("failure", [False, True])
def test_two_experiments_start_before_polling_and_results_are_independent(pbs_request, monkeypatch, failure):
    path, _, _ = pbs_request
    directory, _ = pbs.stage_submission(path)
    monkeypatch.setattr(pbs, "allocated_devices", lambda *_: GPUS)
    monkeypatch.setenv("PBS_JOBID", "123.pbs102")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", ",".join(GPUS))
    events = []

    class Run:
        def __init__(self, server, task, raw, distributed, source, *, directory, devices):
            self.directory, self.devices = directory, devices
            self.state = {"status": "RUNNING"}

        def start(self):
            events.append(("start", self.devices))

        def poll(self):
            assert len([event for event in events if event[0] == "start"]) == 2
            self.state["status"] = "FAILED" if failure and self.devices == GPUS[:4] else "COMPLETED"
            events.append(("poll", self.devices))
            return 1 if self.state["status"] == "FAILED" else 0

        def shutdown(self):
            events.append(("shutdown", self.devices))

    monkeypatch.setattr(pbs, "ServerRun", Run)
    assert pbs.run_allocation(directory) == (1 if failure else 0)
    result = json.loads((directory / "result.json").read_text())
    assert result["status"] == ("FAILED" if failure else "COMPLETED")
    assert [event[1] for event in events[:2]] == [GPUS[:4], GPUS[4:]]
    with pytest.raises(ValueError, match="already ran"):
        pbs.run_allocation(directory)


def test_pbs_signal_cancels_both_experiments_and_restores_handlers(pbs_request, monkeypatch):
    path, _, _ = pbs_request
    directory, _ = pbs.stage_submission(path)
    monkeypatch.setattr(pbs, "allocated_devices", lambda *_: GPUS)
    monkeypatch.setenv("PBS_JOBID", "123.pbs102")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", ",".join(GPUS))
    monkeypatch.setattr(pbs.time, "sleep", lambda _: None)
    instances, started = [], []

    class Run:
        def __init__(self, *args, directory, devices):
            self.directory = directory
            self.state = {"status": "RUNNING"}
            instances.append(self)

        def start(self):
            started.append(self)
            if len(started) == 2:
                signal.raise_signal(signal.SIGTERM)

        def poll(self):
            return 130 if self.state["status"] == "INTERRUPTED" else None

        def cancel(self):
            self.state["status"] = "INTERRUPTED"

        def progress(self):
            return {}

        def shutdown(self):
            assert self.state["status"] == "INTERRUPTED"

    previous = signal.getsignal(signal.SIGTERM)
    monkeypatch.setattr(pbs, "ServerRun", Run)
    assert pbs.run_allocation(directory) == 130
    assert len(started) == 2
    assert signal.getsignal(signal.SIGTERM) == previous


def test_modified_pbs_snapshot_is_rejected_before_resource_or_model_work(pbs_request, monkeypatch):
    path, _, _ = pbs_request
    directory, manifest = pbs.stage_submission(path)
    Path(manifest["experiments"][0]["training_config"]).write_text("modified: true")
    monkeypatch.setattr(pbs, "allocated_devices", lambda *_: pytest.fail("must not inspect GPUs"))
    with pytest.raises(ValueError, match="snapshot changed"):
        pbs.run_allocation(directory)


def test_missing_environment_never_submits_a_billable_job(pbs_request, monkeypatch):
    path, _, _ = pbs_request
    directory, _ = pbs.stage_submission(path)
    def query(argv, **kwargs):
        assert argv == ["conda", "env", "list", "--json"]
        return json.dumps({"envs": ["/opt/conda/envs/vla-liberox"]})
    monkeypatch.setattr(pbs.subprocess, "check_output", query)
    with pytest.raises(ValueError, match="required Conda"):
        pbs.submit(directory)
    assert not (directory / "submission.json").exists()


def test_submit_preserves_scheduler_and_complete_dependency_without_resubmission(pbs_request, monkeypatch):
    path, raw, _ = pbs_request
    raw["queue"] = "ai@pbs102"
    path.write_text(yaml.safe_dump(raw))
    directory, _ = pbs.stage_submission(path)
    assert "#PBS -q ai@pbs102" in (directory / "job.pbs").read_text()
    submitted = []
    def query(argv, **kwargs):
        if argv[0] == "conda":
            return json.dumps({"envs": ["/opt/conda/envs/pi05"]})
        submitted.append(argv)
        return "124.pbs102\n"
    monkeypatch.setattr(pbs.subprocess, "check_output", query)
    assert pbs.submit(directory, afterok="123.pbs102") == "124.pbs102"
    assert submitted[0][-3:-1] == ["-W", "depend=afterok:123.pbs102"]
    assert json.loads((directory / "submission.json").read_text())["afterok"] == "123.pbs102"
    with pytest.raises(ValueError, match="already submitted"):
        pbs.submit(directory, afterok="123.pbs102")
    assert len(submitted) == 1


def test_fingerprint_covers_hpc_and_training_but_not_documentation(tmp_path, monkeypatch):
    monkeypatch.setattr(pbs, "HPC_ROOT", tmp_path)
    monkeypatch.setattr(pbs, "training_fingerprint", lambda: "training-v1")
    (tmp_path / "pbs.py").write_text("scheduler-v1")
    before = pbs.implementation_fingerprint()
    (tmp_path / "README.md").write_text("documentation")
    assert pbs.implementation_fingerprint() == before
    (tmp_path / "pbs.py").write_text("scheduler-v2")
    changed = pbs.implementation_fingerprint()
    assert changed != before
    monkeypatch.setattr(pbs, "training_fingerprint", lambda: "training-v2")
    assert pbs.implementation_fingerprint() != changed


def test_changed_scheduler_implementation_rejects_prepared_request(pbs_request, monkeypatch):
    path, _, _ = pbs_request
    directory, _ = pbs.stage_submission(path)
    monkeypatch.setattr(pbs, "implementation_fingerprint", lambda: "changed")
    with pytest.raises(ValueError, match="implementation changed"):
        pbs.verified_request(directory)
