"""Queue Value Flows return-distribution rollouts of a frozen policy and task."""
from __future__ import annotations

import copy
import re
import shutil
import uuid
from datetime import datetime
from typing import Any

from core.methods import METHODS
from studio.services.tool_jobs import ToolJobService
from studio.storage.files import atomic_write_yaml

ARTIFACTS = frozenset({"returns.mp4", "rollout.mp4", "returns.json", "returns.npz", "summary.json",
                       "effective_config.yaml"})
DEFAULT_SAMPLES = 256


def flow_critic_policy(entry) -> bool:
    """Trained policies whose exported critic models the return distribution."""
    method = METHODS.get(entry.get("algorithm") if isinstance(entry, dict) else entry.algorithm)
    is_base = entry.get("kind") == "base" if isinstance(entry, dict) else entry.is_base
    return not is_base and method is not None and method.critic == "value_flow" and method.exports_critic


class ReturnDistributionService(ToolJobService):
    KIND = "return_distribution"
    ROUTE = "/api/tools/returns"
    ARTIFACTS = ARTIFACTS
    DETAILS = {"summary.json": "return_summary", "progress.json": "return_progress"}

    def defaults(self) -> dict[str, Any]:
        policies = [item["policy_id"] for item in self.jobs.manager.policy_catalog.list_policies()
                    if flow_critic_policy(item)]
        return {"return_samples": DEFAULT_SAMPLES, "policy_ids": policies}

    def start(self, parameters: dict[str, Any]) -> dict[str, Any]:
        from studio.domain.return_distribution import ReturnDistributionRequest
        from studio.policies.snapshots import validate_snapshot
        from studio.services.task_registry import freeze_task

        parameters = ReturnDistributionRequest.model_validate(parameters).model_dump()
        with self.jobs.lock:
            if not flow_critic_policy(self.jobs.manager.policy_catalog.entry(parameters["policy_id"])):
                raise ValueError("Return distributions need a trained policy with a Value Flows critic")
            preview = self.jobs.preview_evaluation({
                "task_id": parameters["task_id"], "policy_id": parameters["policy_id"],
                "trials": 1, "init_state_indices": [parameters["init_state_index"]],
                "base_seed": parameters["seed"], "seed_count": 1, "schedule_seed": 0,
                "max_steps": parameters["max_steps"], "open_loop_steps": parameters["open_loop_steps"],
                "realtime": False,
            })
            policy = copy.deepcopy(validate_snapshot(preview["policy_snapshot"]))
            if not re.fullmatch(r"[0-9a-f]{64}", str(policy.get("content_sha256", ""))):
                raise ValueError("The selected model must have a verified immutable identity")
            job_id = f"returns_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
            directory = self.jobs.jobs_root / job_id
            directory.mkdir(parents=True, exist_ok=False)
            published = False
            try:
                frozen = freeze_task(self.jobs.manager.catalog, parameters["task_id"], directory / "task_snapshot")
                bddl, init = frozen.paths(parameters["task_id"])
                task = {**preview["task_snapshot"], "bddl_path": str(bddl), "init_path": str(init),
                        "task_revision_sha256": frozen.manifest["task_revision_sha256"]}
                effective = {"schema_version": 1, "job_id": job_id, "policy_snapshot": policy,
                    "task_snapshot": task,
                    "rollout": {
                        "max_steps": parameters["max_steps"], "open_loop_steps": preview["config"]["open_loop_steps"],
                        "seed": parameters["seed"], "init_state_index": parameters["init_state_index"],
                        "control_hz": preview["config"]["control_hz"],
                        "success_consecutive_steps": preview["config"]["success_streak"],
                    },
                    "analysis": {"return_samples": parameters["return_samples"], "seed": parameters["seed"]},
                    "artifacts_dir": str((directory / "artifacts").resolve())}
                config_path = directory / "effective_config.yaml"
                atomic_write_yaml(config_path, effective)
                stages = [{"id": phase, "label": label, "environment": self.jobs.ui_config.train_environment,
                    "argv": ["python", "-m", "studio.commands.return_distribution", "--config", str(config_path),
                             "--phase", phase]}
                    for phase, label in (("rollout", "运行策略并记录每次查询的候选块"),
                                         ("analyze", "采样回报分布并生成视频"))]
                job = self.jobs._new_job(kind=self.KIND, dataset_id=None, stages=stages,
                    config_path=config_path, output_path=directory / "artifacts", queued=True,
                    parameters={**parameters, "policy_label": policy["label"],
                                "policy_content_sha256": policy["content_sha256"], "requires_gpu": True})
                published = True
                return job
            finally:
                if not published and directory.exists():
                    shutil.rmtree(directory)
