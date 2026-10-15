"""Opt-in real MuJoCo validation of saved, multi-instruction continuation chains."""
import os
from pathlib import Path

import numpy as np
import pytest
import yaml

from backend.app.services.task_registry import RegistryTaskCatalog, freeze_task
from backend.app.services.run_continuation import prefix_segments, promote_run
from backend.app.simulators.control_state import restore_control_state
from simulation_core import run_control_loop
from trajectory_utils import TrajectoryRecorder, load_trajectory, save_trajectory_bundle


@pytest.mark.skipif(os.environ.get("RUN_MUJOCO_SMOKE") != "1", reason="opt-in MuJoCo smoke test")
def test_custom_scene_two_human_policy_cycles_keep_state_and_instructions(tmp_path, monkeypatch):
    import eval_pickplace_direct as direct
    root = Path(__file__).resolve().parents[2]
    monkeypatch.syspath_prepend(str(root / "third_party/LIBERO-X"))
    runtime = direct.load_runtime()
    bddl, init = direct.resolve_task(root / "third_party/LIBERO-X", "LEVEL1",
        "EXTENSION_KITCHEN_SCENE11_place_the_black_bowl_on_the_flat_stove")
    (tmp_path / "custom.bddl").write_bytes(bddl.read_bytes())
    (tmp_path / "custom.init").write_bytes(init.read_bytes())
    goal = "place the black bowl on the flat stove"
    task_id = "custom::bowl"
    family = dict(schema_version=1, family_id="custom", label="Custom bowl", variants={"scene": dict(
        task_id=task_id, level="LEVEL1", task_name="custom_bowl", goal_prompt=goal,
        bddl="custom.bddl", init="custom.init", instructions=[dict(id="grasp", text="grasp bowl"),
                                                               dict(id="place", text="place bowl")])})
    (tmp_path / "family.yaml").write_text(yaml.safe_dump(family))
    (tmp_path / "registry.yaml").write_text(yaml.safe_dump(dict(schema_version=1,
        default_task_id=task_id, families={"custom": "family.yaml"})))
    catalog = RegistryTaskCatalog(runtime, tmp_path / "registry.yaml")
    frozen = freeze_task(catalog, task_id, tmp_path / "frozen")
    (tmp_path / "custom.bddl").write_text("source changed after snapshot")
    scene, _ = frozen.paths(task_id)
    saved, metadata = None, None
    expected_instructions = []
    for index, source in enumerate(("policy", "human", "policy_requery", "human", "policy_requery")):
        run_id = f"part-{index}"
        instruction = "grasp bowl" if index < 2 else "place bowl"
        start = 0 if saved is None else len(saved["env_action"])
        env = direct.make_env(runtime, scene, 32, start + 5, 20, 0, "vla_views", 64, 32, True)
        try:
            state = frozen.initial_state(task_id, 0) if saved is None else saved["sim_state"][-1]
            observation = direct.restore_state(env, state)
            if saved is None:
                recorder = TrajectoryRecorder(capture_images=False)
                recorder.record_initial(env, observation)
                segments = []
            else:
                np.testing.assert_allclose(env.get_sim_state(), state, atol=1e-9, rtol=0)
                recorder = TrajectoryRecorder.from_prefix(saved, start, 20.)
                segments = prefix_segments(metadata, start, start)
                restore_control_state(env, recorder.env_actions, 20.)
                np.testing.assert_allclose(env.get_sim_state(), state, atol=1e-9, rtol=0)
            recorder.policy_action_codec = "libero_env_v1"
            action = np.array([.02, 0, 0, 0, 0, 0, -1.], dtype=np.float32)
            arguments = dict(manual_query=lambda _: action) if source == "human" else dict(
                policy_query=lambda observation, step: (observation, np.tile(action, (8, 1))),
                policy_action_transform=lambda value: value)
            run_control_loop(env=env, recorder=recorder, initial_observation=observation,
                target_action_count=start + 3, action_source=source, open_loop_steps=2,
                rate_limiter=direct.RealTimeControlLimiter(20., False), **arguments)
            segments.append(dict(start=start, end=recorder.action_count,
                                 origin_run_id=run_id, policy_instruction=instruction))
            metadata = dict(session_id=run_id, root_session_id="part-0",
                parent_session_id=None if index == 0 else f"part-{index-1}",
                source_resume_step=start if index else None, task=goal,
                policy_instruction=instruction, trajectory_segments=segments,
                handoff_steps=list(range(3, recorder.action_count, 3)))
            # Reconstruct both observation streams using the frozen scene.
            direct.postprocess_recorded_trajectory(runtime, env, recorder, combined_path=None,
                main_view_path=None, save_observations=True, fps=20, video_camera="vla_views",
                video_width=64, video_height=32, main_view_width=32, main_view_height=32)
            directory = tmp_path / run_id
            directory.mkdir()
            paths = save_trajectory_bundle(recorder, directory / "trajectory", metadata, create_plot=False)
            saved, metadata = load_trajectory(Path(paths["trajectory"]))
            expected_instructions.extend([instruction] * 3)
            actual = [segment["policy_instruction"] for segment in metadata["trajectory_segments"]
                      for _ in range(segment["start"], segment["end"])]
            assert actual == expected_instructions and metadata["task"] == goal
            assert len(saved["sim_state"]) == len(saved["env_action"]) + 1
            with np.load(paths["trajectory_observations"]) as images:
                assert len(images["agentview_image"]) == len(images["wrist_image"]) == len(saved["sim_state"])
            if index:
                public = promote_run(dict(id=run_id, kind="branch", status="COMPLETED",
                    output_dir=str(directory), trajectory=paths["trajectory"], action_count=len(saved["env_action"])))
                assert public["role"] == "original" and public["branchable"]
        finally:
            env.close()
