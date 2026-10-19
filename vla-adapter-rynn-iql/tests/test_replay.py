from __future__ import annotations

from pathlib import Path
import json
import shutil

import numpy as np
import pytest

from vla_rynn_iql.data import load_manifest, prepare_dataset
from vla_rynn_iql.replay import ActionDataset, ReplayDataset
from vla_rynn_iql.rewards import annotate_manifest, load_reward_index
from vla_rynn_iql.vla_adapter import env_to_dataset_actions


class FakeAnnotator:
    metadata = {"provider": "fake", "revision": "test"}

    def predict(self, prompt, frames):
        values = np.arange(len(frames), 0, -1, dtype=np.float32)
        return {
            "absolute_temporal_distance_seconds": values[:, None],
            "absolute_value_entropy_nats": np.zeros((len(frames), 1), np.float32),
            "absolute_value_logits": np.zeros((len(frames), 1, 256), np.float32),
            "relative_temporal_distance_seconds": np.zeros(len(frames), np.float32),
            "relative_value_logits": np.zeros((len(frames), 256), np.float32),
        }

    def analyze(self, prompt, frames):
        return {
            "generated_text": "- Match: Yes\n- Success: No",
            "generated_token_ids": [1, 2],
            "parsed_for_display": {"match": "Yes", "success": "No"},
        }


def _stats(dim: int) -> dict[str, list[float]]:
    return {"q01": [0.0] * dim, "q99": [1.0] * dim}


@pytest.mark.parametrize("family", ["vla_adapter", "pi05"])
@pytest.mark.parametrize("target,strides", [
    ("none", (1, 1)), ("flip_vertical", (-1, 1)),
    ("flip_horizontal", (1, -1)), ("rotate_180", (-1, -1)),
])
@pytest.mark.parametrize("storage", ["libero_raw", "vla_policy"])
def test_actor_and_critic_images_follow_model_contract(configured, monkeypatch, family, target, strides, storage):
    from vla_rynn_iql import replay as module
    from vla_rynn_iql.models import model_config
    from vla_rynn_iql.base_models import model_contract

    configured.raw["model"] = model_config({"model": {"family": family}})
    contract = model_contract(configured.section("model"))
    contract["observation"].update(image_flip_vertical=strides[0] == -1, image_flip_horizontal=strides[1] == -1)
    configured.raw["model"]["contract"] = contract
    prepare_dataset(configured)
    manifest = load_manifest(configured)
    agent = np.arange(16 * 16 * 3, dtype=np.uint8).reshape(16, 16, 3)
    wrist = np.roll(agent, 2, axis=1)
    for episode in manifest["episodes"]:
        episode["observation_orientation"] = storage
        count = int(episode["recorded_action_count"]) + 1
        images = {"agentview_image": np.stack([agent + i for i in range(count)]),
                  "wrist_image": np.stack([wrist + i for i in range(count)])}
        if storage == "vla_policy":
            images = {key: value[:, ::-1, ::-1].copy() for key, value in images.items()}
        np.savez_compressed(episode["observations_path"], **images)
    monkeypatch.setattr(module, "load_manifest", lambda config: manifest)
    bc = ActionDataset(configured, _stats(7), _stats(8))
    sample = bc[0]
    start, length = sample["start"], sample["chunk_length"].item()
    expected_agent = (agent + start)[::strides[0], ::strides[1]].copy()
    expected_wrist = (wrist + start)[::strides[0], ::strides[1]].copy()
    np.testing.assert_array_equal(sample["agent_image"].numpy(), expected_agent)
    np.testing.assert_array_equal(sample["wrist_image"].numpy(), expected_wrist)
    bc.image_size = 16
    rl_sample = bc._sample(0, transitions=True)
    end = start + length
    np.testing.assert_array_equal(rl_sample["pixels"][:3].numpy(), expected_agent.transpose(2, 0, 1))
    np.testing.assert_array_equal(rl_sample["next_pixels"][:3].numpy(),
                                  (agent + end)[::strides[0], ::strides[1]].transpose(2, 0, 1))


def test_transient_raw_done_does_not_terminate_an_earlier_chunk(configured):
    source = Path(configured.section("paths")["dataset_sources"][0])
    trajectory = next(source.rglob("branch/episodes/episode_000/trajectory.npz"))
    with np.load(trajectory, allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in archive.files}
    arrays["done"] = arrays["done"].copy()
    arrays["done"][8] = True
    np.savez_compressed(trajectory, **arrays)

    prepare_dataset(configured)
    annotate_manifest(configured, FakeAnnotator())
    replay = ReplayDataset(configured, _stats(7), _stats(8), split="train")
    def replay_index(start: int) -> int:
        return next(
            index for index, (episode, chunk_index, _) in enumerate(replay.items)
            if episode["run_id"] == "branch"
            and episode["chunks"][chunk_index]["start"] == start
        )

    assert replay[replay_index(5)]["bootstrap_mask"].item() == 1.0
    assert replay[replay_index(13)]["bootstrap_mask"].item() == 1.0
    assert replay[replay_index(18)]["bootstrap_mask"].item() == 0.0


def test_post_success_actions_images_and_masks_are_sampled(configured):
    source = Path(configured.section("paths")["dataset_sources"][0])
    trajectory = next(source.rglob("branch/episodes/episode_000/trajectory.npz"))
    with np.load(trajectory, allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in archive.files}
    arrays["env_action"][18:, 0] = [0.2, 0.4, 0.6, 0.8]
    arrays["done"][18:] = False
    np.savez_compressed(trajectory, **arrays)
    observations = trajectory.with_name("trajectory_observations.npz")
    agent = np.broadcast_to(np.arange(23, dtype=np.uint8)[:, None, None, None], (23, 16, 16, 3))
    wrist = agent + 50
    np.savez_compressed(observations, agentview_image=agent, wrist_image=wrist)

    prepare_dataset(configured)
    annotate_manifest(configured, FakeAnnotator())
    replay = ReplayDataset(configured, _stats(7), _stats(8), split="train")
    item = next(
        replay[index] for index, (episode, chunk_index, _) in enumerate(replay.items)
        if episode["run_id"] == "branch" and replay.chunks["branch"][chunk_index]["start"] == 18
    )
    assert item["chunk_length"].item() == 4
    assert item["action_mask"].tolist() == [True] * 4 + [False] * 4
    np.testing.assert_allclose(
        item["actions"][:4], env_to_dataset_actions(arrays["env_action"][18:22], _stats(7)),
    )
    assert not item["actions"][4:].any()
    assert item["agent_image"].unique().tolist() == [18]
    assert item["wrist_image"].unique().tolist() == [68]
    assert item["pixels"][:3].unique().tolist() == [18]
    assert item["next_pixels"][:3].unique().tolist() == [22]
    assert item["next_pixels"][3:].unique().tolist() == [72]
    assert item["transition_type"] == "human"
    assert item["bootstrap_mask"].item() == 0.0


def test_post_success_setting_filters_samples_without_changing_data_or_rewards(configured):
    prepared = prepare_dataset(configured)
    annotate_manifest(configured, FakeAnnotator())
    index = load_reward_index(configured)
    files = [prepared.manifest, *(Path(ep["annotation_path"]) for ep in index["episodes"])]
    files += [Path(ep[key]) for ep in load_manifest(configured)["episodes"]
              for key in ("trajectory_path", "observations_path")]
    before = {path: path.read_bytes() for path in files}
    full = ReplayDataset(configured, _stats(7), _stats(8))
    configured.raw["data"]["include_post_success"] = False
    truncated = ReplayDataset(configured, _stats(7), _stats(8))
    bc = ActionDataset(configured, _stats(7), _stats(8))
    assert len(truncated) == len(bc) == len(full) - 1
    assert [(ep["run_id"], i) for ep, i, _ in bc.items] == [
        (ep["run_id"], i) for ep, i, _ in truncated.items]
    branch = [truncated[i] for i, (ep, _, _) in enumerate(truncated.items) if ep["run_id"] == "branch"]
    assert [item["start"] for item in branch] == [0, 5, 13]
    assert branch[-1]["chunk_length"].item() == 5  # Includes the fifth done=True action (17).
    assert branch[-1]["action_mask"].tolist() == [True] * 5 + [False] * 3
    assert branch[-1]["bootstrap_mask"].item() == 0
    old = next(full[i] for i, (ep, n, _) in enumerate(full.items)
               if ep["run_id"] == "branch" and full.chunks["branch"][n]["start"] == 13)
    assert old["bootstrap_mask"].item() == 1
    assert old["reward"].item() == branch[-1]["reward"].item()
    # The failed parent is unaffected, including its final partial chunk.
    assert [(ep["run_id"], i) for ep, i, _ in full.items if ep["run_id"] == "root"] == [
        (ep["run_id"], i) for ep, i, _ in truncated.items if ep["run_id"] == "root"]
    assert {path: path.read_bytes() for path in files} == before
    configured.raw["data"]["include_post_success"] = True
    assert len(ReplayDataset(configured, _stats(7), _stats(8))) == len(full)


def test_truncation_ignores_unconfirmed_success(configured):
    root = Path(configured.section("paths")["dataset_sources"][0])
    path = next(root.rglob("branch/episodes/episode_000/trajectory.npz"))
    with np.load(path, allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in archive.files}
    arrays["done"][:] = False
    arrays["done"][3:7] = True  # Four consecutive, not five.
    np.savez_compressed(path, **arrays)
    configured.raw["data"]["include_post_success"] = False
    prepare_dataset(configured)
    bc = ActionDataset(configured, _stats(7), _stats(8))
    branch = next(ep for ep in bc.manifest["episodes"] if ep["run_id"] == "branch")
    assert branch["terminal_step"] is None
    indices = [index for ep, index, _ in bc.items if ep["run_id"] == "branch"]
    assert bc.chunks["branch"][indices[-1]]["end"] == 22


def test_replay_rejects_legacy_manifest_missing_full_tail(configured):
    prepared = prepare_dataset(configured)
    annotate_manifest(configured, FakeAnnotator())
    rewards = load_reward_index(configured)
    manifest = load_manifest(configured)
    branch = next(ep for ep in manifest["episodes"] if ep["run_id"] == "branch")
    branch["chunks"] = branch["chunks"][:-1]
    branch["action_count"] = 18
    branch.pop("evaluation_chunks")
    manifest.pop("replay_policy")
    prepared.manifest.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="full-recording replay chunks"):
        ReplayDataset(configured, _stats(7), _stats(8), reward_index=rewards)


def test_replay_rejects_rewards_that_omit_post_success_tail(configured):
    prepare_dataset(configured)
    annotate_manifest(configured, FakeAnnotator())
    rewards = load_reward_index(configured)
    branch = next(ep for ep in rewards["episodes"] if ep["run_id"] == "branch")
    path = Path(branch["annotation_path"])
    with np.load(path, allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in archive.files}
    key = "final_reward" if "final_reward" in arrays else "pbrs_chunk_reward"
    arrays[key] = arrays[key][:-1]
    np.savez_compressed(path, **arrays)
    with pytest.raises(ValueError, match="reward array does not cover all 4"):
        ReplayDataset(configured, _stats(7), _stats(8), reward_index=rewards)


def test_replay_exposes_variable_duration_transition_metadata(configured):
    prepare_dataset(configured)
    annotate_manifest(configured, FakeAnnotator())
    replay = ReplayDataset(configured, _stats(7), _stats(8), split="train")
    item = next(
        replay[index] for index, (episode, chunk_index, _) in enumerate(replay.items)
        if episode["run_id"] == "branch"
        and episode["chunks"][chunk_index]["interrupted"]
    )
    assert item["chunk_length"].item() == 5
    assert item["action_mask"].tolist() == [True] * 5 + [False] * 3
    assert item["action_source"] == "policy"
    assert item["transition_type"] == "policy_interrupted"
    assert item["bootstrap_mask"].item() == 1.0


def test_replay_deduplicates_sibling_interrupted_prefixes(configured):
    source = Path(configured.section("paths")["dataset_sources"][0])
    branch_run = next(source.rglob("branch/run.json")).parent
    sibling_run = branch_run.parent / "branch-sibling"
    shutil.copytree(branch_run, sibling_run)
    run = json.loads((sibling_run / "run.json").read_text(encoding="utf-8"))
    run["id"] = "branch-sibling"
    (sibling_run / "run.json").write_text(json.dumps(run), encoding="utf-8")

    prepare_dataset(configured)
    annotate_manifest(configured, FakeAnnotator())
    replay = ReplayDataset(configured, _stats(7), _stats(8), split="train")
    interrupted = [
        (episode["root_run_id"], episode["chunks"][chunk_index]["start"],
         episode["chunks"][chunk_index]["end"])
        for episode, chunk_index, _ in replay.items
        if episode["chunks"][chunk_index]["interrupted"]
    ]
    assert interrupted == [("root", 0, 5)]


def test_replay_deduplicates_full_copied_prefix_against_parent(configured):
    source = Path(configured.section("paths")["dataset_sources"][0])
    run_json = next(source.rglob("branch/run.json"))
    run = json.loads(run_json.read_text(encoding="utf-8"))
    run["resume_step"] = 13
    run_json.write_text(json.dumps(run), encoding="utf-8")
    trajectory = run_json.parent / "episodes" / "episode_000" / "trajectory.npz"
    with np.load(trajectory, allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in archive.files}
    sources = np.asarray(["policy"] * len(arrays["env_action"]), dtype="<U32")
    sources[13:] = "human"
    arrays["action_source"] = sources
    np.savez_compressed(trajectory, **arrays)

    prepare_dataset(configured)
    annotate_manifest(configured, FakeAnnotator())
    replay = ReplayDataset(configured, _stats(7), _stats(8), split="train")
    copied_zero_to_eight = [
        (episode["run_id"], episode["chunks"][chunk_index]["start"],
         episode["chunks"][chunk_index]["end"])
        for episode, chunk_index, _ in replay.items
        if episode["root_run_id"] == "root"
        and episode["chunks"][chunk_index]["start"] == 0
        and episode["chunks"][chunk_index]["end"] == 8
    ]
    assert copied_zero_to_eight == [("root", 0, 8)]
    assert any(
        episode["run_id"] == "branch"
        and episode["chunks"][chunk_index]["start"] == 8
        and episode["chunks"][chunk_index]["end"] == 13
        for episode, chunk_index, _ in replay.items
    )


def test_replay_frames_are_decoded_once_per_episode_and_reused_from_disk(configured, monkeypatch):
    from vla_rynn_iql import frame_cache

    prepare_dataset(configured)
    observations = {Path(episode["observations_path"]).resolve()
                    for episode in load_manifest(configured)["episodes"]}
    decoded = []
    original = frame_cache.np.load

    def counting_load(path, *args, **kwargs):
        if Path(path).resolve() in observations:
            decoded.append(Path(path).resolve())
        return original(path, *args, **kwargs)

    monkeypatch.setattr(frame_cache.np, "load", counting_load)
    first = ActionDataset(configured, _stats(7), _stats(8))
    samples = [first[index] for index in range(len(first))] * 2
    assert len(decoded) == len({episode["run_id"] for episode, _, _ in first.items})

    for episode, chunk_index, _ in first.items[:3]:
        start = int(first.chunks[episode["run_id"]][chunk_index]["start"])
        with original(episode["observations_path"], allow_pickle=False) as source:
            expected = source["agentview_image"][start]
        item = first.items.index((episode, chunk_index, None))
        np.testing.assert_array_equal(
            first[item]["agent_image"].numpy(),
            expected[::-1, ::-1])

    decoded.clear()
    second = ActionDataset(configured, _stats(7), _stats(8))
    for index in range(len(second)):
        np.testing.assert_array_equal(second[index]["agent_image"].numpy(),
                                      samples[index]["agent_image"].numpy())
    assert decoded == []
    assert any((Path(configured.section("paths")["annotation_cache"]) / "replay-frames").iterdir())


@pytest.mark.parametrize("damage", ["empty", "wrong_dtype", "wrong_shape", "bad_json"])
def test_damaged_frame_cache_is_rebuilt_from_the_recording(configured, damage):
    prepare_dataset(configured)
    reference = ActionDataset(configured, _stats(7), _stats(8))
    expected = [reference[index]["agent_image"].numpy() for index in range(len(reference))]
    for folder in (Path(configured.section("paths")["annotation_cache"]) / "replay-frames").iterdir():
        if not folder.is_dir():
            continue
        target = folder / "agentview_image.npy"
        if damage == "empty":
            target.write_bytes(b"")
        elif damage == "wrong_dtype":
            np.save(target, np.load(target).astype(np.float32))
        elif damage == "wrong_shape":
            np.save(target, np.load(target)[:, :, :, :1])
        else:
            (folder / "frames.json").write_text("null")
    rebuilt = ActionDataset(configured, _stats(7), _stats(8))
    for index, image in enumerate(expected):
        actual = rebuilt[index]["agent_image"].numpy()
        assert actual.dtype == np.uint8
        np.testing.assert_array_equal(actual, image)
    fresh = ActionDataset(configured, _stats(7), _stats(8))
    assert all(fresh._frames._load(key, sorted(indices)) is not None
               for key, indices in ((fresh._frames.key(episode, fresh._frame_indices[episode["run_id"]]),
                                     fresh._frame_indices[episode["run_id"]])
                                    for episode, _, _ in fresh.items))


def test_concurrent_rebuilds_of_a_damaged_frame_cache_decode_once(configured, monkeypatch):
    import threading
    import time
    from vla_rynn_iql.frame_cache import FrameCache

    prepare_dataset(configured)
    reference = ActionDataset(configured, _stats(7), _stats(8))
    reference[0]  # publish the entry
    episode = reference.items[0][0]
    indices = sorted(reference._frame_indices[episode["run_id"]])
    directory = Path(configured.section("paths")["annotation_cache"]) / "replay-frames"
    key = FrameCache.key(episode, indices)
    expected = np.load(directory / key / "agentview_image.npy")
    (directory / key / "frames.json").write_text("null")

    decoded, original = [], FrameCache._decode
    def slow_decode(self, *args):
        decoded.append(threading.get_ident())
        time.sleep(0.3)  # keep the first rebuild in flight while the other process checks
        return original(self, *args)
    monkeypatch.setattr(FrameCache, "_decode", slow_decode)

    results, start = [None, None], threading.Barrier(2)
    def worker(slot):
        start.wait()
        results[slot] = FrameCache(directory).episode(episode, indices).frame("agentview_image", indices[0])
    threads = [threading.Thread(target=worker, args=(slot,)) for slot in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert len(decoded) == 1
    for result in results:
        np.testing.assert_array_equal(result, expected[0])
    assert FrameCache(directory)._load(key, indices) is not None
