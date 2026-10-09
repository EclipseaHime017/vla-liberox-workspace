import json
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

from keyframe_annotation.config import Config, load_config
from keyframe_annotation.data import Recording, sha256
from keyframe_annotation.intervals import forward_windows, merge_regions
from keyframe_annotation.pipeline import Experiment
from keyframe_annotation.prompts import regions_prompt
from keyframe_annotation.schema import json_response, validate_contract, validate_plan, validate_regions


def source(tmp_path, run_id="one", success=True):
    directory = tmp_path / "sources" / run_id
    directory.mkdir(parents=True)
    count = 24
    np.savez(directory / "trajectory.npz", done=np.array([False]*18 + [success]*6),
        time_seconds=np.arange(count+1)/10, metadata_json=json.dumps({"control_hz": 10, "task_id": "pick"}))
    images = np.zeros((count+1, 8, 8, 3), dtype=np.uint8)
    images[:, 0] = 255
    np.savez_compressed(directory / "trajectory_observations.npz", agentview_image=images, wrist_image=images)
    (directory / "stage_annotation.json").write_text('{"do_not_touch": true}')
    (directory / "rynnvalue_evaluation.npz").write_bytes(b"unchanged")
    return {"run_id": run_id, "task_id": "pick", "prompt": "place the bowl", "control_hz": None,
        "trajectory_path": str(directory / "trajectory.npz"),
        "observations_path": str(directory / "trajectory_observations.npz"), "orientation": "libero_raw"}


def stage(name="grasp", initial="unmet", deps=None, refs=None):
    return {"id": name, "label": name, "achieved_when": "held", "lost_when": "dropped",
            "initial_state": initial, "initial_reason": "visible", "depends_on": deps or [], "requirement_ids": refs or []}


CONTRACT = {"requirements": [{"id": "r1", "instruction_span": "place the bowl", "condition": "bowl placed", "depends_on": []}]}
PLAN = {"initial_scene": "bowl on table", "stages": [stage(), stage("place", deps=["grasp"], refs=["r1"])], "notes": "test"}


@pytest.mark.parametrize("override", [{"unexpected": 1}, {"revision": "main"}, {"seed": True},
    {"cameras": []}, {"coarse_fps": float("nan")}, {"coarse_fps": 0}, {"coarse_fps": True},
    {"coarse_fps": "1"}, {"coarse_fps": 10.1}, {"coarse_samples": 24}, {"assessment_frames": 1},
    {"window_seconds": float("inf")}, {"window_seconds": True}, {"window_seconds": "2"}, {"coarse_fps": .25, "window_seconds": 2}, {"max_coarse_samples": 0},
    {"device": "cpu"}, {"environment": "a;rm"}, {"local_files_only": "false"}, {"mode": "other"}, {"mode": []}, {"mode": True}])
def test_strict_config(tmp_path, override):
    with pytest.raises(ValueError): Config.from_dict(override, tmp_path)


def test_yaml_and_relative_path(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text("seed: 1\nseed: 2\n")
    with pytest.raises(ValueError, match="Duplicate"): load_config(path)
    path.write_text("cache_dir: ../model\n")
    assert load_config(path).cache_dir == str(tmp_path.parent / "model")


def test_default_model_cache_is_separate_from_policy_models(tmp_path):
    assert Config.from_dict({}, tmp_path).cache_dir == str(tmp_path / "evaluator/keyframe-vlm")
    config_path = Path(__file__).resolve().parents[1] / "configs/qwen3_vl.yaml"
    cache = Path(load_config(config_path).cache_dir)
    assert cache.parent.name == "evaluator"


def test_config_errors_identify_file_and_keys(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text("old_option: 1\n")
    with pytest.raises(ValueError) as error:
        load_config(path)
    assert str(path) in str(error.value) and "old_option" in str(error.value)
    assert "restart the UI/backend" in str(error.value)
    path.write_text("- not a mapping\n")
    with pytest.raises(ValueError, match="must be a mapping"):
        load_config(path)


def test_checked_in_config_accepts_manual_window_overrides():
    config = load_config(Path(__file__).parents[1] / "configs/qwen3_vl.yaml")
    assert config.coarse_fps == 5 and config.window_seconds == 2
    assert Config.from_dict({"coarse_fps": 10, "window_seconds": 3}, Path.cwd()).window_seconds == 3


def test_recording_full_timeline_fps_and_orientation(tmp_path):
    record = Recording(source(tmp_path), Config())
    assert record.hz == 10 and record.count == 24 and record.success_step == 23
    assert np.asarray(record.image("agentview_image", 24))[-1].min() == 255
    assert record.coarse_steps(Config()) == list(range(0, 25, 2))
    steps = record.coarse_steps(Config(coarse_fps=4))
    assert steps[0] == 0 and steps[-1] == 24 and len(steps) == 11
    with pytest.raises(ValueError, match="needs"):
        record.coarse_steps(Config(coarse_fps=4, max_coarse_samples=8))


def test_bad_recording_fails(tmp_path):
    item = source(tmp_path)
    with pytest.raises(ValueError, match="disagree"): Recording({**item, "control_hz": 20}, Config())
    np.savez(item["observations_path"], agentview_image=np.zeros((24, 8, 8, 3), dtype=np.uint8))
    with pytest.raises(ValueError, match="N\\+1"): Recording(item, Config())


def test_plan_contract():
    with pytest.raises(ValueError): json_response('{"states": [], "states": []}')
    assert validate_contract(CONTRACT, "place the bowl") == CONTRACT
    assert validate_plan(PLAN, CONTRACT) == PLAN
    with pytest.raises(ValueError): validate_plan({**PLAN, "stages": []}, CONTRACT)
    with pytest.raises(ValueError, match="depends_on"):
        validate_plan({**PLAN, "stages": [stage(deps=["grasp"])]}, CONTRACT)


def test_requirement_identity_and_dependency_validation():
    for changes in [{"id": ""}, {"instruction_span": "invented instruction"}, {"depends_on": ["r1"]}, {"depends_on": "r1"}]:
        contract = deepcopy(CONTRACT)
        contract["requirements"][0].update(changes)
        with pytest.raises(ValueError): validate_contract(contract, "place the bowl")
    with pytest.raises(ValueError): validate_contract({"requirements": CONTRACT["requirements"]*2}, "place the bowl")
    with pytest.raises(ValueError): validate_contract({"requirements": []}, "place the bowl")
    for refs in [[], ["unknown"], ["r1", "r1"], "r1"]:
        plan = deepcopy(PLAN)
        plan["stages"][-1]["requirement_ids"] = refs
        with pytest.raises(ValueError): validate_plan(plan, CONTRACT)
    plan = deepcopy(PLAN)
    plan["stages"][0]["requirement_ids"] = ["r1"]
    with pytest.raises(ValueError, match="exactly one"): validate_plan(plan, CONTRACT)


def test_all_required_outcomes_and_transitive_order_survive_initial_state():
    contract = {"requirements": [*CONTRACT["requirements"], {"id": "r2", "instruction_span": "close the drawer",
        "condition": "drawer closed after placement", "depends_on": ["r1"]}]}
    with pytest.raises(ValueError, match="omits"): validate_plan(PLAN, contract)
    plan = deepcopy(PLAN)
    plan["stages"].extend([stage("withdraw", deps=["place"]), stage("close", "met", ["withdraw"], ["r2"])])
    assert validate_plan(plan, contract) == plan  # Initial met cannot bypass the required later event.
    plan["stages"][-1]["depends_on"] = ["grasp"]
    with pytest.raises(ValueError, match="must depend"): validate_plan(plan, contract)
    plan["stages"][-1]["depends_on"] = []  # Merely being later in the list is insufficient.
    with pytest.raises(ValueError, match="must depend"): validate_plan(plan, contract)


def test_regular_sampling_preserves_short_tail_and_full_post_done(tmp_path):
    item = source(tmp_path)
    record = Recording(item, Config())
    record.count = 23
    assert record.coarse_steps(Config()) == [*range(0, 24, 2), 23]
    record.count = 49
    record.hz = 20
    steps = record.coarse_steps(Config())
    assert steps == [*range(0, 49, 4), 49]
    windows = list(forward_windows(steps, 20, 2))
    assert windows[0] == list(range(0, 41, 4))  # 11 timepoints in 2 seconds
    assert windows[1] == [40, 44, 48, 49]
    assert all((w[-1]-w[0])/20 <= 2 for w in windows)
    assert list(forward_windows([0, 1], 20, 2)) == [[0, 1]]
    with pytest.raises(ValueError, match="control_hz"):
        record.coarse_steps(Config(coarse_fps=21))


def test_overlap_conflict_is_yellow_unassessed_is_not_gray():
    steps = [0, 4, 8, 12, 16]
    passes = [{"steps": [0, 4, 8], "labels": ["confirmed", "confirmed"]},
              {"steps": [4, 8, 12], "labels": ["outside", "outside"]}]
    regions = merge_regions(steps, passes)
    assert [r["status"] for r in regions] == ["confirmed", "uncertain", "outside", "pending"]
    assert regions[1]["window_indices"] == [0, 1]
    assert regions[-1]["window_indices"] == []
    assert merge_regions(steps, []) == [{"start_step": 0, "end_step": 16,
                                        "status": "pending", "window_indices": []}]


@pytest.mark.parametrize("count,hz,fps,seconds", [(2000, 20, 5, 2), (2001, 20, 5, 2), (49, 20, 5, 2), (101, 20, 3, .75)])
def test_nonoverlap_windows_cover_each_span_exactly_once(count, hz, fps, seconds):
    steps = list(range(0, count+1, round(hz/fps)))
    if steps[-1] != count:
        steps.append(count)
    windows = list(forward_windows(steps, hz, seconds))
    spans = [(lo, hi) for window in windows for lo, hi in zip(window, window[1:])]
    assert spans == list(zip(steps, steps[1:]))
    assert all(left[-1] == right[0] for left, right in zip(windows, windows[1:]))
    assert all((window[-1]-window[0])/hz <= seconds for window in windows)
    if count == 2000:
        assert len(windows) == 50


def region_response(steps, label=False):
    return {"goals": [{"stage_id": s["id"], "labels": {f"{lo}:{hi}": label for lo, hi in zip(steps, steps[1:])}}
                      for s in PLAN["stages"]]}


def semantic_response(steps, label="unrelated"):
    value = region_response(steps, label)
    for goal in value["goals"]:
        goal["reason"] = "The visible gripper moves away while the object remains on its support."
    return value


def test_region_schema_requires_all_goals_and_all_spans():
    steps = [0, 4, 8]
    value = region_response(steps)
    assert validate_regions(value, PLAN["stages"], steps)["goals"][0]["labels"] == ["outside"]*2
    for labels in [[], [False]*2, {}, {"0:4": False}, {"0:4": "met", "4:8": False},
                   {"0:4": "False", "4:8": False}, {"0:4": 0, "4:8": False},
                   {"0:4": 1, "4:8": False}, {"0:4": False, "4:9": False}, "outside"]:
        broken = deepcopy(value)
        broken["goals"][0]["labels"] = labels
        with pytest.raises(ValueError): validate_regions(broken, PLAN["stages"], steps)
    for goals in [value["goals"][:1], value["goals"]*2, [{"stage_id": "unknown", "labels": {"0:4": False, "4:8": False}}]]:
        with pytest.raises(ValueError): validate_regions({"goals": goals}, PLAN["stages"], steps)


def test_span_keys_are_reordered_without_filling_or_mutating_outputs():
    value = region_response([0, 4, 8])
    value["goals"][0]["labels"] = {"4:8": True, "0:4": False}
    raw = deepcopy(value)
    assert validate_regions(value, PLAN["stages"], [0, 4, 8])["goals"][0]["labels"] == ["outside", "confirmed"]
    assert value == raw
    value["goals"][0]["labels"]["8:12"] = False
    with pytest.raises(ValueError, match="exactly"):
        validate_regions(value, PLAN["stages"], [0, 4, 8])


def test_stage_keyed_regions_preserve_ids_and_all_judgments():
    steps = list(range(40, 81, 4))
    value = region_response(steps, "null")
    value["goals"][0]["labels"]["40:44"] = True
    value["goals"][1]["labels"]["76:80"] = "false"
    expected = validate_regions(value, PLAN["stages"], steps)
    keyed = {"goals": {goal["stage_id"]: {"labels": dict(reversed(list(goal["labels"].items())))}
                       for goal in reversed(value["goals"])}}
    original = deepcopy(keyed)
    actual = validate_regions(keyed, PLAN["stages"], steps)
    assert {g["stage_id"]: g["labels"] for g in actual["goals"]} == {
        g["stage_id"]: g["labels"] for g in expected["goals"]}
    assert keyed == original


@pytest.mark.parametrize("goals", [None, True, "grasp", {},
    {"grasp": {"labels": {"0:4": None}}},
    {"unknown": {"labels": {"0:4": None}}},
    {"grasp": None}, {"grasp": []}, {"grasp": {"0:4": None}},
    {"grasp": {"labels": {"0:4": None}, "stage_id": "place"}},
    {"grasp": {"labels": {"0:4": None}, "reason": "extra"}},
    {"grasp": {"labels": {}}, "place": {"labels": {"0:4": None}}},
    {"grasp": {"labels": {"0:4": None, "4:8": None}}, "place": {"labels": {"0:4": None}}},
])
def test_keyed_regions_do_not_fill_or_guess_incomplete_structure(goals):
    with pytest.raises(ValueError):
        validate_regions({"goals": goals}, PLAN["stages"], [0, 4])


def test_keyed_regions_reject_duplicates_extra_goals_and_wrapper_fields():
    value = {"goals": {goal["stage_id"]: {"labels": goal["labels"]}
                       for goal in region_response([0, 4])["goals"]}}
    with pytest.raises(ValueError, match="top-level"):
        validate_regions({**value, "reason": "extra"}, PLAN["stages"], [0, 4])
    with pytest.raises(ValueError, match="Duplicate JSON key"):
        json_response('{"goals":{"grasp":{"labels":{}},"grasp":{"labels":{}}}}')
    value["goals"]["other"] = {"labels": {"0:4": None}}
    with pytest.raises(ValueError, match="stage_id"):
        validate_regions(value, PLAN["stages"], [0, 4])


def test_boolean_judgments_preserve_uncertainty_without_reasons():
    steps = [0, 4, 8, 12]
    value = region_response(steps)
    value["goals"][0]["labels"] = {"0:4": True, "4:8": False, "8:12": None}
    assert validate_regions(value, PLAN["stages"], steps)["goals"][0]["labels"] == ["confirmed", "outside", "uncertain"]


def test_prompt_requests_brief_evidence_and_does_not_force_label_quota():
    prompt = regions_prompt("place the bowl", PLAN, [0, 4, 8], 20)
    assert '"reason"' in prompt and '"0:4": "<label>"' in prompt
    assert "confirmed, unrelated or uncertain" in prompt
    assert "never invent evidence to satisfy a percentage" in prompt
    assert "The final completed condition need not already hold" in prompt
    assert '"reason"' not in regions_prompt("place the bowl", PLAN, [0, 4, 8], 20, include_reason=False)


@pytest.mark.parametrize("label,expected", [("confirmed", "confirmed"), ("unrelated", "outside"), ("uncertain", "uncertain")])
@pytest.mark.parametrize("keyed", [False, True])
def test_semantic_labels_and_evidence_preserved_without_quota(label, expected, keyed):
    value = semantic_response([0, 4, 8], label)
    if keyed:
        value = {"goals": {g["stage_id"]: {k: v for k, v in g.items() if k != "stage_id"} for g in value["goals"]}}
    original = deepcopy(value)
    parsed = validate_regions(value, PLAN["stages"], [0, 4, 8], require_reason=True)
    assert all(g["labels"] == [expected, expected] for g in parsed["goals"])
    assert all(g["reason"].startswith("The visible gripper") for g in parsed["goals"])
    assert value == original


@pytest.mark.parametrize("reason", [None, "", "   ", [], {}, "x"*601, "<brief visible evidence>"])
def test_reason_validation(reason):
    value = semantic_response([0, 4])
    value["goals"][0]["reason"] = reason
    with pytest.raises(ValueError, match="reason"):
        validate_regions(value, PLAN["stages"], [0, 4], require_reason=True)


def test_current_protocol_requires_evidence_and_semantic_labels():
    value = semantic_response([0, 4])
    del value["goals"][0]["reason"]
    with pytest.raises(ValueError, match="reason"):
        validate_regions(value, PLAN["stages"], [0, 4], require_reason=True)
    value = semantic_response([0, 4], None)
    with pytest.raises(ValueError, match="not boolean/null"):
        validate_regions(value, PLAN["stages"], [0, 4], require_reason=True)


def test_quoted_literals_are_losslessly_normalized_without_mutating_response():
    steps = [0, 4, 8, 12, 16, 20, 24]
    value = region_response(steps)
    value["goals"][0]["labels"] = dict(zip(
        (f"{lo}:{hi}" for lo, hi in zip(steps, steps[1:])),
        ("true", True, "false", False, "null", None),
    ))
    original = deepcopy(value)
    assert validate_regions(value, PLAN["stages"], steps)["goals"][0]["labels"] == [
        "confirmed", "confirmed", "outside", "outside", "uncertain", "uncertain"]
    assert value == original


@pytest.mark.parametrize("label", [0, 1, 0.0, 1.0, "0", "1", "yes", "no", "True", "NULL",
    " false ", "unknown", "<judgment>", "", [], {}, {"value": True}])
def test_ambiguous_labels_report_goal_span_and_value(label):
    value = region_response([0, 4, 8])
    value["goals"][0]["labels"]["4:8"] = label
    with pytest.raises(ValueError) as error:
        validate_regions(value, PLAN["stages"], [0, 4, 8])
    assert "grasp.labels['4:8']" in str(error.value)
    assert repr(label) in str(error.value)


class FakeModel:
    metadata = {"model_id": "test-only"}
    def __init__(self, config): self.plans = 0
    def generate(self, prompt, recording, steps):
        if "Extract the task requirements" in prompt:
            assert steps == []
            return json.dumps(CONTRACT)
        if "Build the milestone plan" in prompt:
            assert steps == [0]
            self.plans += 1
            return json.dumps(PLAN)
        assert steps == sorted(steps)
        assert (steps[-1]-steps[0])/recording.hz <= 2
        return json.dumps(semantic_response(steps))
    def memory_gib(self): return 0


@pytest.mark.parametrize("keyed_second_window", [False, True])
def test_semantic_output_does_not_retry_or_change_visual_judgments(tmp_path, keyed_second_window):
    class Semantic(FakeModel):
        def generate(self, prompt, recording, steps):
            if "Locate local ACTION SEGMENTS" not in prompt:
                return super().generate(prompt, recording, steps)
            assert "Previous response" not in prompt
            value = semantic_response(steps, "uncertain")
            value["goals"][0]["labels"][f"{steps[0]}:{steps[1]}"] = "confirmed"
            value["goals"][1]["labels"][f"{steps[0]}:{steps[1]}"] = "unrelated"
            if keyed_second_window and steps[0] > 0:
                value = {"goals": {g["stage_id"]: {"reason": g["reason"], "labels": g["labels"]} for g in value["goals"]}}
            return json.dumps(value)
    output = tmp_path / "quoted"
    state = Experiment(Config(), [source(tmp_path)], output, Semantic).run()
    assert state["status"] == "COMPLETED" and state["calls"] == 4  # Two plans + two windows.
    assert not list((output / "one/calls").glob("*_1.json"))
    call = json.loads((output / "one/calls/forward_0000_0.json").read_text())
    assert call["valid"] and '"uncertain"' in call["raw_response"]
    assert call["parsed"]["goals"][0]["labels"] == ["confirmed"] + ["uncertain"]*9
    assert call["parsed"]["goals"][1]["labels"] == ["outside"] + ["uncertain"]*9
    second = json.loads((output / "one/calls/forward_0001_0.json").read_text())
    assert second["valid"]
    assert isinstance(json.loads(second["raw_response"])["goals"], dict if keyed_second_window else list)
    assert second["parsed"]["goals"][0]["labels"] == ["confirmed", "uncertain"]


def test_pipeline_sources_immutable_and_overwrites_only_selected(tmp_path):
    sources = [source(tmp_path), source(tmp_path, "two", success=False)]
    before = {str(p): sha256(p) for p in (tmp_path / "sources").rglob("*") if p.is_file()}
    model, output = FakeModel(Config()), tmp_path / "experiment"
    assert Experiment(Config(mode="localize"), sources, output, lambda _: model).run()["status"] == "COMPLETED"
    result = json.loads((output / "one/result.json").read_text())
    assert result["schema_version"] == 10 and result["localization"]["assigned_goals"] == 0
    assert result["sampling"]["stride_seconds"] == result["sampling"]["window_seconds"] == 2
    assert all(item["reason"].startswith("The visible gripper") for goal in result["goal_ranges"] for item in goal["passes"])
    assert model.plans == 2 and result["sampling"]["steps"][-1] == 24
    old_second = sha256(output / "two/result.json")
    (output / "one/stale.txt").write_text("remove")
    Experiment(Config(mode="localize"), sources[:1], output, FakeModel).run()
    assert not (output / "one/stale.txt").exists() and sha256(output / "two/result.json") == old_second
    assert before == {str(p): sha256(p) for p in (tmp_path / "sources").rglob("*") if p.is_file()}
    with pytest.raises(ValueError, match="source directory"):
        Experiment(Config(), sources, Path(sources[0]["trajectory_path"]).parent / "new", FakeModel)
    with pytest.raises(ValueError, match="source directory"):
        Experiment(Config(), sources, tmp_path, FakeModel)


def test_forward_only_regions_preserve_independent_goals_and_partial_failure(tmp_path):
    class Model(FakeModel):
        def generate(self, prompt, recording, steps):
            if "Locate local ACTION SEGMENTS" not in prompt:
                return super().generate(prompt, recording, steps)
            assert steps == sorted(set(steps))
            assert steps == list(range(0, 21, 2)) or steps == [20, 22, 24]
            return json.dumps(semantic_response(steps, "confirmed"))
    output = tmp_path / "run"
    assert Experiment(Config(), [source(tmp_path)], output, Model).run()["status"] == "COMPLETED"
    result = json.loads((output / "one/result.json").read_text())
    assert result["localization"]["assigned_goals"] == 2
    assert result["goal_ranges"][0]["regions"] == result["goal_ranges"][1]["regions"]
    assert result["goal_ranges"][0]["regions"][0]["end_step"] == 24
    calls = sorted(p.name for p in (output / "one/calls").glob("*.json"))
    assert calls == ["forward_0000_0.json", "forward_0001_0.json", "plan_0.json", "task_contract_0.json"]

    class Broken(Model):
        def generate(self, prompt, recording, steps):
            if steps and steps[0] == 20: return '{"goals":[]}'
            return super().generate(prompt, recording, steps)
    output = tmp_path / "broken"
    assert Experiment(Config(), [source(tmp_path, "partial")], output, Broken).run()["status"] == "FAILED"
    partial = json.loads((output / "partial/result.json").read_text())
    assert partial["localization"]["windows_completed"] == 1
    assert partial["goal_ranges"][0]["regions"][-1]["status"] == "pending"
    assert len(partial["goal_ranges"][0]["passes"]) == 1


def test_plan_only_never_scans_and_preserves_all_source_hash_checks(tmp_path, monkeypatch):
    def forbidden(*_): raise AssertionError("Plan-only must not enter localization")
    monkeypatch.setattr(Experiment, "localize", forbidden)
    monkeypatch.setattr(Recording, "coarse_steps", forbidden)
    sources = [source(tmp_path), source(tmp_path, "two")]
    output = tmp_path / "plans"
    state = Experiment(Config(mode="plan_only"), sources, output, FakeModel).run()
    assert state["status"] == "COMPLETED" and state["completed_runs"] == 2 and state["calls"] == 4
    for run_id in ["one", "two"]:
        result = json.loads((output / run_id / "result.json").read_text())
        assert result["mode"] == "plan_only" and result["task_contract"] == CONTRACT and result["plan"] == PLAN
        assert not result["goal_ranges"] and "localization" not in result and "sampling" not in result
    class Changed(FakeModel):
        def generate(self, prompt, recording, steps):
            if "Build the milestone plan" in prompt:
                Path(sources[0]["trajectory_path"]).write_bytes(b"changed during model call")
            return super().generate(prompt, recording, steps)
    state = Experiment(Config(mode="plan_only"), sources[:1], tmp_path / "changed", Changed).run()
    assert state["status"] == "FAILED" and "Source changed" in state["runs"][0]["error"]


def test_omitted_requirement_stops_before_localization_and_can_be_repaired(tmp_path, monkeypatch):
    def forbidden(*_): raise AssertionError("Incomplete plan must not enter localization")
    monkeypatch.setattr(Experiment, "localize", forbidden)
    class Missing(FakeModel):
        def generate(self, prompt, recording, steps):
            if "Build the milestone plan" in prompt:
                return json.dumps({**PLAN, "stages": [stage()]})
            return super().generate(prompt, recording, steps)
    sources, output = [source(tmp_path)], tmp_path / "missing"
    state = Experiment(Config(mode="localize"), sources, output, Missing).run()
    assert state["status"] == "FAILED" and state["calls"] == 3
    result = json.loads((output / "one/result.json").read_text())
    assert result["task_contract"] == CONTRACT and result["plan"] is None
    assert "omits required outcomes" in result["error"]
    class Repaired(Missing):
        def generate(self, prompt, recording, steps):
            if "Validation error" in prompt: return json.dumps(PLAN)
            return super().generate(prompt, recording, steps)
    assert Experiment(Config(mode="plan_only"), sources, tmp_path / "repaired", Repaired).run()["status"] == "COMPLETED"


def test_malformed_output_and_cancellation_remain_diagnosable(tmp_path):
    sources = [source(tmp_path)]
    class Invalid(FakeModel):
        def generate(self, *args): return "not JSON"
    output = tmp_path / "invalid"
    assert Experiment(Config(), sources, output, Invalid).run()["status"] == "FAILED"
    assert len(list((output / "one/calls").glob("*.json"))) == 2
    class Interrupted(FakeModel):
        def generate(self, *args): raise KeyboardInterrupt()
    output = tmp_path / "canceled"
    with pytest.raises(KeyboardInterrupt): Experiment(Config(), sources, output, Interrupted).run()
    assert json.loads((output / "progress.json").read_text())["status"] == "CANCELED"
