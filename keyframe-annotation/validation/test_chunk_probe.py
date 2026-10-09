"""CPU-only checks for the isolated diagnostic, not platform integration tests."""
import importlib.util
import json
from pathlib import Path

spec = importlib.util.spec_from_file_location("chunk_probe", Path(__file__).with_name("chunk_probe.py"))
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)


def test_full_and_tail_chunk_observations():
    assert probe.input_steps(8, 16, 20, "video_dual") == list(range(8, 17))
    assert probe.input_steps(496, 500, 20, "video_dual") == list(range(496, 501))


def test_context_does_not_use_future_and_respects_hz():
    assert probe.input_steps(8, 16, 20, "context_dual") == list(range(17))
    assert probe.input_steps(80, 88, 20, "context_dual") == list(range(48, 89))
    assert probe.input_steps(80, 88, 10, "context_dual") == list(range(68, 89))


def test_main_context_preserves_dual_context_frames_and_prompt():
    for start in range(0, 500, 8):
        end = min(start+8, 500)
        assert probe.input_steps(start, end, 20, "context_main") == probe.input_steps(start, end, 20, "context_dual")
        assert probe.caption_prompt("TASK", start, end, 20, "context_main") == probe.caption_prompt("TASK", start, end, 20, "context_dual")


def test_main_context_changes_only_camera_selection():
    class FakeModel:
        config = probe.load_config(probe.ROOT / "keyframe-annotation/configs/qwen3_vl.yaml")

        def _prepare_inputs(self, prompt, recording, steps):
            return self.config.cameras, prompt, recording, steps

    model = FakeModel()
    recording = object()
    steps = probe.input_steps(80, 88, 20, "context_main")
    assert probe.prepare_inputs(model, recording, "PROMPT", steps, "context_main") == (
        (probe.CAMERAS[0],), "PROMPT", recording, steps)
    assert probe.prepare_inputs(model, recording, "PROMPT", steps, "context_dual") == (
        probe.CAMERAS, "PROMPT", recording, steps)


def test_prompt_is_description_not_annotation_or_outcome():
    prompt = probe.caption_prompt("place the black bowl on the stove", 8, 16, 20, "video_dual")
    assert "0.40-0.80 seconds" in prompt
    assert "at most 40 English words" in prompt
    assert "not labels or JSON" in prompt
    assert "intended goals, not observed facts" in prompt
    assert not any(word in prompt for word in ("success", "failure", "done", "confirmed", "uncertain"))


def test_neutral_control_removes_only_background():
    standard = probe.caption_prompt("TASK", 8, 16, 20, "video_dual")
    neutral = probe.caption_prompt("TASK", 8, 16, 20, "neutral_dual")
    assert standard.endswith(neutral)
    assert "TASK" not in neutral and "Subgoals" not in neutral


def test_probe_selection_has_only_valid_distinct_chunk_starts():
    for run_id, count in zip(probe.RUNS, (500, 300)):
        starts = probe.PROBES[run_id]
        assert len(starts) == len(set(starts))
        assert all(0 <= start < count and start % 8 == 0 for start in starts)


def test_full_trajectory_variants_include_all_chunks_and_short_tail():
    for run_id, count in zip(probe.RUNS, (500, 300)):
        for variant in probe.VARIANTS:
            starts = probe.chunk_starts(run_id, count, variant, full_trajectory=True)
            assert starts == list(range(0, count, 8))
            assert count-starts[-1] == 4


def test_default_contrast_sampling_is_unchanged():
    for run_id, count in zip(probe.RUNS, (500, 300)):
        assert probe.chunk_starts(run_id, count, "video_dual") == list(range(0, count, 8))
        for variant in probe.VARIANTS[1:]:
            assert probe.chunk_starts(run_id, count, variant) == list(probe.PROBES[run_id])


def test_review_covers_exact_windows_without_missing_or_invented_scores():
    review = json.loads(Path(__file__).with_name("chunk_probe_review.json").read_text())
    for run_id, count in zip(probe.RUNS, (500, 300)):
        record = review["recordings"][run_id]
        assert len(record["baseline_ratings"]) == (count+7)//8
        assert set(record["baseline_ratings"]) <= {"A", "P", "W"}
        assert record["contrast_starts"] == list(probe.PROBES[run_id])
        for ratings in record["ratings"].values():
            assert len(ratings) == len(probe.PROBES[run_id])
            assert set(ratings) <= {"A", "P", "W"}


def test_full_success_review_preserves_previous_probe_ratings():
    review = json.loads(Path(__file__).with_name("chunk_success_full_review.json").read_text())
    previous = json.loads(Path(__file__).with_name("chunk_probe_review.json").read_text())
    success = previous["recordings"][probe.RUNS[0]]
    assert review["protocol"]["chunk_count"] == 63
    assert review["ratings"]["video_dual"] == success["baseline_ratings"]
    for variant in ("context_dual", "video_main"):
        ratings = review["ratings"][variant]
        assert len(ratings) == 63
        assert set(ratings) <= {"A", "P", "W"}
        assert "".join(ratings[start//8] for start in probe.PROBES[probe.RUNS[0]]) == success["ratings"][variant]


def test_combined_context_review_covers_all_chunks_and_critical_ranges():
    review = json.loads(Path(__file__).with_name("chunk_success_context_main_review.json").read_text())
    ratings = review["ratings"]
    assert review["protocol"]["run_id"] == probe.RUNS[0]
    assert len(ratings) == review["protocol"]["chunk_count"] == 63
    assert set(ratings) <= {"A", "P", "W"}
    assert {label: ratings.count(label) for label in "APW"} == review["counts"]
    critical = [ratings[i] for i in range(63) if any(
        interval["start"] <= i*8 < interval["end"] for interval in review["critical_action_ranges"])]
    assert len(critical) == 16
    assert {label: critical.count(label) for label in "APW"} == review["critical_action_counts"]
