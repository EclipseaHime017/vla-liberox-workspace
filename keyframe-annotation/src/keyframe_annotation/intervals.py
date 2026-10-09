from bisect import bisect_right


def forward_windows(steps, hz, seconds):
    """Cover each sampled span once; neighboring windows share only an endpoint."""
    start = 0
    while start < len(steps) - 1:
        end = bisect_right(steps, steps[start] + seconds * hz + 1e-9) - 1
        if end <= start:
            raise ValueError("Window is shorter than the recorded sampling interval")
        yield steps[start:end+1]
        if end == len(steps) - 1:
            return
        start = end


def merge_regions(steps, passes):
    """Keep goals independent; disagreement is uncertainty, not a confidence vote."""
    evidence = {}
    for index, item in enumerate(passes):
        for lo, hi, label in zip(item["steps"], item["steps"][1:], item["labels"]):
            evidence.setdefault((lo, hi), []).append((label, index))
    regions = []
    for lo, hi in zip(steps, steps[1:]):
        votes = evidence.get((lo, hi), [])
        labels = {label for label, _ in votes}
        status = next(iter(labels)) if len(labels) == 1 else "uncertain" if labels else "pending"
        indices = {index for _, index in votes}
        if regions and regions[-1]["status"] == status:
            regions[-1]["end_step"] = hi
            regions[-1]["window_indices"] = sorted(set(regions[-1]["window_indices"]) | indices)
        else:
            regions.append({"start_step": lo, "end_step": hi, "status": status,
                            "window_indices": sorted(indices)})
    return regions
