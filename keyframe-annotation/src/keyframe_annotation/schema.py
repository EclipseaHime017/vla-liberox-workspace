import json


def json_response(text):
    value = text.strip()
    if value.startswith("```json") and value.endswith("```"):
        value = value[7:-3].strip()
    def pairs(items):
        result = {}
        for key, item in items:
            if key in result:
                raise ValueError(f"Duplicate JSON key: {key}")
            result[key] = item
        return result
    result = json.loads(value, object_pairs_hook=pairs)
    if not isinstance(result, dict):
        raise ValueError("Model must return a JSON object")
    return result


def validate_contract(value, instruction):
    if set(value) != {"requirements"} or not isinstance(value["requirements"], list) or not 1 <= len(value["requirements"]) <= 5:
        raise ValueError("Task contract must contain 1-5 requirements")
    seen = set()
    for requirement in value["requirements"]:
        if not isinstance(requirement, dict) or set(requirement) != {"id", "instruction_span", "condition", "depends_on"}:
            raise ValueError("Invalid requirement fields")
        if any(not isinstance(requirement[k], str) or not requirement[k].strip() or len(requirement[k]) > 2000
               for k in ("id", "instruction_span", "condition")):
            raise ValueError("Invalid requirement text")
        if requirement["id"] in seen:
            raise ValueError("Duplicate requirement ID")
        if requirement["instruction_span"] not in instruction:
            raise ValueError("instruction_span must quote an exact clause from the original instruction")
        _dependencies(requirement["depends_on"], seen)
        seen.add(requirement["id"])
    return value


def _dependencies(deps, seen):
    if not isinstance(deps, list) or any(not isinstance(d, str) or d not in seen for d in deps) or len(set(deps)) != len(deps):
        raise ValueError("depends_on must name unique previously listed IDs")


def validate_plan(value, contract):
    if set(value) != {"initial_scene", "stages", "notes"}:
        raise ValueError("Task plan must contain initial_scene, stages and notes")
    if not isinstance(value["initial_scene"], str) or not value["initial_scene"].strip():
        raise ValueError("initial_scene MUST be one nonempty text STRING, NOT an object or list; combine the scene facts as text")
    if not isinstance(value["notes"], str):
        raise ValueError("notes must be one string, not a list; preserve the stage list and combine notes as text")
    stages = value["stages"]
    if not isinstance(stages, list) or not 1 <= len(stages) <= 5:
        raise ValueError("Expected 1-5 discrete completion milestones covering every requirement")
    required = {r["id"]: r for r in contract["requirements"]}
    seen, owners, ancestors = set(), {}, {}
    for stage in stages:
        if not isinstance(stage, dict) or set(stage) != {"id", "label", "achieved_when", "lost_when", "initial_state", "initial_reason", "depends_on", "requirement_ids"}:
            raise ValueError("Invalid stage fields")
        if any(not isinstance(v, str) or not v.strip() or len(v) > 2000 for k, v in stage.items() if k not in {"depends_on", "requirement_ids"}):
            raise ValueError("Invalid stage text")
        if stage["id"] in seen:
            raise ValueError("Duplicate stage ID")
        deps = stage["depends_on"]
        _dependencies(deps, seen)
        ancestors[stage["id"]] = set(deps).union(*(ancestors[d] for d in deps))
        refs = stage["requirement_ids"]
        if not isinstance(refs, list) or len(refs) > 1 or any(not isinstance(r, str) or r not in required for r in refs):
            raise ValueError("Each milestone must reference one known requirement ID, or [] for an auxiliary event")
        for ref in refs:
            if ref in owners:
                raise ValueError(f"Requirement {ref} must belong to exactly one milestone")
            owners[ref] = stage["id"]
        if stage["initial_state"] not in {"met", "unmet", "unknown"}:
            raise ValueError("Invalid initial_state")
        seen.add(stage["id"])
    if required.keys() - owners.keys():
        raise ValueError(f"Plan omits required outcomes: {sorted(required.keys() - owners.keys())}; include each as a milestone")
    for ref, requirement in required.items():
        for previous in requirement["depends_on"]:
            if owners[previous] not in ancestors[owners[ref]]:
                raise ValueError(f"Requirement {ref} must depend on the milestone covering {previous}")
    return value


def _region_status(label, stage_id, span_id):
    if isinstance(label, str) and label in {"confirmed", "unrelated", "uncertain"}:
        return {"confirmed": "confirmed", "unrelated": "outside", "uncertain": "uncertain"}[label]
    # Quoted literals are a serialization difference, not a new visual judgment.
    if isinstance(label, str) and label in {"true", "false", "null"}:
        return {"true": "confirmed", "false": "outside", "null": "uncertain"}[label]
    if type(label) is bool or label is None:
        return {True: "confirmed", False: "outside", None: "uncertain"}[label]
    raise ValueError(f"Invalid label at {stage_id}.labels[{span_id!r}]: {label!r}; "
                     "expected confirmed, unrelated or uncertain (legacy boolean literals also accepted)")


def _region_goals(value):
    """Normalize explicit IDs only; missing goals and judgments are never inferred."""
    if not isinstance(value, dict) or set(value) != {"goals"}:
        raise ValueError("Region response requires exactly one top-level goals field")
    goals = value["goals"]
    if isinstance(goals, list):
        return goals
    if isinstance(goals, dict):
        normalized = []
        for identifier, item in goals.items():
            if not isinstance(item, dict) or set(item) not in ({"labels"}, {"reason", "labels"}):
                raise ValueError(f"goals[{identifier!r}] must contain labels and optionally a short reason")
            normalized.append({"stage_id": identifier, **item})
        return normalized
    raise ValueError("goals must be a list of stage_id/labels objects or an object keyed by stage ID")


def validate_regions(value, stages, steps, *, require_reason=False):
    goals = _region_goals(value)
    expected = {stage["id"] for stage in stages}
    found = set()
    parsed = []
    span_ids = [f"{lo}:{hi}" for lo, hi in zip(steps, steps[1:])]
    for goal in goals:
        if not isinstance(goal, dict) or set(goal) not in ({"stage_id", "labels"}, {"stage_id", "reason", "labels"}):
            raise ValueError("Invalid region fields")
        identifier = goal["stage_id"]
        if not isinstance(identifier, str) or identifier not in expected or identifier in found:
            raise ValueError("Return every supplied stage_id exactly once")
        found.add(identifier)
        labels = goal["labels"]
        if not isinstance(labels, dict) or set(labels) != set(span_ids):
            raise ValueError(f"labels must be an object with exactly these start:end keys: {span_ids}")
        if "reason" in goal or require_reason:
            reason = goal.get("reason")
            if not isinstance(reason, str) or not reason.strip() or len(reason) > 600 or "<brief visible evidence>" in reason:
                raise ValueError(f"{identifier}.reason must be a short, nonempty visual observation")
        if require_reason and any(not isinstance(label, str) or label not in {"confirmed", "unrelated", "uncertain"}
                                  for label in labels.values()):
            raise ValueError(f"{identifier}: use string labels confirmed, unrelated or uncertain, not boolean/null")
        parsed.append({**goal, "labels": [_region_status(labels[key], identifier, key) for key in span_ids]})
    if found != expected:
        raise ValueError(f"Missing stage IDs: {sorted(expected - found)}")
    return {"goals": parsed}
