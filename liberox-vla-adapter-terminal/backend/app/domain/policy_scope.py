"""Filter recorded model identities, independently of the live model registry."""
from __future__ import annotations

UNKNOWN_FAMILY = "__unknown__"
UNKNOWN_POLICY = "__unknown__"


def matches_policy(run: dict, family: str | None = None, policy_id: str | None = None) -> bool:
    return (not family or (run.get("policy_family") or UNKNOWN_FAMILY) == family) and (
        not policy_id or (run.get("policy_id") or UNKNOWN_POLICY) == policy_id)


def policy_options(runs: list[dict]) -> list[dict]:
    values = {}
    for run in runs:
        identifier = run.get("policy_id") or UNKNOWN_POLICY
        family = run.get("policy_family") or UNKNOWN_FAMILY
        values[(family, identifier)] = dict(policy_id=identifier, family=family,
            label=run.get("policy_label") or (identifier if identifier != UNKNOWN_POLICY else "未记录模型"))
    return sorted(values.values(), key=lambda item: (item["family"], item["label"], item["policy_id"]))
