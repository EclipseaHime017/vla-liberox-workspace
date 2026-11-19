"""GET /api/methods serves the method catalog the training page renders."""
from __future__ import annotations

from fastapi import FastAPI
from fastapi.testclient import TestClient

from studio.api import methods


def test_methods_api_serves_catalog_and_default():
    app = FastAPI()
    app.include_router(methods.router)
    body = TestClient(app).get("/api/methods").json()
    assert body["default"] == "iql"
    assert [(item["id"], f"{item['label']} · {item['summary']}") for item in body["methods"]] == [
        ("iql", "IQL · 奖励加权后训练"), ("bc", "BC · 等权行为克隆"),
        ("ifql", "IFQL · flow 策略采样 + IQL critic 选择")]
    assert all({"requires_rewards", "critic", "critic_keys", "parameters", "families"} <= item.keys()
               for item in body["methods"])
    ifql = body["methods"][2]
    assert ifql["families"] == ["pi05"] and ifql["exports_critic"] and ifql["actor_capabilities"] == ["sample"]
    assert "beta" not in ifql["critic_keys"] and "expectile" in ifql["critic_keys"]
    assert [field["name"] for field in ifql["parameters"]] == ["num_samples"]
