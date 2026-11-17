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
        ("iql", "IQL · 奖励加权后训练"), ("bc", "BC · 等权行为克隆")]
    assert all({"requires_rewards", "critic", "parameters", "families"} <= item.keys() for item in body["methods"])
