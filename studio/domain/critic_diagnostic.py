"""Validated inputs shared by the critic tool's HTTP and service adapters."""
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, StrictInt, StrictStr


class CriticDiagnosticRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    dataset_id: StrictStr = Field(min_length=1)
    reward_source: Literal["final", "rynnvalue", "stage", "sparse"]
    task_id: StrictStr = Field(min_length=1)
    policy_id: StrictStr = Field(min_length=1)
    max_steps: StrictInt = Field(default=300, ge=1, le=10000)
    open_loop_steps: StrictInt | None = Field(default=None, ge=1)
    seed: StrictInt = Field(default=0, ge=0, le=2147483647)
    init_state_index: StrictInt = Field(default=0, ge=0)
    train_steps: StrictInt = Field(default=3000, ge=1, le=1000000)
    batch_size: StrictInt = Field(default=32, ge=1, le=4096)
