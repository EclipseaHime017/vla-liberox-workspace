"""Validated inputs of the Value Flows return distribution tool."""
from pydantic import BaseModel, ConfigDict, Field, StrictInt, StrictStr


class ReturnDistributionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    task_id: StrictStr = Field(min_length=1)
    policy_id: StrictStr = Field(min_length=1)
    max_steps: StrictInt = Field(default=300, ge=1, le=10000)
    open_loop_steps: StrictInt | None = Field(default=None, ge=1)
    seed: StrictInt = Field(default=0, ge=0, le=2147483647)
    init_state_index: StrictInt = Field(default=0, ge=0)
    return_samples: StrictInt = Field(default=256, ge=16, le=4096)
