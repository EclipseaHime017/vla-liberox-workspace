from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class DraftRequest(StrictModel):
    task_id: str = Field(min_length=1)
    policy_id: str = Field(default="base", min_length=1)
    max_steps: int = Field(ge=1, le=10000)
    open_loop_steps: int = Field(ge=1, le=8)


class UpdateDraftRequest(StrictModel):
    task_id: str | None = Field(default=None, min_length=1)
    policy_id: str | None = Field(default=None, min_length=1)
    max_steps: int | None = Field(default=None, ge=1, le=10000)
    open_loop_steps: int | None = Field(default=None, ge=1, le=8)

    @model_validator(mode="after")
    def require_update(self):
        if (
            self.task_id is None
            and self.policy_id is None
            and self.max_steps is None
            and self.open_loop_steps is None
        ):
            raise ValueError("At least one draft field must be provided")
        return self


class CreateBranchRequest(StrictModel):
    resume_step: int = Field(ge=0)
    control_mode: Literal["policy", "manual"]
    open_loop_steps: int = Field(ge=1, le=8)
    translation_gain: float | None = Field(default=None, ge=0.05, le=1.0)
    rotation_gain: float | None = Field(default=None, ge=0.05, le=1.0)


class DeleteSessionRequest(StrictModel):
    confirm_session_id: str = Field(min_length=1)
