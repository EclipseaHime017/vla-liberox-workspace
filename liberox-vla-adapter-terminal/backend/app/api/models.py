from typing import Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    StrictFloat,
    StrictInt,
    StrictStr,
    model_validator,
)


PolicyCamera = Literal["agentview", "robot0_eye_in_hand"]


def _validate_camera_selection(cameras: list[PolicyCamera] | None) -> None:
    if cameras is None:
        return
    if len(cameras) != len(set(cameras)):
        raise ValueError("disabled_policy_cameras must not contain duplicates")
    if len(cameras) >= 2:
        raise ValueError("At least one VLA policy camera must remain enabled")


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class DraftRequest(StrictModel):
    task_id: str = Field(min_length=1)
    policy_id: str = Field(default="base", min_length=1)
    max_steps: int = Field(ge=1, le=10000)
    open_loop_steps: int = Field(ge=1, le=8)
    seed: int | None = Field(default=None, ge=0, le=2147483647)
    init_state_index: int = Field(default=0, ge=0)
    disabled_policy_cameras: list[PolicyCamera] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_cameras(self):
        _validate_camera_selection(self.disabled_policy_cameras)
        return self


class UpdateDraftRequest(StrictModel):
    task_id: str | None = Field(default=None, min_length=1)
    policy_id: str | None = Field(default=None, min_length=1)
    max_steps: int | None = Field(default=None, ge=1, le=10000)
    open_loop_steps: int | None = Field(default=None, ge=1, le=8)
    seed: int | None = Field(default=None, ge=0, le=2147483647)
    init_state_index: int | None = Field(default=None, ge=0)
    disabled_policy_cameras: list[PolicyCamera] | None = None

    @model_validator(mode="after")
    def require_update(self):
        if (
            self.task_id is None
            and self.policy_id is None
            and self.max_steps is None
            and self.open_loop_steps is None
            and self.seed is None
            and self.init_state_index is None
            and self.disabled_policy_cameras is None
        ):
            raise ValueError("At least one draft field must be provided")
        _validate_camera_selection(self.disabled_policy_cameras)
        return self


class CreateBranchRequest(StrictModel):
    resume_step: int = Field(ge=0)
    control_mode: Literal["policy", "manual"]
    open_loop_steps: int = Field(ge=1, le=8)
    translation_gain: float | None = Field(default=None, ge=0.05, le=1.0)
    rotation_gain: float | None = Field(default=None, ge=0.05, le=1.0)


class DeleteSessionRequest(StrictModel):
    confirm_session_id: str = Field(min_length=1)
    force: bool = False


DatasetSourceType = Literal["inference", "manual", "policy_requery"]
DatasetOutcome = Literal["success", "failure"]


class SelectionQuota(StrictModel):
    source_type: DatasetSourceType
    outcome: DatasetOutcome
    count: int = Field(ge=0)
    order: Literal["random", "oldest", "newest"] = "random"


class DatasetSelectionRequest(StrictModel):
    mode: Literal["random", "sequential", "rule", "manual"]
    size: int | None = Field(default=None, ge=1)
    seed: int = Field(default=0, ge=0, le=2147483647)
    order: Literal["oldest", "newest"] = "newest"
    source_types: list[DatasetSourceType] = Field(
        default_factory=lambda: ["inference", "manual", "policy_requery"]
    )
    outcomes: list[DatasetOutcome] = Field(
        default_factory=lambda: ["success", "failure"]
    )
    run_ids: list[str] = Field(default_factory=list)
    quotas: list[SelectionQuota] = Field(default_factory=list)


class DatasetPreviewRequest(StrictModel):
    task_id: str = Field(min_length=1)
    selection: DatasetSelectionRequest


class CreateTrainingDatasetRequest(DatasetPreviewRequest):
    name: str = Field(min_length=1, max_length=100)
    validation_fraction: float = Field(default=0.2, ge=0, le=0.9)
    split_seed: int = Field(default=7, ge=0, le=2147483647)
    success_consecutive_steps: int = Field(default=5, ge=1, le=100)


class DeriveTrainingDatasetRequest(StrictModel):
    name: str = Field(min_length=1, max_length=100)
    selection: DatasetSelectionRequest
    validation_fraction: float = Field(default=0.2, ge=0, le=0.9)
    split_seed: int = Field(default=7, ge=0, le=2147483647)
    success_consecutive_steps: int = Field(default=5, ge=1, le=100)


class DeleteTrainingDatasetRequest(StrictModel):
    confirm_dataset_id: str = Field(min_length=1)
    force: StrictBool = False


class TrainingRunRequest(StrictModel):
    dataset_id: str = Field(min_length=1)
    parameters: dict[str, StrictInt | StrictFloat | StrictStr | None]
