export type TaskInfo = {
  task_id: string;
  level: string;
  task_name: string;
  prompt: string;
  init_state_count: number;
  init_state_index_min: number;
  init_state_index_max: number;
};

export type PolicyInfo = {
  policy_id: string;
  label: string;
  base_checkpoint: string;
  stats_key: string;
  kind: "base" | "rynn_iql_overlay";
  training_step: number | null;
  compatibility_sha256: string | null;
};

export type PolicyDetail = PolicyInfo & {
  manifest: null | Record<string, string | number | null>;
  components: Array<{ name: string; filename: string; size_bytes: number; sha256: string }>;
  training_records: OfflineJob[];
};

export type PolicyCameraId = "agentview" | "robot0_eye_in_hand";

export type Bootstrap = {
  config: {
    max_steps: number; open_loop_steps: number; seed: number;
    disabled_policy_cameras: PolicyCameraId[]; control_hz: number; video_fps: number;
    preview: {
      width: number; height: number; fps: number; layout: "2x2";
      stream_width: number; stream_height: number;
      cameras: Array<{ id: string; label: string; policy_input: boolean }>;
      recorded_cameras: string[];
    };
    manual: { translation_gain: number; rotation_gain: number };
    spacemouse: {
      configured: boolean; dependency_version: string | null; config_error: string | null;
      device_name: string | null; vendor_id: number | null; product_id: number | null;
      stale_timeout_ms: number | null;
    };
  };
  model: { checkpoint: string; gpu: string; loaded: boolean; policy_id: string; policy_label: string; action_schema: { predicted_chunk_size: number } };
  policy_catalog: PolicyInfo[];
  task: TaskInfo;
  task_catalog: TaskInfo[];
  capabilities: { model_switching: boolean; task_switching: boolean };
  evaluation_capabilities?: Record<"rynnvalue" | "robometer", {
    available: boolean; reason: string | null;
  }>;
};

export type Draft = {
  id: string; task_id: string; max_steps: number; open_loop_steps: number;
  seed: number; init_state_index: number; disabled_policy_cameras: PolicyCameraId[];
  policy_id: string; policy_label: string;
  preview_status: "PREPARING" | "RENDERING" | "READY" | "ERROR";
  preview_revision: number; preview_ready: boolean; preview_available: boolean;
  error: string | null; task: TaskInfo;
};

export type PolicyBranchDraft = {
  parent_session_id: string;
  task_prompt: string;
  source_episode: string;
  resume_step: number;
  target_steps: number;
  open_loop_steps: number;
  policy_id: string;
  policy_label: string;
};

export type Session = {
  id: string; kind: "original" | "branch"; task_id: string | null; level: string | null;
  task_name: string | null; task: string | null; parent_session_id: string | null;
  resume_step?: number | null;
  control_mode: string; manual_source: "browser" | "spacemouse" | "factr" | null;
  policy_id: string; policy_label: string | null; policy_base_checkpoint: string | null;
  policy_overlay: string | null; policy_compatibility_sha256: string | null;
  manual_translation_gain: number | null; manual_rotation_gain: number | null;
  spacemouse_status: string | null; spacemouse_connected: boolean | null;
  spacemouse_stale: boolean | null; spacemouse_latency_ms: number | null;
  spacemouse_deadman_ms: number | null; status: string; created_at: string | null;
  controller_status?: string | null; controller_connected?: boolean | null;
  controller_stale?: boolean | null; controller_latency_ms?: number | null;
  controller_deadman_ms?: number | null;
  max_steps: number; open_loop_steps: number; current_step: number; state_count: number;
  seed: number; init_state_index: number; disabled_policy_cameras: PolicyCameraId[];
  action_count: number; policy_queries: number; success: boolean; error: string | null;
  stopped_reason: string | null; measured_control_hz: number | null;
  simulated_duration_seconds: number; branchable: boolean; legacy: boolean; managed: boolean;
  preparation_phase: string | null; preparation_message: string | null;
  countdown_remaining: number | null; preview_ready: boolean;
  preparation_timing: Record<string, number | null>; artifacts: Record<string, string>;
  source_type?: "inference" | "manual" | "policy_requery" | "incomplete";
  outcome?: "success" | "failure";
  training_eligible?: boolean; ineligible_reason?: string | null;
  training_start_step?: number; training_action_count?: number; training_chunk_count?: number;
  is_test?: boolean;
  rynn_evaluation?: {
    status: "NOT_EVALUATED" | "READY";
    evaluated_at?: string | null; model?: string | null; revision?: string | null;
    boundary_count?: number; source_key?: string | null;
  };
  robometer_evaluation?: {
    status: "NOT_EVALUATED" | "READY";
    evaluated_at?: string | null; model?: string | null; revision?: string | null;
    sample_count?: number; source_key?: string | null;
  };
};

export type PaginatedRuns = {
  items: Session[]; total: number; eligible_count: number; evaluated_count: number;
  rynn_evaluated_count: number; robometer_evaluated_count: number;
  both_evaluated_count: number; test_count: number;
  page: number; page_size: number; pages: number;
};

export type TrajectoryDetail = {
  global_evaluation?: { source: RewardSource; config: RewardParameters; evaluated_at?: string; origin?: string } | null;
  global_evaluation_pending?: boolean;
  global_evaluation_error?: string | null;
  reward_evaluations?: Partial<Record<TrainingRewardSource, DatasetRewardEvaluation>>;
  evaluation_sources?: Partial<Record<RewardSource, EvaluationSourceContext>>;
  dataset_context?: DatasetDetailContext | null;
  available_dataset_contexts?: Array<{
    dataset_id: string; dataset_name: string; reward_version_id: string | null;
    robometer_version_id: string | null; versions: RewardVersion[];
  }>;
  reward_evaluation?: DatasetRewardEvaluation | null;
  run: Session;
  artifacts: Record<string, string>;
  series: {
    time_seconds: number[]; action_time_seconds: number[];
    env_action: number[][]; raw_action: number[][];
    eef_position: number[][]; eef_axis_angle: number[][]; gripper_qpos: number[][];
    done: boolean[];
  };
  evaluation: null | RynnValueEvaluation;
  rynnvalue_evaluation: null | RynnValueEvaluation;
  robometer_evaluation: null | {
    status: "READY"; evaluated_at: string; model: string | null; revision: string | null;
    version_id?: string;
    observation_steps: number[]; time_seconds: number[];
    progress_pred: number[]; success_probs: number[];
    evaluation_config: Record<string, unknown>;
  };
};

export type RewardSource = "sparse" | "stage" | "rynnvalue" | "robometer";
export type TrainingRewardSource = Exclude<RewardSource, "robometer">;
export type EvaluationSourceContext = {
  status: string; origin: "global" | "dataset";
  config?: RewardParameters; error?: string | null;
  evaluated_at?: string | null; version_id?: string;
};
export type RewardParameters = {
  gamma?: number; shaping_weight?: number; stage_exponent?: number;
  accumulate_primitive_steps?: boolean; max_frames?: number; batch_size?: number;
  sampling_hz?: number; force_model?: boolean; checkpoint?: string; revision?: string;
  overwrite_global?: boolean;
  prefix_frames?: number;
};
export type RewardVersion = {
  id: string; evaluator: RewardSource; status: string; parameters: RewardParameters;
  created_at: string; completed_at?: string | null; error?: string | null; legacy?: boolean;
};
export type DatasetDetailContext = {
  dataset_id: string; dataset_name: string; version_id: string | null;
  source?: RewardSource | null; config?: RewardParameters; status?: string;
};
export type DatasetRewardEvaluation = {
  status: "READY"; version_id: string; reward_config: RewardParameters;
  source: "sparse" | "stage" | "rynnvalue";
  time_seconds?: number[]; stage_scores?: number[]; observation_steps?: number[];
  boundary_steps: number[]; chunk_lengths: number[];
  chunk_start_steps: number[]; chunk_end_steps: number[];
  sparse_reward?: number[]; dense_reward?: number[]; shape_reward?: number[]; final_reward: number[];
};

export type StageKeyframe = { step: number; kind: "positive" | "negative" };

export type StageAnnotation = {
  run_id: string; status: "missing" | "ready" | "stale"; error?: string | null;
  derivation_error?: string | null;
  action_count: number; time_seconds: number[]; success_step: number | null;
  success_consecutive_steps: number; exponent: number; keyframes: StageKeyframe[];
  scores: number[]; revision: string | null;
};

export type RynnValueEvaluation = {
    status: "READY"; evaluated_at: string; model: string | null;
    boundary_steps: number[];
    official_outputs: {
      absolute_temporal_distance_seconds: number[][];
      absolute_value_entropy_nats: number[][];
      absolute_value_logits: number[][][];
      relative_temporal_distance_seconds: number[];
      relative_value_logits: number[][];
      inference_method: string | null; prefix_image_slots: number | null;
      absolute_slot: string | null; relative_slot: string | null;
      analysis: null | {
        generated_text?: string; generated_token_ids?: number[];
        parsed_for_display?: {
          description?: string | null; match?: string | null; success?: string | null;
        };
      };
    };
    pbrs_reward: {
      sparse_reward: number[]; dense_reward: number[]; shape_reward: number[];
      final_reward: number[];
      chunk_start_steps: number[];
      chunk_end_steps: number[]; chunk_lengths: number[];
      accumulate_primitive_steps?: boolean;
      description?: string | null;
    };
    reward_config: Record<string, unknown>;
};

export type DatasetSelection = {
  mode: "random" | "sequential" | "rule" | "manual";
  size?: number | null; seed: number; order: "oldest" | "newest";
  source_types: Array<"inference" | "manual" | "policy_requery">;
  outcomes: Array<"success" | "failure">;
  run_ids: string[];
  quotas: Array<{
    source_type: "inference" | "manual" | "policy_requery";
    outcome: "success" | "failure"; count: number;
    order: "random" | "oldest" | "newest";
  }>;
};

export type DatasetPreview = {
  task_id: string; eligible_count: number; selected_count: number;
  action_count: number; chunk_count: number; categories: Record<string, number>;
  run_ids: string[]; runs: Session[];
};

export type TrainingDataset = {
  evaluation_version_ids?: Partial<Record<RewardSource, string | null>>;
  reward_version_id?: string | null; robometer_version_id?: string | null;
  evaluation_versions?: RewardVersion[];
  id: string; project_id: string; name: string; task_id: string;
  status: "FROZEN"; integrity_status: "HEALTHY" | "BROKEN";
  integrity_error: string | null; annotation_status: "NOT_STARTED" | "RUNNING" | "READY" | "ERROR" | "CANCELED";
  annotation_id: string | null; parent_dataset_id: string | null;
  annotation_config?: {
    max_frames?: number | null; accumulate_primitive_steps?: boolean | null;
  } | null;
  annotation_history?: Array<{
    annotation_id: string; status: string; completed_at: string;
    config?: {
      max_frames?: number | null; accumulate_primitive_steps?: boolean | null;
    } | null;
  }>;
  created_at: string; updated_at: string; member_count: number;
  action_count: number; chunk_count: number; categories: Record<string, number>;
  validation_fraction: number; split_seed: number; success_consecutive_steps: number;
  dataset_sha256: string; members: Array<{
    run_id: string; root_run_id: string; parent_run_id: string | null;
    source_type: string; outcome: string; resume_step: number; end_step: number;
    action_count: number; chunk_count: number; split: "train" | "validation";
  }>;
  automatic_evaluation_job_id?: string;
  automatic_evaluation_error?: string;
};

export type OfflineJob = {
  id: string; kind: "annotation" | "trajectory_evaluation" | "training" | "evaluation";
  status: "STARTING" | "RUNNING" | "STOPPING" | "COMPLETED" | "FAILED" | "CANCELED";
  dataset_id: string | null; created_at: string; started_at: string | null; completed_at: string | null;
  stage: string; stage_label: string; error: string | null; output_path: string;
  warning?: string | null;
  parameters: Record<string, unknown>; log_size: number;
  metrics?: Record<string, number | string | null>;
  training_summary?: Record<string, unknown>;
  evaluation_summary?: EvaluationAggregate & { evaluation_id?: string };
};

export type EvaluationStatus = OfflineJob["status"];

export type EvaluationConfig = {
  task_id: string;
  policy_id: string;
  trials: number;
  max_steps: number;
  open_loop_steps: number;
  realtime: boolean;
  init_state_indices: number[] | null;
  base_seed: number;
  seed_count: number | null;
  schedule_seed: number;
};

export type EvaluationScheduleItem = {
  trial_index: number;
  init_state_index: number;
  seed: number;
};

export type EvaluationBreakdown = {
  trials: number;
  successes: number;
  failures: number;
  errors: number;
  success_rate: number;
};

export type EvaluationTrial = {
  trial_index: number;
  init_state_index: number;
  seed: number;
  success: boolean;
  error: string | null;
  steps: number;
  first_success_step: number | null;
  max_done_streak: number;
  final_done: boolean;
  policy_queries: number;
  inference_latency_ms: number | null;
  measured_control_hz: number | null;
  deadline_misses: number;
  elapsed_seconds: number;
};

export type EvaluationAggregate = {
  total_trials: number;
  attempted_trials: number;
  completed_trials: number;
  successes: number;
  failures: number;
  errors: number;
  success_rate: number;
  wilson_lower: number;
  wilson_upper: number;
  completion_rate: number;
  by_init_state: Record<string, EvaluationBreakdown>;
  by_seed: Record<string, EvaluationBreakdown>;
  by_combination: Record<string, EvaluationBreakdown>;
  first_success_step_mean: number | null;
  policy_queries_mean: number | null;
  inference_latency_ms_mean: number | null;
  measured_control_hz_mean: number | null;
  elapsed_seconds_mean: number | null;
};

export type EvaluationPreview = {
  config: EvaluationConfig;
  schedule: EvaluationScheduleItem[];
  schedule_sha256: string;
  init_state_counts: Record<string, number>;
  seed_counts: Record<string, number>;
  combination_counts: Record<string, number>;
  estimated_duration_seconds: number;
};

export type EvaluationRecord = {
  id: string;
  status: EvaluationStatus;
  created_at: string;
  started_at: string | null;
  completed_at: string | null;
  task_id: string;
  task_name: string;
  task_prompt: string;
  policy_id: string;
  policy_label: string;
  base_checkpoint: string;
  overlay_id: string | null;
  training_step: number | null;
  compatibility_sha256: string | null;
  config: EvaluationConfig;
  schedule: EvaluationScheduleItem[];
  schedule_sha256: string;
  success_rule: {
    done_consecutive_steps: number;
    success_latched: boolean;
    run_full_horizon: boolean;
    errors_in_denominator: boolean;
  };
  trials: EvaluationTrial[];
  aggregate: EvaluationAggregate;
  model_load_seconds: number | null;
  wall_time_seconds: number | null;
  simulated_time_seconds: number | null;
  error: string | null;
  output_path: string;
};

export type EvaluationFilters = {
  task_id?: string;
  policy_id?: string;
  status?: EvaluationStatus;
  date_from?: string;
  date_to?: string;
};

export type TrainingDefaults = {
  reward_availability?: {
    pending?: boolean;
    ready: boolean; origin?: "dataset" | "global" | null;
    missing_run_ids?: string[]; errors?: { run_id: string; error: string }[];
  };
  reward_version?: RewardVersion | null;
  reward_parameters_locked?: boolean;
  reward_locked_parameters?: string[];
  reward_editable_parameters?: string[];
  basic: Record<string, number>;
  advanced: Record<string, number | string | boolean>;
  monitoring: Record<string, number | string | boolean | null>;
  fixed: Record<string, string | number | boolean>;
  environments: Record<string, string>;
  checkpoints: Array<{ path: string; label: string }>;
};

export type TensorBoardStatus = {
  url: string; running: boolean; managed: boolean; pid: number | null;
  logdir: string; starting?: boolean;
};

export type ControllerId = "spacemouse" | "factr";

export type ControllerStatus = {
  controller_id?: ControllerId;
  state: "DISCONNECTED" | "UNCALIBRATED" | "CALIBRATING" | "ALIGNING" | "READY" | "ARMED" | "ERROR";
  connected: boolean; calibrated: boolean; calibration_progress: number; movement_resets: number;
  message: string; error: string | null; armed_session_id: string | null;
  latency_ms: number | null; latency_level: "green" | "yellow" | "red" | null; stale: boolean;
  gravity_supported?: boolean;
  gravity_enabled?: boolean;
  gravity_state?: "on" | "off" | "unknown";
  cycle_ms?: number | null;
  reference_joint_positions?: number[];
  translation_gain?: number; rotation_gain?: number;
};

export type FrameState = {
  step: number; time_seconds: number; eef_position_m: number[]; eef_axis_angle_rad: number[];
  gripper_qpos: number[]; raw_action: number[] | null; env_action: number[] | null; success: boolean;
};

export type DatasetSummary = {
  project_id: string; dataset_root: string; catalog: string; runs: number; completed: number;
  errors: number; successes: number; success_rate: number; legacy_indexed: number;
  tasks: Array<{ task_id: string; task_name: string; level: string; runs: number; successes: number; success_rate: number }>;
};

export const ACTIVE = new Set(["LOADING", "READY", "RUNNING", "STOPPING", "POSTPROCESSING"]);
export const TERMINAL = new Set(["COMPLETED", "ERROR"]);
