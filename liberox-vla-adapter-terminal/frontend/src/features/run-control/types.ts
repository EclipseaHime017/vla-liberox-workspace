export type TaskInfo = {
  task_id: string;
  level: string;
  task_name: string;
  prompt: string;
  init_state_index: number;
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
};

export type Draft = {
  id: string; task_id: string; max_steps: number; open_loop_steps: number;
  seed: number; disabled_policy_cameras: PolicyCameraId[];
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
  control_mode: string; manual_source: "browser" | "spacemouse" | null;
  policy_id: string; policy_label: string | null; policy_base_checkpoint: string | null;
  policy_overlay: string | null; policy_compatibility_sha256: string | null;
  manual_translation_gain: number | null; manual_rotation_gain: number | null;
  spacemouse_status: string | null; spacemouse_connected: boolean | null;
  spacemouse_stale: boolean | null; spacemouse_latency_ms: number | null;
  spacemouse_deadman_ms: number | null; status: string; created_at: string | null;
  max_steps: number; open_loop_steps: number; current_step: number; state_count: number;
  seed: number; disabled_policy_cameras: PolicyCameraId[];
  action_count: number; policy_queries: number; success: boolean; error: string | null;
  stopped_reason: string | null; measured_control_hz: number | null;
  simulated_duration_seconds: number; branchable: boolean; legacy: boolean; managed: boolean;
  preparation_phase: string | null; preparation_message: string | null;
  countdown_remaining: number | null; preview_ready: boolean;
  preparation_timing: Record<string, number | null>; artifacts: Record<string, string>;
};

export type ControllerStatus = {
  state: "DISCONNECTED" | "UNCALIBRATED" | "CALIBRATING" | "READY" | "ARMED" | "ERROR";
  connected: boolean; calibrated: boolean; calibration_progress: number; movement_resets: number;
  message: string; error: string | null; armed_session_id: string | null;
  latency_ms: number | null; latency_level: "green" | "yellow" | "red" | null; stale: boolean;
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
