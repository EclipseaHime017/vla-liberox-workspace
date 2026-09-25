import { api } from "../../api/client";
import type { ControllerId, ControllerStatus, Session } from "./types";

export const CONTROLLER_LABELS: Record<ControllerId, string> = {
  spacemouse: "SpaceMouse",
  factr: "FACTR Franka",
};

export function controllerStatusText(status: ControllerStatus | null) {
  if (!status) return "正在检测控制器";
  if (status.state === "ERROR") return "控制器异常，请查看诊断";
  if (!status.connected) return "控制器未连接";
  if (status.state === "CALIBRATING") return "控制器校准中";
  if (status.state === "ALIGNING") return "控制器正在对齐仿真姿态";
  if (status.state === "ARMED") return "控制器接管中";
  return status.calibrated ? "控制器已校准" : "控制器已连接 · 待校准";
}

export function controllerConnection(controllers: ControllerStatus[]) {
  if (!controllers.length) return { text: "正在检测控制器", level: "" };
  const connected = controllers.some((item) => item.connected && item.state !== "ERROR");
  return { text: connected ? "控制器已连接" : "控制器未连接", level: connected ? "green" : "" };
}

export const getControllers = () => api<{ controllers: ControllerStatus[] }>("/api/controllers");
export const getController = (id: ControllerId) => api<ControllerStatus>(
  `/api/controller?controller_id=${id}`,
);
export const calibrateController = (id: ControllerId) => (
  api<ControllerStatus>(`/api/controller/calibrate?controller_id=${id}`, {
    method: "POST",
    ...(id === "factr" ? { headers: { "X-FACTR-USB-Repair": "1" } } : {}),
  })
);

export const setControllerGravity = (enabled: boolean) => api<ControllerStatus>("/api/controller/gravity?controller_id=factr", {
  method: "POST", body: JSON.stringify({ enabled }),
});

// Live session telemetry is also delivered over the existing WebSocket. Legacy
// SpaceMouse sessions keep their old field names; FACTR never reads those fields.
export function controllerTelemetry(status: ControllerStatus | null, session: Session | null) {
  const isManual = session?.control_mode === "manual";
  const legacyMouse = session?.manual_source === "spacemouse";
  const latency = isManual
    ? session?.controller_latency_ms ?? (legacyMouse ? session?.spacemouse_latency_ms : null) ?? status?.latency_ms ?? null
    : status?.latency_ms ?? null;
  const stale = isManual
    ? session?.controller_stale ?? (legacyMouse ? session?.spacemouse_stale : null) ?? status?.stale
    : status?.stale;
  const connected = isManual
    ? session?.controller_connected ?? (legacyMouse ? session?.spacemouse_connected : null) ?? status?.connected
    : status?.connected;
  const showLatency = status?.state === "ARMED" || Boolean(isManual && session?.status === "RUNNING");
  const level = !connected || stale || status?.connected === false || status?.stale
    || status?.state === "ERROR" || status?.state === "DISCONNECTED" || latency === null || latency >= 250
    ? "red" : latency >= 50 ? "yellow" : "green";
  return { latency, showLatency, level };
}
