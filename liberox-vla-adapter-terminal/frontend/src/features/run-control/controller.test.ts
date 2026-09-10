import { beforeEach, describe, expect, it, vi } from "vitest";
import { api } from "../../api/client";
import { calibrateController, setControllerGravity, controllerTelemetry, controllerConnection, controllerStatusText, getController, getControllers } from "./controller";
import type { ControllerStatus, Session } from "./types";

vi.mock("../../api/client", () => ({ api: vi.fn() }));
beforeEach(() => vi.clearAllMocks());

const status: ControllerStatus = {
  controller_id: "factr", state: "ARMED", connected: true,
  calibrated: true, calibration_progress: 1, movement_resets: 0,
  message: "接管中", error: null, armed_session_id: "branch",
  latency_ms: 5, latency_level: "green", stale: false,
};

describe("controller transport", () => {
  it("uses generic connection and calibration labels for either device", () => {
    const missing: ControllerStatus = {...status, state: "DISCONNECTED", connected: false, calibrated: false,
      message: "未发现 HID 设备 256f:c63a"};
    expect(controllerConnection([missing, {...status, controller_id: "factr"}]).text).toBe("控制器已连接");
    expect(controllerConnection([missing]).text).toBe("控制器未连接");
    for (const id of ["spacemouse", "factr"] as const) {
      expect(controllerStatusText({...missing, controller_id: id})).toBe("控制器未连接");
      expect(controllerStatusText({...status, controller_id: id, state: "CALIBRATING"})).toBe("控制器校准中");
      expect(controllerStatusText({...status, controller_id: id, state: "READY"})).toBe("控制器已校准");
    }
  });
  it("uses explicit IDs and keeps SpaceMouse calibration free of FACTR fields", () => {
    getControllers();
    getController("factr");
    calibrateController("spacemouse");
    calibrateController("factr");
    expect(api).toHaveBeenNthCalledWith(1, "/api/controllers");
    expect(api).toHaveBeenNthCalledWith(2, "/api/controller?controller_id=factr");
    expect(api).toHaveBeenNthCalledWith(3, "/api/controller/calibrate?controller_id=spacemouse", { method: "POST" });
    expect(api).toHaveBeenNthCalledWith(4, "/api/controller/calibrate?controller_id=factr", { method: "POST" });
    setControllerGravity(true);
    expect(api).toHaveBeenLastCalledWith("/api/controller/gravity?controller_id=factr", {method: "POST", body: '{"enabled":true}'});
  });
});

describe("live controller latency", () => {
  it.each([[49.99, "green"], [50, "yellow"], [249, "yellow"], [250, "red"]])("uses the shared latency thresholds for %s ms", (latency, color) => {
    expect(controllerTelemetry({ ...status, latency_ms: Number(latency) }, null).level).toBe(color);
  });

  it("uses WebSocket generic fields for FACTR without reusing SpaceMouse telemetry", () => {
    const session = {
      control_mode: "manual", manual_source: "factr", status: "RUNNING",
      controller_connected: true, controller_stale: false, controller_latency_ms: 60,
      spacemouse_latency_ms: 1,
    } as Session;
    expect(controllerTelemetry(status, session)).toEqual({ latency: 60, showLatency: true, level: "yellow" });
    expect(controllerTelemetry(status, { ...session, controller_stale: true }).level).toBe("red");
    expect(controllerTelemetry(status, { ...session, controller_connected: false }).level).toBe("red");
  });

  it("supports old SpaceMouse sessions", () => {
    const session = {
      control_mode: "manual", manual_source: "spacemouse", status: "RUNNING",
      spacemouse_connected: true, spacemouse_stale: false, spacemouse_latency_ms: 51,
    } as Session;
    expect(controllerTelemetry({ ...status, controller_id: "spacemouse" }, session).latency).toBe(51);
  });

  it("keeps a fault red even when the previous successful sample was recent", () => {
    expect(controllerTelemetry({ ...status, state: "ERROR" }, { control_mode: "manual", status: "RUNNING" } as Session).level).toBe("red");
    expect(controllerTelemetry({ ...status, state: "DISCONNECTED", connected: false }, {
      control_mode: "manual", status: "RUNNING", controller_connected: true,
      controller_stale: false, controller_latency_ms: 5,
    } as Session).level).toBe("red");
  });
});
