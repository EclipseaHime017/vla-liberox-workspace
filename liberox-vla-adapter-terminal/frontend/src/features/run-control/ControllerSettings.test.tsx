import { cleanup, fireEvent, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { ControllerSettings } from "./ControllerSettings";
import type { ControllerStatus } from "./types";
afterEach(cleanup);
const status: ControllerStatus = {
  controller_id: "factr", state: "UNCALIBRATED", connected: true, calibrated: false,
  calibration_progress: 0, movement_resets: 0, message: "待校准", error: null,
  armed_session_id: null, latency_ms: 5, latency_level: "green", stale: false,
  gravity_supported: true, gravity_enabled: false,
};
const props = () => ({controllerId: "factr" as const, controller: status, active: false,
  manualActive: false, busy: false, translationGain: .25, rotationGain: .25,
  onSelect: vi.fn(), onCalibrate: vi.fn(), onGravity: vi.fn(),
  onTranslationGain: vi.fn(), onRotationGain: vi.fn()});

describe("shared controller settings", () => {
  it("one calibration call replaces reference/open/closed phases", () => {
    const p = props(); render(<ControllerSettings {...p} />);
    fireEvent.click(screen.getByRole("button", {name: "校准控制器"}));
    expect(p.onCalibrate).toHaveBeenCalledExactlyOnceWith();
    expect(screen.queryByRole("button", {name: /端点|取消校准/})).toBeNull();
    expect((screen.getByRole("button", {name: "开启重力补偿"}) as HTMLButtonElement).disabled).toBe(true);
  });
  it("calibration stays disabled in simulation but gravity is independently controllable", () => {
    const p = props(); const view = render(<ControllerSettings {...p} active manualActive
      controller={{...status, state: "ARMED", calibrated: true}} />);
    expect((screen.getByRole("button", {name: "重新校准控制器"}) as HTMLButtonElement).disabled).toBe(true);
    fireEvent.click(screen.getByRole("button", {name: "开启重力补偿"}));
    expect(p.onGravity).toHaveBeenCalledWith(true);
    view.rerender(<ControllerSettings {...p} controller={{...status, state: "READY", calibrated: true, gravity_enabled: true}} />);
    expect(screen.getByText("补偿已开启")).toBeTruthy();
    expect((screen.getByLabelText("人工控制器") as HTMLSelectElement).disabled).toBe(true);
    fireEvent.click(screen.getByRole("button", {name: "关闭重力补偿"}));
    expect(p.onGravity).toHaveBeenLastCalledWith(false);
  });
  it("retains SpaceMouse controls with no motor buttons", () => {
    const p = props(); render(<ControllerSettings {...p} controllerId="spacemouse" />);
    expect(screen.queryByRole("button", {name: "开启重力补偿"})).toBeNull();
    expect(screen.getByRole("slider", {name: "位移增益"})).toBeTruthy();
    fireEvent.click(screen.getByRole("button", {name: "校准控制器"}));
    expect(p.onCalibrate).toHaveBeenCalledExactlyOnceWith();
  });
  it("shows faults and supports explicit reconnect, never auto-enables", () => {
    const p = props(); render(<ControllerSettings {...p} controller={{...status, state: "ERROR", connected: false, error: "bus fault"}} />);
    fireEvent.click(screen.getByRole("button", {name: "重新连接并校准"}));
    expect(screen.getByRole("alert").textContent).toContain("bus fault");
    expect(p.onGravity).not.toHaveBeenCalled();
  });
  it("locks setup while calibrating and displays progress", () => {
    render(<ControllerSettings {...props()} controller={{...status, state: "CALIBRATING"}} />);
    expect((screen.getByRole("button", {name: "校准控制器"}) as HTMLButtonElement).disabled).toBe(true);
    expect(screen.getByLabelText("校准进度")).toBeTruthy();
  });
  it("does not offer Cartesian gains for exact joint following", () => {
    const p = props(); render(<ControllerSettings {...p} active manualActive controller={{...status, calibrated: true, state: "ARMED"}} />);
    expect(screen.queryByRole("slider", {name: "位移增益"})).toBeNull();
    expect(screen.queryByText(/七关节|参考姿态|仿真保持/)).toBeNull();
  });
});
