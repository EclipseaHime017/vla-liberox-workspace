import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import CollectPage from "./CollectPage";
import { api } from "../api/client";
import { getController, getControllers } from "../features/run-control/controller";
import type { Bootstrap, ControllerStatus, Session } from "../features/run-control/types";

vi.mock("../api/client", () => ({ api: vi.fn(), ApiError: class extends Error {} }));
vi.mock("../features/run-control/controller", async (importOriginal) => ({
  ...await importOriginal<typeof import("../features/run-control/controller")>(),
  getController: vi.fn(), getControllers: vi.fn(),
}));
vi.mock("../api/websocket", () => ({
  sessionWebSocket: () => ({ close: vi.fn(), send: vi.fn(), readyState: 0 }),
}));
vi.mock("../features/run-control/SessionMonitor", () => ({ SessionMonitor: () => null }));

const device: ControllerStatus = {
  controller_id: "spacemouse", state: "READY", connected: true, calibrated: true,
  calibration_progress: 1, movement_resets: 0, message: "控制器已校准",
  error: null, armed_session_id: null, latency_ms: 10, latency_level: "green", stale: false,
};
const source = {
  id: "source", kind: "original", control_mode: "policy", manual_source: null,
  status: "COMPLETED", branchable: true, managed: true, state_count: 11, action_count: 10,
  current_step: 10, max_steps: 10, open_loop_steps: 8, simulated_duration_seconds: 0.5,
  artifacts: {}, policy_queries: 2, policy_id: "base", policy_label: "Base",
  level: "LEVEL1", task_id: "task", task_name: "pick", task: "pick bowl", disabled_policy_cameras: [],
} as unknown as Session;
const bootstrap = {
  config: {
    max_steps: 300, open_loop_steps: 8, seed: 0, control_hz: 20, video_fps: 20,
    disabled_policy_cameras: [], preview: { width: 512, height: 512, stream_width: 1024, stream_height: 1024 },
    manual: { translation_gain: 0.25, rotation_gain: 0.08 },
  },
  model: { gpu: "cpu", checkpoint: "base", policy_label: "Base", action_schema: { predicted_chunk_size: 8 } },
  task: { task_id: "task", task_name: "pick", prompt: "pick bowl", level: "LEVEL1", init_state_index_min: 0 },
  task_catalog: [], policy_catalog: [],
} as unknown as Bootstrap;

beforeEach(() => {
  vi.clearAllMocks();
  vi.mocked(getController).mockImplementation(async (id) => ({ ...device, controller_id: id, gravity_enabled: id === "factr" }));
  vi.mocked(getControllers).mockImplementation(async () => ({
    controllers: await Promise.all([getController("spacemouse"), getController("factr")]),
  }));
  vi.mocked(api).mockImplementation(async (path) => {
    if (path === "/api/draft") return { discarded: false };
    if (path === "/api/bootstrap") return bootstrap;
    if (path === "/api/sessions") return [source];
    if (path.includes("/frames/")) return { step: 0, time_seconds: 0 };
    if (path === "/api/sessions/source/branches") return {
      ...source, id: "factr-branch", kind: "branch", control_mode: "manual", manual_source: "factr",
      status: "READY", managed: true, branchable: false, manual_translation_gain: 0.25, manual_rotation_gain: 0.25,
    };
    throw new Error(`Unexpected request: ${path}`);
  });
});
afterEach(cleanup);

describe("collection controller integration", () => {
  it("shows recorded FACTR history like other manual sessions", async () => {
    const original = vi.mocked(api).getMockImplementation()!;
    vi.mocked(api).mockImplementation(async (path, options) => path === "/api/sessions"
      ? [{...source, id: "factr-recording", managed: true, branchable: false,
          control_mode: "manual", manual_source: "factr", manual_translation_gain: .25, manual_rotation_gain: .25}]
      : original(path, options));
    render(<CollectPage />);
    await screen.findByText("回溯与分支");
    expect(screen.queryByText(/不记录轨迹或视频/)).toBeNull();
    await waitFor(() => expect(vi.mocked(api).mock.calls.some(([path]) => path.includes("/frames/"))).toBe(true));
  });

  it("shows generic connection and selects the available FACTR instead of missing HID", async () => {
    vi.mocked(getController).mockImplementation(async (id) => id === "spacemouse"
      ? {...device, state: "DISCONNECTED", connected: false, calibrated: false,
         message: "未发现 HID 设备 256f:c63a", error: "未发现 HID 设备 256f:c63a"}
      : {...device, controller_id: id});
    render(<CollectPage />);
    await waitFor(() => expect((screen.getByLabelText("人工控制器") as HTMLSelectElement).value).toBe("factr"));
    expect(screen.getByLabelText("控制器连接状态").textContent).toBe("控制器已连接");
    expect(screen.getByLabelText("控制器连接状态").getAttribute("title")).toBeNull();
    expect(screen.queryByText(/未发现 HID/)).toBeNull();
  });

  it("defaults to SpaceMouse and submits the selected FACTR source only for a manual branch", async () => {
    render(<CollectPage />);
    const selector = await screen.findByLabelText("人工控制器") as HTMLSelectElement;
    expect(selector.value).toBe("spacemouse");
    fireEvent.change(selector, { target: { value: "factr" } });
    await waitFor(() => expect(getController).toHaveBeenCalledWith("factr"));
    const takeover = await screen.findByRole("button", { name: "FACTR Franka 接管" });
    await waitFor(() => expect((takeover as HTMLButtonElement).disabled).toBe(false));
    expect(screen.queryByRole("slider", { name: "旋转增益" })).toBeNull();
    fireEvent.click(takeover);
    await waitFor(() => expect(api).toHaveBeenCalledWith("/api/sessions/source/branches", {
      method: "POST",
      body: JSON.stringify({ resume_step: 0, control_mode: "manual", open_loop_steps: 8,
        controller_id: "factr", translation_gain: 0.25, rotation_gain: 0.25 }),
    }));
    await waitFor(() => expect((screen.getByLabelText("人工控制器") as HTMLSelectElement).disabled).toBe(true));
  });

  it("blocks FACTR takeover before official calibration finishes", async () => {
    vi.mocked(getController).mockImplementation(async (id) => id === "factr"
      ? { ...device, controller_id: id, state: "UNCALIBRATED", calibrated: false }
      : device);
    render(<CollectPage />);
    fireEvent.change(await screen.findByLabelText("人工控制器"), { target: { value: "factr" } });
    await screen.findByRole("button", { name: "开启重力补偿" });
    expect((screen.getByRole("button", { name: "FACTR Franka 接管" }) as HTMLButtonElement).disabled).toBe(true);
  });

  it("preserves SpaceMouse gains when switching through joint-only FACTR", async () => {
    vi.mocked(getControllers).mockResolvedValue({ controllers: [device, {
      ...device, controller_id: "factr", translation_gain: 0.12, rotation_gain: 0.34,
    }] });
    vi.mocked(getController).mockImplementation(async (id) => ({
      ...device, controller_id: id, translation_gain: 0.12, rotation_gain: 0.34,
    }));
    render(<CollectPage />);
    const selector = await screen.findByLabelText("人工控制器");
    const translation = screen.getByRole("slider", { name: "位移增益" }) as HTMLInputElement;
    const rotation = screen.getByRole("slider", { name: "旋转增益" }) as HTMLInputElement;
    fireEvent.change(translation, { target: { value: "0.6" } });
    fireEvent.change(rotation, { target: { value: "0.7" } });
    fireEvent.change(selector, { target: { value: "factr" } });
    await waitFor(() => expect((selector as HTMLSelectElement).value).toBe("factr"));
    expect(screen.queryByRole("slider", { name: "位移增益" })).toBeNull();
    fireEvent.change(selector, { target: { value: "spacemouse" } });
    expect((screen.getByRole("slider", { name: "位移增益" }) as HTMLInputElement).value).toBe("0.6");
    expect((screen.getByRole("slider", { name: "旋转增益" }) as HTMLInputElement).value).toBe("0.7");
  });
});
