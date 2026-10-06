import { act, cleanup, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import CollectPage from "./CollectPage";
import App from "../app/App";
import { api, ApiError } from "../api/client";
import { getController, getControllers } from "../features/run-control/controller";
import { sessionWebSocket } from "../api/websocket";
import type { Bootstrap, ControllerStatus, Session } from "../features/run-control/types";

vi.mock("../api/client", async (original) => ({ ...await original<typeof import("../api/client")>(), api: vi.fn() }));
vi.mock("../features/run-control/controller", async (importOriginal) => ({
  ...await importOriginal<typeof import("../features/run-control/controller")>(),
  getController: vi.fn(), getControllers: vi.fn(),
}));
vi.mock("../api/websocket", () => ({
  sessionWebSocket: vi.fn(() => ({ close: vi.fn(), send: vi.fn(), readyState: 0 })),
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
    manual: { translation_gain: 0.5, rotation_gain: 0.25 },
  },
  model: { gpu: "cpu", checkpoint: "base", policy_label: "Base", action_schema: { predicted_chunk_size: 8 } },
  task: { task_id: "task", task_name: "pick", prompt: "pick bowl", level: "LEVEL1", init_state_index_min: 0 },
  task_catalog: [], policy_catalog: [],
} as unknown as Bootstrap;

beforeEach(() => {
  vi.clearAllMocks();
  vi.mocked(sessionWebSocket).mockImplementation(() => ({
    close: vi.fn(), send: vi.fn(), readyState: WebSocket.CONNECTING,
  }) as unknown as WebSocket);
  vi.mocked(getController).mockImplementation(async (id) => ({ ...device, controller_id: id, gravity_enabled: id === "factr" }));
  vi.mocked(getControllers).mockImplementation(async () => ({
    controllers: await Promise.all([getController("spacemouse"), getController("factr")]),
  }));
  let branchCount = 0;
  vi.mocked(api).mockImplementation(async (path, options) => {
    if (path === "/api/draft") return { discarded: false };
    if (path === "/api/bootstrap") return bootstrap;
    if (path === "/api/build-info") return {dist_fingerprint: "test-build"};
    if (path === "/api/sessions") return [source];
    if (path.includes("/frames/")) return { step: 0, time_seconds: 0 };
    if (path === "/api/sessions/source/branches") {
      const request = JSON.parse(String(options?.body));
      branchCount += 1;
      return {...source, id: `${request.controller_id}-branch${branchCount > 1 ? `-${branchCount}` : ""}`, kind: "branch", control_mode: "manual",
        manual_source: request.controller_id, status: "READY", managed: true, branchable: false,
        manual_translation_gain: request.translation_gain, manual_rotation_gain: request.rotation_gain,
        manual_control_frame: request.control_frame};
    }
    throw new Error(`Unexpected request: ${path}`);
  });
});
afterEach(cleanup);

describe("collection controller integration", () => {
  it("shows missing historical model as WARN without falling back to base", async () => {
    const original = vi.mocked(api).getMockImplementation()!;
    vi.mocked(api).mockImplementation(async (path, options) => {
      if (path.endsWith("/branches")) throw new ApiError(409, { severity: "warning" }, "原策略模型已删除");
      return original(path, options);
    });
    render(<CollectPage />);
    fireEvent.click(await screen.findByRole("button", { name: "从此帧重新推理" }));
    fireEvent.click(screen.getByRole("button", { name: "开始二次推理" }));
    const warning = await screen.findByRole("alert");
    expect(warning.className).toContain("warning-banner");
    expect(warning.textContent).toContain("WARN");
    expect(warning.textContent).toContain("原策略模型已删除");
    expect(vi.mocked(api).mock.calls.filter(([path]) => path.endsWith("/branches"))).toHaveLength(1);
  });
  it("shows live status in the top bar without remounting the console when navigating", async () => {
    const original = vi.mocked(api).getMockImplementation()!;
    vi.mocked(api).mockImplementation(async (path, options) => path === "/api/sessions"
      ? [{...source, id: "live", status: "RUNNING", control_mode: "manual", manual_source: "spacemouse",
          manual_translation_gain: .5, manual_rotation_gain: .5}]
      : original(path, options));
    render(<App />);
    const header = screen.getByRole("banner");
    const stream = await screen.findByAltText("主视角、腕部、左侧和右侧实时画面");
    const connection = within(header).getByLabelText("控制器连接状态");
    expect(within(header).getByText("LIBERO-X仿真与干预控制台").tagName).toBe("STRONG");
    expect(within(header).getByLabelText("系统状态").textContent).toContain("RUNNING · live");
    expect(screen.queryByRole("heading", {level: 1})).toBeNull();
    expect(screen.queryByText("LOCAL ROBOTICS WORKBENCH")).toBeNull();
    const socket = vi.mocked(sessionWebSocket).mock.results[0].value as WebSocket;
    fireEvent.click(screen.getByRole("button", {name: /设置/}));
    expect(connection.closest("[hidden]")).not.toBeNull();
    fireEvent.click(screen.getByRole("button", {name: /控制台/}));
    expect(connection.closest("[hidden]")).toBeNull();
    expect(screen.getByAltText("主视角、腕部、左侧和右侧实时画面")).toBe(stream);
    expect(sessionWebSocket).toHaveBeenCalledTimes(1);
    expect(socket.close).not.toHaveBeenCalled();
  });

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
    fireEvent.change(screen.getByLabelText("控制坐标"), {target: {value: "tool"}});
    fireEvent.change(selector, { target: { value: "factr" } });
    await waitFor(() => expect((selector as HTMLSelectElement).value).toBe("factr"));
    expect(screen.queryByRole("slider", { name: "位移增益" })).toBeNull();
    fireEvent.change(selector, { target: { value: "spacemouse" } });
    expect((screen.getByRole("slider", { name: "位移增益" }) as HTMLInputElement).value).toBe("0.6");
    expect((screen.getByRole("slider", { name: "旋转增益" }) as HTMLInputElement).value).toBe("0.7");
    expect((screen.getByLabelText("控制坐标") as HTMLSelectElement).value).toBe("tool");
  });

  it("submits the selected tool frame only for SpaceMouse takeover", async () => {
    render(<CollectPage />);
    fireEvent.change(await screen.findByLabelText("控制坐标"), {target: {value: "tool"}});
    const takeover = await screen.findByRole("button", {name: "SpaceMouse 接管"});
    await waitFor(() => expect((takeover as HTMLButtonElement).disabled).toBe(false));
    fireEvent.click(takeover);
    await waitFor(() => expect(api).toHaveBeenCalledWith("/api/sessions/source/branches", {
      method: "POST", body: JSON.stringify({resume_step: 0, control_mode: "manual", open_loop_steps: 8,
        controller_id: "spacemouse", translation_gain: .5, rotation_gain: .25, control_frame: "tool"}),
    }));
  });

  it("preserves tool mode on reconnect and gain changes, sending frame only on explicit selection", async () => {
    const socket = {readyState: WebSocket.CONNECTING, close: vi.fn(), send: vi.fn(), onopen: null} as unknown as WebSocket;
    vi.mocked(sessionWebSocket).mockReturnValueOnce(socket);
    const original = vi.mocked(api).getMockImplementation()!;
    vi.mocked(api).mockImplementation(async (path, options) => path === "/api/sessions"
      ? [source, {...source, id: "active-mouse", kind: "branch", status: "RUNNING", control_mode: "manual",
          manual_source: "spacemouse", manual_control_frame: "tool", manual_requested_control_frame: "tool",
          manual_translation_gain: .25, manual_rotation_gain: .08}]
      : original(path, options));
    render(<CollectPage />);
    await waitFor(() => expect((screen.getByLabelText("控制坐标") as HTMLSelectElement).value).toBe("tool"));
    expect((screen.getByRole("slider", {name: "位移增益"}) as HTMLInputElement).value).toBe("0.25");
    expect((screen.getByRole("slider", {name: "旋转增益"}) as HTMLInputElement).value).toBe("0.08");
    await waitFor(() => expect(socket.onopen).toBeTruthy());
    fireEvent.change(screen.getByRole("slider", {name: "位移增益"}), {target: {value: ".5"}});
    // Opening a delayed connection must send the latest slider value, not the
    // initial gain captured from the running session.
    expect(sessionWebSocket).toHaveBeenLastCalledWith("active-mouse");
    expect(socket.send).not.toHaveBeenCalled();
    Object.assign(socket, {readyState: WebSocket.OPEN});
    socket.onopen?.call(socket, new Event("open"));
    expect(JSON.parse(String(vi.mocked(socket.send).mock.lastCall?.[0]))).toMatchObject({
      translation_gain: .5, rotation_gain: .08,
    });
    for (const [message] of vi.mocked(socket.send).mock.calls) {
      expect(JSON.parse(String(message))).not.toHaveProperty("control_frame");
    }
    fireEvent.change(screen.getByLabelText("控制坐标"), {target: {value: "world"}});
    expect(JSON.parse(String(vi.mocked(socket.send).mock.lastCall?.[0]))).toMatchObject({
      type: "manual_settings", translation_gain: .5, rotation_gain: .08, control_frame: "world",
    });
  });

  it.each(["spacemouse", "factr"])("does not restore %s history gains as current preferences", async (manual_source) => {
    const original = vi.mocked(api).getMockImplementation()!;
    vi.mocked(api).mockImplementation(async (path, options) => path === "/api/sessions"
      ? [{...source, id: "old-manual", control_mode: "manual", manual_source,
          manual_translation_gain: .1, manual_rotation_gain: .2}, source]
      : original(path, options));
    render(<CollectPage />);
    const translation = await screen.findByRole("slider", {name: "位移增益"}) as HTMLInputElement;
    const rotation = screen.getByRole("slider", {name: "旋转增益"}) as HTMLInputElement;
    expect(translation.value).toBe("0.5");
    expect(rotation.value).toBe("0.25");
    fireEvent.change(translation, {target: {value: ".6"}});
    fireEvent.change(rotation, {target: {value: ".7"}});
    fireEvent.click(screen.getByRole("button", {name: /原始 · source /}));
    fireEvent.click(screen.getByRole("button", {name: /原始 · old-manual /}));
    expect(translation.value).toBe("0.6");
    expect(rotation.value).toBe("0.7");
  });

  it("keeps live gain edits through completion and the next takeover", async () => {
    vi.mocked(sessionWebSocket).mockImplementation(() => ({
      readyState: WebSocket.OPEN, close: vi.fn(), send: vi.fn(),
    }) as unknown as WebSocket);
    render(<CollectPage />);
    const takeover = await screen.findByRole("button", {name: "SpaceMouse 接管"});
    await waitFor(() => expect((takeover as HTMLButtonElement).disabled).toBe(false));
    fireEvent.click(takeover);
    await waitFor(() => expect(sessionWebSocket).toHaveBeenLastCalledWith("spacemouse-branch"));
    const socket = vi.mocked(sessionWebSocket).mock.results.at(-1)!.value as WebSocket;
    fireEvent.change(screen.getByRole("slider", {name: "位移增益"}), {target: {value: ".6"}});
    fireEvent.change(screen.getByRole("slider", {name: "旋转增益"}), {target: {value: ".7"}});
    expect(JSON.parse(String(vi.mocked(socket.send).mock.lastCall?.[0]))).toMatchObject({
      type: "manual_settings", translation_gain: .6, rotation_gain: .7,
    });
    act(() => socket.onmessage?.call(socket, new MessageEvent("message", {data: JSON.stringify({
      type: "session", session: {...source, id: "spacemouse-branch", kind: "branch", branchable: false,
        control_mode: "manual", manual_source: "spacemouse", manual_translation_gain: .5, manual_rotation_gain: .5},
    })})));
    fireEvent.click(screen.getByRole("button", {name: /原始 · source /}));
    fireEvent.click(screen.getByRole("button", {name: /分支 · spacemouse-branch /}));
    expect((screen.getByRole("slider", {name: "位移增益"}) as HTMLInputElement).value).toBe("0.6");
    fireEvent.click(screen.getByRole("button", {name: /原始 · source /}));
    const next = screen.getByRole("button", {name: "SpaceMouse 接管"});
    await waitFor(() => expect((next as HTMLButtonElement).disabled).toBe(false));
    fireEvent.click(next);
    await waitFor(() => {
      const calls = vi.mocked(api).mock.calls.filter(([path]) => path === "/api/sessions/source/branches");
      expect(calls).toHaveLength(2);
      expect(JSON.parse(String(calls[1][1]?.body))).toMatchObject({translation_gain: .6, rotation_gain: .7});
    });
    expect((screen.getByRole("slider", {name: "旋转增益"}) as HTMLInputElement).value).toBe("0.7");
  });
});
