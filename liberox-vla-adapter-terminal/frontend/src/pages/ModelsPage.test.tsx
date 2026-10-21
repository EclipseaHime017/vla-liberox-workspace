import { act, cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, expect, it, vi } from "vitest";
import { ModelsPage } from "./ModelsPage";
import { getModel, getModelTrainingConfig, listModels } from "../features/run-control/api";
import type { OfflineJob, PolicyDetail } from "../features/run-control/types";

vi.mock("../features/run-control/api", () => ({
  getModelTrainingConfig: vi.fn(), getModel: vi.fn(), listModels: vi.fn(), copyModel: vi.fn(), deleteModel: vi.fn(), renameModel: vi.fn(),
}));
afterEach(() => { cleanup(); vi.clearAllMocks(); });

it.each([
  [{ image_flip_vertical: true, image_flip_horizontal: false }, "On", "Off"],
  [{ image_flip_vertical: false, image_flip_horizontal: true }, "Off", "On"],
  [{ image_flip_vertical: true, image_flip_horizontal: true }, "On", "On"],
  [{ image_flip_vertical: false, image_flip_horizontal: false }, "Off", "Off"],
])("displays model-owned camera flips", async (flips, vertical, horizontal) => {
  const model: PolicyDetail = {
    policy_id: "base", label: "Test model", kind: "base", base_checkpoint: "test", stats_key: "test",
    training_step: null, compatibility_sha256: null, manifest: null, components: [], training_records: [],
    model_config: { family: "vla_adapter", backbone: "frozen" },
    io: { native_action_horizon: 8, replay_horizon: 8 }, observation: flips,
  };
  vi.mocked(listModels).mockResolvedValue([model]);
  vi.mocked(getModel).mockResolvedValue(model);
  render(<ModelsPage />);
  expect((await screen.findByText("上下翻转")).nextElementSibling?.textContent).toBe(vertical);
  expect(screen.getByText("左右镜像").nextElementSibling?.textContent).toBe(horizontal);
});


const detail = (id: string, jobs: OfflineJob[] = []): PolicyDetail => ({
  policy_id: id, label: id, kind: "rynn_iql_overlay", base_checkpoint: "checkpoint", stats_key: "test",
  training_step: 32, compatibility_sha256: null, manifest: null, components: [], training_records: jobs,
});
const job = { id: "train-17", kind: "training", status: "COMPLETED", dataset_id: "ds",
  created_at: "2026-10-20T00:00:00Z", model_relation: "output",
  parameters: { algorithm: "bc", dataset_name: "Takeover dataset", train_steps: 32, seed: 17,
    task_id: "place bowl", model: { base_id: "registered-base" }, reward: null },
} as unknown as OfflineJob;

it("shows readable training history and opens its exact effective YAML", async () => {
  const model = detail("trained", [job]);
  vi.mocked(listModels).mockResolvedValue([model]); vi.mocked(getModel).mockResolvedValue(model);
  vi.mocked(getModelTrainingConfig).mockResolvedValue({filename: "effective_config.yaml", text: "training:\n  seed: 17\n"});
  render(<ModelsPage />);
  await screen.findByText("BC · Takeover dataset");
  expect(screen.getByText("训练步数").nextElementSibling?.textContent).toBe("32");
  expect(screen.getByText("Seed").nextElementSibling?.textContent).toBe("17");
  fireEvent.click(screen.getByRole("button", {name: "查看训练 YAML"}));
  await screen.findByText("effective_config.yaml");
  expect(document.querySelector(".model-training-yaml pre")?.textContent).toBe("training:\n  seed: 17\n");
  expect(getModelTrainingConfig).toHaveBeenCalledWith("trained", "train-17");
  expect(screen.getByRole("button", {name: "下载 YAML"})).toBeTruthy();
});

it("ignores a stale model detail response when another model is selected", async () => {
  const base = detail("initial");
  vi.mocked(listModels).mockResolvedValue([base, detail("A"), detail("B")]);
  let resolveA: (value: PolicyDetail) => void = () => {};
  vi.mocked(getModel).mockImplementation((id) => id === "A"
    ? new Promise((resolve) => {resolveA = resolve;}) : Promise.resolve(detail(id)));
  render(<ModelsPage />);
  await screen.findByRole("heading", {name: "initial"});
  fireEvent.click(screen.getByRole("button", {name: /A A IQL/}));
  fireEvent.click(screen.getByRole("button", {name: /B B IQL/}));
  await screen.findByRole("heading", {name: "B"});
  await act(async () => {resolveA(detail("A"));});
  expect(screen.queryByRole("heading", {name: "A"})).toBeNull();
  expect(screen.getByRole("heading", {name: "B"})).toBeTruthy();
});

it("discards an in-flight YAML result after switching models", async () => {
  vi.mocked(listModels).mockResolvedValue([detail("A", [job]), detail("B")]);
  vi.mocked(getModel).mockImplementation(async (id) => detail(id, id === "A" ? [job] : []));
  let resolveConfig: (value: {filename: string; text: string}) => void = () => {};
  vi.mocked(getModelTrainingConfig).mockImplementation(() => new Promise((resolve) => {resolveConfig = resolve;}));
  render(<ModelsPage />);
  fireEvent.click(await screen.findByRole("button", {name: "查看训练 YAML"}));
  fireEvent.click(screen.getByRole("button", {name: /B B IQL/}));
  await screen.findByRole("heading", {name: "B"});
  await act(async () => {resolveConfig({filename: "effective_config.yaml", text: "stale-A-configuration"});});
  await waitFor(() => expect(screen.queryByText("stale-A-configuration")).toBeNull());
});
