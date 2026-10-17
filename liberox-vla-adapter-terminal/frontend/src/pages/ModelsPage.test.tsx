import { cleanup, render, screen } from "@testing-library/react";
import { afterEach, expect, it, vi } from "vitest";
import { ModelsPage } from "./ModelsPage";
import { getModel, listModels } from "../features/run-control/api";
import type { PolicyDetail } from "../features/run-control/types";

vi.mock("../features/run-control/api", () => ({
  getModel: vi.fn(), listModels: vi.fn(), copyModel: vi.fn(), deleteModel: vi.fn(), renameModel: vi.fn(),
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
