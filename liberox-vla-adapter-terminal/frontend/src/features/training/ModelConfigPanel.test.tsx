import { cleanup, fireEvent, render, screen } from "@testing-library/react";
import { afterEach, expect, it, vi } from "vitest";
import { ModelConfigPanel, modelConfigurationError } from "./ModelConfigPanel";

afterEach(cleanup);

it("offers expert and full π₀.₅ training without VLA-specific toggles", () => {
  const onChange = vi.fn();
  render(<ModelConfigPanel models={[{ id: "pi05", label: "π₀.₅ · LIBERO", backbone_modes: ["frozen", "full"] }]}
    parameters={{ model_family: "pi05", model_backbone: "frozen" }} onChange={onChange} />);
  fireEvent.click(screen.getByText("模型训练配置"));
  expect(screen.queryByLabelText("Action head")).toBeNull();
  expect(screen.queryByLabelText("Proprio projector")).toBeNull();
  expect(screen.queryByText("LoRA")).toBeNull();
  fireEvent.change(screen.getByLabelText("微调范围"), { target: { value: "full" } });
  expect(onChange).toHaveBeenCalledWith("model_backbone", "full");
  expect(modelConfigurationError({ model_family: "pi05", model_backbone: "frozen" })).toBeNull();
  expect(modelConfigurationError({ model_family: "pi05", model_backbone: "lora" })).not.toBeNull();
});
