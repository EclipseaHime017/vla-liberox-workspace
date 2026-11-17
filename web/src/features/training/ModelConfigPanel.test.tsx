import { cleanup, fireEvent, render, screen } from "@testing-library/react";
import { afterEach, expect, it, vi } from "vitest";
import { ModelConfigPanel, modelConfigurationError } from "./ModelConfigPanel";
import type { TrainingDefaults } from "../run-control/types";

afterEach(cleanup);

const models: NonNullable<TrainingDefaults["models"]> = [{
  id: "fixture-family", label: "Fixture", backbone_modes: ["frozen", "full"],
  training_ui: {
    fields: [
      { name: "model_backbone", label: "适配范围", type: "select", default: "frozen", options: [
        { value: "frozen", label: "固定编码器" }, { value: "full", label: "完整模型" },
      ] },
      { name: "model_gain", label: "测试增益", type: "number", default: 2, min: 0, step: 1,
        visible_when: { model_backbone: "full" } },
    ],
    hints: { frozen: "由模型提供的训练说明。" },
  },
}];

it("renders a registered family using only its declared capabilities", () => {
  const onChange = vi.fn();
  render(<ModelConfigPanel models={models}
    parameters={{ model_family: "fixture-family", model_backbone: "frozen" }} onChange={onChange} />);
  fireEvent.click(screen.getByText("模型训练配置"));
  expect(screen.getByText("由模型提供的训练说明。")).toBeTruthy();
  expect(screen.queryByLabelText("Action head")).toBeNull();
  expect(screen.queryByLabelText("测试增益")).toBeNull();
  fireEvent.change(screen.getByLabelText("适配范围"), { target: { value: "full" } });
  expect(onChange).toHaveBeenCalledWith("model_backbone", "full");
  expect(modelConfigurationError({ model_family: "fixture-family", model_backbone: "frozen" }, models)).toBeNull();
  expect(modelConfigurationError({ model_family: "fixture-family", model_backbone: "lora" }, models)).not.toBeNull();
});

it("supports declared numeric fields without a model-name branch", () => {
  const onChange = vi.fn();
  render(<ModelConfigPanel models={models}
    parameters={{ model_family: "fixture-family", model_backbone: "full", model_gain: 4 }} onChange={onChange} />);
  fireEvent.change(screen.getByLabelText("测试增益"), { target: { value: "6" } });
  expect(onChange).toHaveBeenCalledWith("model_gain", 6);
});

it("rejects unknown families without falling back to another model", () => {
  expect(modelConfigurationError({ model_family: "missing" }, models)).not.toBeNull();
});

it("uses the model-provided all-frozen rule", () => {
  const scoped = [{
    ...models[0], training_ui: {
      ...models[0].training_ui!, trainable_any: ["model_backbone", "model_head"],
      fields: [...models[0].training_ui!.fields,
        { name: "model_head", label: "Head", type: "select" as const, default: "train",
          options: [{ value: "train", label: "训练" }, { value: "frozen", label: "冻结" }] }],
    },
  }];
  expect(modelConfigurationError({
    model_family: "fixture-family", model_backbone: "frozen", model_head: "frozen",
  }, scoped)).not.toBeNull();
  expect(modelConfigurationError({ model_family: "fixture-family", model_backbone: "frozen" }, scoped)).toBeNull();
});
