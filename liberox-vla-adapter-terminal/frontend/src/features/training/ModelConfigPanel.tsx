import type { TrainingDefaults } from "../run-control/types";

type Props = {
  models: TrainingDefaults["models"];
  parameters: Record<string, number | string | boolean | null>;
  onChange: (name: string, value: number | string | boolean | null) => void;
};
type Model = NonNullable<Props["models"]>[number];

function selectedModel(parameters: Props["parameters"], models: Props["models"]) {
  return models?.find((model) => model.id === parameters.model_family) ?? (!parameters.model_family ? models?.[0] : undefined);
}

function fieldsFor(model: Model) {
  return model.training_ui?.fields ?? [{
    name: "model_backbone", label: "微调范围", type: "select" as const,
    options: model.backbone_modes.map((mode) => ({ value: mode, label: mode })),
    default: model.backbone_modes[0],
  }];
}

export function modelConfigurationError(parameters: Props["parameters"], models?: Props["models"]): string | null {
  if (!models) return null;
  const model = selectedModel(parameters, models);
  if (!model) return "当前模型没有已注册的训练适配器。";
  const fields = fieldsFor(model);
  const value = (name: string) => parameters[name] ?? fields.find((field) => field.name === name)?.default;
  for (const field of fields) {
    if (field.visible_when && !Object.entries(field.visible_when).every(([key, expected]) => value(key) === expected)) continue;
    if (field.type === "select" && !field.options?.some((option) => option.value === value(field.name))) {
      return `请选择有效的${field.label}。`;
    }
  }
  const trainable = model.training_ui?.trainable_any;
  return trainable?.length && trainable.every((name) => value(name) === "frozen") ? "至少保留一个可训练组件。" : null;
}

export function ModelConfigPanel({ models, parameters, onChange }: Props) {
  const model = selectedModel(parameters, models);
  const fields = model ? fieldsFor(model) : [];
  const value = (name: string) => parameters[name] ?? fields.find((field) => field.name === name)?.default;
  const error = modelConfigurationError(parameters, models);
  const hint = model?.training_ui?.hints?.[String(value("model_backbone"))];
  return <details className="model-training-config"><summary>模型训练配置</summary>
    <div className="parameter-grid advanced-parameters">
      {fields.filter((field) => !field.visible_when || Object.entries(field.visible_when)
        .every(([key, expected]) => value(key) === expected)).map((field) => <label key={field.name}>{field.label}
        {field.type === "select" ? <select value={String(value(field.name) ?? "")} onChange={(event) => onChange(field.name, event.target.value)}>
          {field.options?.map((option) => <option key={option.value} value={option.value}>{option.label}</option>)}
        </select> : <input type="number" min={field.min} max={field.max} step={field.step}
          value={Number(value(field.name) ?? 0)} onChange={(event) => onChange(field.name, Number(event.target.value))} />}
      </label>)}
    </div>
    {hint && <p className="model-training-hint">{hint}</p>}
    {error && <p role="alert" className="error-banner">{error}</p>}
  </details>;
}
