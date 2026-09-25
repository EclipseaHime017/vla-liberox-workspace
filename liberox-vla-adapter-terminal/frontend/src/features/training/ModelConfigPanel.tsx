import type { TrainingDefaults } from "../run-control/types";

type Props = {
  models: TrainingDefaults["models"];
  parameters: Record<string, number | string | boolean | null>;
  onChange: (name: string, value: number | string | boolean | null) => void;
};

export function modelConfigurationError(parameters: Props["parameters"]): string | null {
  return (parameters.model_backbone ?? "frozen") === "frozen"
    && parameters.model_action_head === "frozen" && parameters.model_proprio_projector === "frozen"
    ? "至少保留一个可训练组件。" : null;
}

export function ModelConfigPanel({ models, parameters, onChange }: Props) {
  const family = String(parameters.model_family ?? "vla_adapter");
  const backbone = String(parameters.model_backbone ?? "frozen");
  const error = modelConfigurationError(parameters);
  return     <details className="model-training-config"><summary>模型训练配置</summary>
      <div className="parameter-grid advanced-parameters">
        <label>Backbone 适配方式<select value={backbone} onChange={(event) => onChange("model_backbone", event.target.value)}>
          {(models?.find((model) => model.id === family)?.backbone_modes ?? ["frozen", "lora", "full"]).map((mode) =>
            <option key={mode} value={mode}>{{ frozen: "冻结", lora: "LoRA", full: "全量微调" }[mode] ?? mode}</option>)}
        </select></label>
        {([['model_action_head', 'Action head'], ['model_proprio_projector', 'Proprio projector']] as const).map(([key, label]) =>
          <label key={key}>{label}<select value={String(parameters[key] ?? "train")} onChange={(event) => onChange(key, event.target.value)}>
            <option value="train">训练</option><option value="frozen">冻结</option>
          </select></label>)}
        {backbone === "lora" && <>
          <label>LoRA rank<input type="number" min={1} step={1} value={Number(parameters.model_lora_rank ?? 32)} onChange={(event) => onChange("model_lora_rank", Number(event.target.value))} /></label>
          <label>LoRA alpha<input type="number" min={1} step={1} value={Number(parameters.model_lora_alpha ?? 64)} onChange={(event) => onChange("model_lora_alpha", Number(event.target.value))} /></label>
          <label>LoRA dropout<input type="number" min={0} max={0.99} step={0.01} value={Number(parameters.model_lora_dropout ?? 0)} onChange={(event) => onChange("model_lora_dropout", Number(event.target.value))} /></label>
        </>}
      </div>
      <p className="model-training-hint">{backbone === "frozen"
        ? "保持 Backbone 不变，按所选范围训练动作组件。"
        : backbone === "lora"
          ? "使用 VLA-Adapter 的 all-linear LoRA，并训练 action queries；基础权重保持冻结。"
          : "更新完整 Backbone（视觉、语言、多模态投影与 action queries），显存和 checkpoint 开销显著增加。"}</p>
      {backbone !== "frozen" && <p className="model-training-hint">导出包含完整适配后的 Backbone；在仿真中切换这类模型需重新加载，不复用旧 Backbone。</p>}
      {error && <p role="alert" className="error-banner">{error}</p>}
    </details>;
}
