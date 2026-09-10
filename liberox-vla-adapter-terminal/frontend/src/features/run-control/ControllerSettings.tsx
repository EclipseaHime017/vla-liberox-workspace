import { Gain } from "../metrics/MetricsPanel";
import { CONTROLLER_LABELS, controllerStatusText } from "./controller";
import type { ControllerId, ControllerStatus } from "./types";

type Props = {
  controllerId: ControllerId;
  controller: ControllerStatus | null;
  active: boolean;
  manualActive: boolean;
  busy: boolean;
  translationGain: number;
  rotationGain: number;
  onSelect: (id: ControllerId) => void;
  onCalibrate: () => void;
  onGravity: (enabled: boolean) => void;
  onTranslationGain: (gain: number) => void;
  onRotationGain: (gain: number) => void;
};

export function ControllerSettings(props: Props) {
  const { controllerId, controller, active, manualActive, busy } = props;
  const factr = controllerId === "factr";
  const calibrating = controller?.state === "CALIBRATING";
  const gravity = Boolean(controller?.gravity_enabled);
  const retry = controller?.state === "ERROR";
  const locked = active || busy || calibrating || controller?.state === "ALIGNING";
  return <section className="panel manual-panel" aria-label="人工控制器设置">
    <div className="panel-title">
      <h2>{manualActive ? "人工接管 · " : "控制器设置 · "}{CONTROLLER_LABELS[controllerId]}</h2>
      <span>{controllerStatusText(controller)}</span>
    </div>
    <div className="controller-selection">
      <label>人工控制器
        <select value={controllerId} disabled={locked || gravity} onChange={(event) => props.onSelect(event.target.value as ControllerId)}>
          <option value="spacemouse">SpaceMouse</option>
          <option value="factr">FACTR Franka</option>
        </select>
      </label>
    </div>
    {!factr && <div className="gain-grid">
      <Gain label="位移增益" value={props.translationGain} setValue={props.onTranslationGain} disabled={active && !manualActive} />
      <Gain label="旋转增益" value={props.rotationGain} setValue={props.onRotationGain} disabled={active && !manualActive} />
    </div>}
    <div className="calibration-row" aria-live="polite">
      <button className="primary" disabled={locked || gravity || (!controller?.connected && !retry)} onClick={() => props.onCalibrate()}>
        {retry ? "重新连接并校准" : controller?.calibrated ? "重新校准控制器" : "校准控制器"}
      </button>
      {calibrating && <progress aria-label="校准进度" max={1} value={controller?.calibration_progress ?? 0} />}
    </div>
    {factr && <div className="calibration-row" aria-live="polite">
      <button className={gravity ? "danger" : "primary"}
        disabled={busy || calibrating || (!gravity && (!controller?.calibrated || !controller?.connected || controller?.stale))}
        onClick={() => props.onGravity(!gravity)}>
        {gravity ? "关闭重力补偿" : "开启重力补偿"}
      </button>
      <span>{controller?.gravity_state === "unknown" ? "补偿状态未确认" : gravity ? "补偿已开启" : "补偿已关闭"}</span>
    </div>}
    {controller?.state === "ERROR" && controller.error && <p className="hint controller-error" role="alert">{controller.error}</p>}
  </section>;
}
