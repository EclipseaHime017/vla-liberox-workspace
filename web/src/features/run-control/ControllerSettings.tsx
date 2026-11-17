import { Gain } from "../metrics/MetricsPanel";
import { CONTROLLER_LABELS, controllerStatusText } from "./controller";
import type { ControllerId, ControllerStatus, ControlFrame } from "./types";

const INTENT_LABELS = { idle: "待输入", translation: "平移", rotation: "旋转", combined: "六轴联动" };

type Props = {
  controllerId: ControllerId;
  controller: ControllerStatus | null;
  active: boolean;
  manualActive: boolean;
  busy: boolean;
  translationGain: number;
  rotationGain: number;
  controlFrame: ControlFrame;
  onControlFrame: (frame: ControlFrame) => void;
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
  const pendingFrame = controller?.pending_control_frame ?? (
    manualActive && controller?.state === "ARMED" && controller.control_frame !== props.controlFrame
      ? props.controlFrame : null
  );
  return <section className="panel manual-panel" aria-label="人工控制器设置">
    <div className="panel-title">
      <h2>{manualActive ? "人工接管 · " : "控制器设置 · "}{CONTROLLER_LABELS[controllerId]}</h2>
      <span>{controllerStatusText(controller)}</span>
    </div>
    <div className={`controller-toolbar${factr ? " controller-toolbar-single" : ""}`}>
      <label className="controller-field">人工控制器
        <select value={controllerId} disabled={locked || gravity} onChange={(event) => props.onSelect(event.target.value as ControllerId)}>
          <option value="spacemouse">SpaceMouse</option>
          <option value="factr">FACTR Franka</option>
        </select>
      </label>
      {!factr && <div className="controller-gains">
        <Gain label="位移增益" value={props.translationGain} setValue={props.onTranslationGain} disabled={active && !manualActive} />
        <Gain label="旋转增益" value={props.rotationGain} setValue={props.onRotationGain} disabled={active && !manualActive} />
      </div>}
      {!factr && <label className="controller-field">控制坐标
        <select value={props.controlFrame}
          disabled={busy || calibrating || (active && !manualActive)}
          onChange={(event) => props.onControlFrame(event.target.value as ControlFrame)}>
          <option value="world">世界坐标</option>
          <option value="tool">工具坐标</option>
        </select>
      </label>}
    </div>
    {!factr && manualActive && pendingFrame && <p className="controller-notice" role="status">
      当前{controller?.control_frame === "tool" ? "工具坐标" : "世界坐标"}；请松开摇杆回中，等待切换为{pendingFrame === "tool" ? "工具坐标" : "世界坐标"}。
    </p>}
    <div className="controller-footer">
      {!factr && controller?.motion_mode && <p className="controller-intent">
        {controller.motion_mode === "exclusive" ? "平移／旋转互斥" : "六轴联动"}
        {" · 当前："}{INTENT_LABELS[controller.motion_intent ?? "idle"]}
      </p>}
      <div className="calibration-row" aria-live="polite">
        <button className="primary" disabled={locked || gravity || (!controller?.connected && !retry)} onClick={() => props.onCalibrate()}>
          {retry ? "重新连接并校准" : controller?.calibrated ? "重新校准控制器" : "校准控制器"}
        </button>
        {calibrating && <progress aria-label="校准进度" max={1} value={controller?.calibration_progress ?? 0} />}
        {calibrating && controller?.message && <span>{controller.message}</span>}
      </div>
      {factr && <div className="calibration-row" aria-live="polite">
        <button className={gravity ? "danger" : "primary"}
          disabled={busy || calibrating || (!gravity && (!controller?.calibrated || !controller?.connected || controller?.stale))}
          onClick={() => props.onGravity(!gravity)}>
          {gravity ? "关闭重力补偿" : "开启重力补偿"}
        </button>
        <span>{controller?.gravity_state === "unknown" ? "补偿状态未确认" : gravity ? "补偿已开启" : "补偿已关闭"}</span>
      </div>}
    </div>
    {(factr || controller?.state === "ERROR") && controller?.error && <p className="hint controller-error" role="alert">{controller.error}</p>}
  </section>;
}
