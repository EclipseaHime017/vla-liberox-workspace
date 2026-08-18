import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  selectMainVideoArtifact,
  stepToVideoTime,
  videoTimeToStep,
} from "../features/simulation-view/controls";
import { api } from "../api/client";
import { sessionWebSocket } from "../api/websocket";
import { ACTIVE, TERMINAL, type Bootstrap, type ControllerStatus, type Draft, type FrameState, type PolicyBranchDraft, type Session, type TaskInfo } from "../features/run-control/types";
import { Gain, Info, Metric } from "../features/metrics/MetricsPanel";
import { RunConfigForm } from "../features/run-config/RunConfigForm";
import { RunStatus } from "../features/run-control/RunStatus";
import { SessionMonitor } from "../features/run-control/SessionMonitor";
import { formatComputeDevice } from "../features/run-control/formatters";
import { ALL_TASKS, filterSessionsByTask } from "../features/run-control/sessionFilters";
import { useSimulationStream } from "../features/simulation-view/useSimulationStream";

function fixed(values: number[] | null, digits = 4): string {
  return values ? values.map((value) => value.toFixed(digits)).join(", ") : "—";
}

function CollectPage() {
  const [bootstrap, setBootstrap] = useState<Bootstrap | null>(null);
  const [draft, setDraft] = useState<Draft | null>(null);
  const [policyBranchDraft, setPolicyBranchDraft] = useState<PolicyBranchDraft | null>(null);
  const [sessions, setSessions] = useState<Session[]>([]);
  const [selectedId, setSelectedId] = useState("");
  const [selectedStep, setSelectedStep] = useState(0);
  const [frameState, setFrameState] = useState<FrameState | null>(null);
  const [maxSteps, setMaxSteps] = useState(300);
  const [openLoop, setOpenLoop] = useState(8);
  const [taskId, setTaskId] = useState("");
  const [sessionTaskFilter, setSessionTaskFilter] = useState(ALL_TASKS);
  const [translationGain, setTranslationGain] = useState(0.25);
  const [rotationGain, setRotationGain] = useState(0.08);
  const [controller, setController] = useState<ControllerStatus | null>(null);
  const [deleteTarget, setDeleteTarget] = useState<Session | null>(null);
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);
  const websocket = useRef<WebSocket | null>(null);
  const trajectoryVideo = useRef<HTMLVideoElement | null>(null);
  const gainRef = useRef({ translationGain, rotationGain });

  const selected = useMemo(
    () => sessions.find((session) => session.id === selectedId) ?? null,
    [sessions, selectedId],
  );
  const visibleSessions = useMemo(
    () => filterSessionsByTask(sessions, sessionTaskFilter),
    [sessions, sessionTaskFilter],
  );
  const active = sessions.find((session) => ACTIVE.has(session.status)) ?? null;
  const manualSessionActive = Boolean(
    selected?.control_mode === "manual" && ACTIVE.has(selected.status) && !selected.legacy,
  );
  const liveStreamUrl = useSimulationStream(selected && ACTIVE.has(selected.status) ? selected.id : null);
  const timelineReady = Boolean(selected && TERMINAL.has(selected.status) && selected.state_count);
  const mainVideoName = useMemo(
    () => selected && timelineReady ? selectMainVideoArtifact(selected.artifacts) : null,
    [selected, timelineReady],
  );

  const syncVideoToStep = useCallback((step: number) => {
    if (!selected?.action_count || !bootstrap?.config.video_fps || !trajectoryVideo.current) return;
    const target = stepToVideoTime(
      step,
      selected.action_count,
      bootstrap.config.video_fps,
    );
    if (Math.abs(trajectoryVideo.current.currentTime - target)
        > 0.5 / bootstrap.config.video_fps) {
      trajectoryVideo.current.currentTime = target;
    }
  }, [bootstrap?.config.video_fps, selected?.action_count]);

  useEffect(() => {
    api<{ discarded: boolean }>("/api/draft", { method: "DELETE" })
      .catch(() => ({ discarded: false }))
      .then(() => Promise.all([
        api<Bootstrap>("/api/bootstrap"),
        api<Session[]>("/api/sessions"),
        api<ControllerStatus>("/api/controller"),
      ]))
      .then(([boot, history, controllerStatus]) => {
        setBootstrap(boot);
        setSessions(history);
        setMaxSteps(boot.config.max_steps);
        setOpenLoop(boot.config.open_loop_steps);
        setTaskId(boot.task.task_id);
        setTranslationGain(boot.config.manual.translation_gain);
        setRotationGain(boot.config.manual.rotation_gain);
        setController(controllerStatus);
        if (history[0]) setSelectedId(history[0].id);
      })
      .catch((reason) => setError(String(reason)));
  }, []);

  useEffect(() => {
    if (!draft || draft.preview_status === "READY" || draft.preview_status === "ERROR") return;
    const timer = window.setInterval(() => {
      api<Draft | null>("/api/draft")
        .then((value) => {
          setDraft(value);
          if (value) {
            setTaskId(value.task_id);
            setMaxSteps(value.max_steps);
            setOpenLoop(value.open_loop_steps);
          }
        })
        .catch((reason) => setError(String(reason)));
    }, 250);
    return () => window.clearInterval(timer);
  }, [draft?.id, draft?.preview_status]);

  useEffect(() => {
    if (selected?.control_mode !== "manual") return;
    if (selected.manual_translation_gain !== null) {
      setTranslationGain(selected.manual_translation_gain);
    }
    if (selected.manual_rotation_gain !== null) {
      setRotationGain(selected.manual_rotation_gain);
    }
  }, [selected?.id]);

  useEffect(() => {
    const timer = window.setInterval(() => {
      api<Session[]>("/api/sessions")
        .then((history) => {
          setSessions(history);
          if (!selectedId && history[0]) setSelectedId(history[0].id);
        })
        .catch((reason) => setError(String(reason)));
    }, 1000);
    return () => window.clearInterval(timer);
  }, [selectedId]);

  useEffect(() => {
    const interval = controller?.state === "ARMED"
      ? 100
      : controller?.state === "CALIBRATING" ? 250 : 1000;
    const timer = window.setInterval(() => {
      api<ControllerStatus>("/api/controller")
        .then(setController)
        .catch((reason) => setError(String(reason)));
    }, interval);
    return () => window.clearInterval(timer);
  }, [controller?.state]);

  useEffect(() => {
    if (!selectedId) return;
    websocket.current?.close();
    const socket = sessionWebSocket(selectedId);
    websocket.current = socket;
    socket.onopen = () => {
      if (
        selected?.control_mode === "manual"
        && !selected.legacy
        && ACTIVE.has(selected.status)
      ) {
        const gains = {
          translationGain: selected.manual_translation_gain ?? gainRef.current.translationGain,
          rotationGain: selected.manual_rotation_gain ?? gainRef.current.rotationGain,
        };
        socket.send(JSON.stringify({
          type: "manual_settings",
          translation_gain: gains.translationGain,
          rotation_gain: gains.rotationGain,
        }));
      }
    };
    socket.onmessage = (event) => {
      const message = JSON.parse(event.data);
      if (message.type === "session") {
        const session = message.session as Session;
        setSessions((current) => {
          const existing = current.some((value) => value.id === session.id);
          return existing
            ? current.map((value) => (value.id === session.id ? session : value))
            : [session, ...current];
        });
      } else if (message.type === "error") setError(message.detail);
    };
    socket.onerror = () => setError("会话 WebSocket 连接失败");
    return () => socket.close();
  }, [selectedId]);

  useEffect(() => {
    if (!selected || !TERMINAL.has(selected.status) || selected.state_count < 1) {
      setFrameState(null);
      return;
    }
    const safeStep = Math.min(selectedStep, selected.state_count - 1);
    if (safeStep !== selectedStep) setSelectedStep(safeStep);
    api<FrameState>("/api/sessions/" + selected.id + "/frames/" + safeStep + "/state")
      .then(setFrameState)
      .catch((reason) => setError(String(reason)));
  }, [selected?.id, selected?.status, selected?.state_count, selectedStep]);

  useEffect(() => {
    // During an original rollout or branch, the disabled timeline must follow
    // the newest recorded state. Keeping it fixed at resume_step while its max
    // grows makes the thumb appear to move backwards.
    if (selected && ACTIVE.has(selected.status)) {
      setSelectedStep(selected.current_step);
    }
  }, [selected?.id, selected?.status, selected?.current_step]);

  useEffect(() => {
    gainRef.current = { translationGain, rotationGain };
  }, [translationGain, rotationGain]);

  useEffect(() => {
    if (!manualSessionActive) return;
    const socket = websocket.current;
    if (socket?.readyState !== WebSocket.OPEN) return;
    socket.send(JSON.stringify({
      type: "manual_settings",
      translation_gain: translationGain,
      rotation_gain: rotationGain,
    }));
  }, [manualSessionActive, translationGain, rotationGain]);

  const createDraft = async () => {
    setBusy(true);
    setError("");
    try {
      const value = await api<Draft>("/api/draft", {
        method: "POST",
        body: JSON.stringify({ task_id: taskId, max_steps: maxSteps, open_loop_steps: openLoop }),
      });
      setDraft(value);
    } catch (reason) {
      setError(String(reason));
    } finally { setBusy(false); }
  };

  const updateDraft = async (patch: Partial<Pick<Draft, "task_id" | "max_steps" | "open_loop_steps">>) => {
    if (!draft) return;
    setError("");
    try {
      const value = await api<Draft>("/api/draft", {
        method: "PATCH",
        body: JSON.stringify(patch),
      });
      setDraft(value);
      setTaskId(value.task_id);
      setMaxSteps(value.max_steps);
      setOpenLoop(value.open_loop_steps);
    } catch (reason) { setError(String(reason)); }
  };

  const cancelDraft = async () => {
    setBusy(true);
    setError("");
    try {
      await api<{ discarded: boolean }>("/api/draft", { method: "DELETE" });
      setDraft(null);
    } catch (reason) { setError(String(reason)); }
    finally { setBusy(false); }
  };

  const startDraft = async () => {
    if (!draft) return;
    setBusy(true);
    setError("");
    try {
      const synchronized = await api<Draft>("/api/draft", {
        method: "PATCH",
        body: JSON.stringify({
          task_id: taskId,
          max_steps: maxSteps,
          open_loop_steps: openLoop,
        }),
      });
      setDraft(synchronized);
      if (!synchronized.preview_ready) return;
      const session = await api<Session>("/api/draft/start", { method: "POST" });
      setDraft(null);
      setSessions((current) => [session, ...current]);
      setSessionTaskFilter(session.task_id ?? ALL_TASKS);
      setSelectedId(session.id);
      setSelectedStep(0);
    } catch (reason) { setError(String(reason)); }
    finally { setBusy(false); }
  };

  const stop = async () => {
    if (!active) return;
    setError("");
    try {
      const session = await api<Session>("/api/sessions/" + active.id + "/stop", { method: "POST" });
      setSessions((current) => current.map((value) => value.id === session.id ? session : value));
    } catch (reason) { setError(String(reason)); }
  };

  const configurePolicyBranch = () => {
    if (!selected) return;
    setError("");
    trajectoryVideo.current?.pause();
    setPolicyBranchDraft({
      parent_session_id: selected.id,
      task_prompt: selected.task ?? "该历史轨迹没有任务提示词",
      source_episode: "episode_000 · " + selected.id,
      resume_step: selectedStep,
      end_step: selected.action_count,
      open_loop_steps: selected.open_loop_steps,
    });
  };

  const branch = async (controlMode: "policy" | "manual") => {
    if (!selected) return;
    const parentSessionId = controlMode === "policy" && policyBranchDraft
      ? policyBranchDraft.parent_session_id
      : selected.id;
    setBusy(true);
    setError("");
    try {
      const session = await api<Session>("/api/sessions/" + parentSessionId + "/branches", {
        method: "POST",
        body: JSON.stringify({
          resume_step: controlMode === "policy" && policyBranchDraft
            ? policyBranchDraft.resume_step
            : selectedStep,
          control_mode: controlMode,
          open_loop_steps: controlMode === "policy" && policyBranchDraft
            ? policyBranchDraft.open_loop_steps
            : openLoop,
          ...(controlMode === "manual" ? {
            translation_gain: translationGain,
            rotation_gain: rotationGain,
          } : {}),
        }),
      });
      setSessions((current) => [session, ...current]);
      setPolicyBranchDraft(null);
      setSelectedId(session.id);
      setSelectedStep(session.current_step);
    } catch (reason) {
      setError(String(reason));
    } finally { setBusy(false); }
  };

  const calibrate = async () => {
    setError("");
    try {
      setController(await api<ControllerStatus>("/api/controller/calibrate", {
        method: "POST",
      }));
    } catch (reason) { setError(String(reason)); }
  };

  const removeSession = async () => {
    if (!deleteTarget) return;
    setBusy(true);
    setError("");
    try {
      await api<{ deleted: string }>("/api/sessions/" + deleteTarget.id, {
        method: "DELETE",
        body: JSON.stringify({ confirm_session_id: deleteTarget.id }),
      });
      const remaining = sessions.filter((item) => item.id !== deleteTarget.id);
      setSessions(remaining);
      if (selectedId === deleteTarget.id) {
        setSelectedId(filterSessionsByTask(remaining, sessionTaskFilter)[0]?.id ?? "");
        setSelectedStep(0);
      }
      setDeleteTarget(null);
    } catch (reason) { setError(String(reason)); }
    finally { setBusy(false); }
  };

  const changeSessionTaskFilter = (value: string) => {
    setSessionTaskFilter(value);
    setPolicyBranchDraft(null);
    const next = filterSessionsByTask(sessions, value)[0] ?? null;
    setSelectedId(next?.id ?? "");
    setSelectedStep(0);
  };

  if (!bootstrap) return <div className="loading">正在初始化 LIBERO-X 控制台…</div>;
  const selectedTask: TaskInfo | null = selected ? {
    task_id: selected.task_id ?? "",
    level: selected.level ?? "未知",
    task_name: selected.task_name ?? "未知任务",
    prompt: selected.task ?? "该历史轨迹没有任务提示词",
    init_state_index: 0,
  } : null;
  const displayTask = draft?.task ?? selectedTask ?? bootstrap.task;
  const progress = selected
    ? Math.min(100, (selected.current_step / Math.max(1, selected.max_steps)) * 100)
    : 0;
  const videoArtifactUrl = selected && mainVideoName
    ? "/api/sessions/" + selected.id + "/artifacts/" + encodeURIComponent(mainVideoName)
    : null;
  const livePreview = Boolean(draft || (selected && ACTIVE.has(selected.status)));
  const viewerAspectRatio = livePreview
    ? bootstrap.config.preview.stream_width + " / " + bootstrap.config.preview.stream_height
    : bootstrap.config.preview.width + " / " + bootstrap.config.preview.height;
  const controllerReady = controller?.state === "READY";
  const controllerPillText = controller?.state === "ARMED"
    ? (controller.latency_ms === null ? "控制器 · 等待样本" : `控制器 · ${controller.latency_ms.toFixed(1)} ms`)
    : (controller ? `${controller.state} · ${controller.message}` : "正在检测控制器");

  return (
    <section className="collect-page">
      <header>
        <div>
          <p className="eyebrow">LOCAL ROBOTICS WORKBENCH</p>
          <h1>LIBERO-X 仿真与干预控制台</h1>
          <p className="subtitle">单会话 · 20 Hz 实时控制 · 精确状态回溯</p>
        </div>
        <div className="state-pills">
          <div className={"system-state controller-state " + (controller?.state === "ARMED" ? (controller.latency_level ?? "red") : "")}>
            <span className="pulse" />{controllerPillText}
          </div>
          <div className={"system-state " + (active ? "running" : "")}>
            <span className="pulse" />
            {active ? active.status + " · " + active.id : "IDLE · 可开始"}
          </div>
          {(controller?.state === "UNCALIBRATED" || controller?.state === "ERROR") && <button
            className="calibrate-shortcut"
            disabled={Boolean(active) || !controller.connected}
            onClick={calibrate}
          >校准</button>}
        </div>
      </header>

      {error && <div className="error-banner"><span>{error}</span><button onClick={() => setError("")}>关闭</button></div>}

      <section className="metadata-grid">
        <Info label="Checkpoint" value={bootstrap.model.checkpoint} note="当前后端暂不支持切换" />
        <Info label="任务难度" value={displayTask.level} />
        <Info label="任务提示词" value={displayTask.prompt} />
        <Info label="计算设备" value={formatComputeDevice(bootstrap.model.gpu)} />
        <Info label="控制与预测" value={bootstrap.config.control_hz + " Hz / " + bootstrap.model.action_schema.predicted_chunk_size + " 步预测"} />
      </section>

      <section className="workspace">
        <aside className="panel session-panel">
          <div className="panel-title"><h2>会话</h2><span>{visibleSessions.length}/{sessions.length}</span></div>
          <RunConfigForm draft={draft} branchDraft={policyBranchDraft} tasks={bootstrap.task_catalog} active={Boolean(active)} busy={busy} taskId={taskId} maxSteps={maxSteps} openLoop={openLoop} onCreate={createDraft} onStart={startDraft} onCancel={cancelDraft} onStop={stop} onTask={(value) => { setTaskId(value); void updateDraft({ task_id: value }); }} onMaxSteps={(value, commit) => { setMaxSteps(value); if (commit) void updateDraft({ max_steps: value }); }} onOpenLoop={(value, commit) => { setOpenLoop(value); if (commit) void updateDraft({ open_loop_steps: value }); }} onBranchOpenLoop={(value) => setPolicyBranchDraft((current) => current ? { ...current, open_loop_steps: value } : current)} onStartBranch={() => void branch("policy")} onCancelBranch={() => setPolicyBranchDraft(null)} />
          <div className="session-filter">
            <label>数据检索
              <select value={sessionTaskFilter} disabled={Boolean(active)} onChange={(event) => changeSessionTaskFilter(event.target.value)}>
                <option value={ALL_TASKS}>全部任务数据</option>
                {bootstrap.task_catalog.map((task) => <option key={task.task_id} value={task.task_id}>{task.prompt}</option>)}
              </select>
            </label>
          </div>
          <div className="session-list">
            {visibleSessions.map((session) => (
              <div key={session.id} className={"session-entry " + (selectedId === session.id ? "selected" : "")}>
                <button className="session-item" onClick={() => { setPolicyBranchDraft(null); setSelectedId(session.id); setSelectedStep(0); }}>
                  <RunStatus run={session} />
                </button>
                {session.managed && <button
                  className="delete-icon"
                  title="永久删除此 UI 会话"
                  aria-label={"删除会话 " + session.id}
                  disabled={Boolean(active) || Boolean(policyBranchDraft) || controller?.state === "CALIBRATING" || busy}
                  onClick={() => setDeleteTarget(session)}
                >🗑</button>}
              </div>
            ))}
            {!visibleSessions.length && <p className="session-empty">当前任务暂无可预览数据。</p>}
          </div>
        </aside>

        <section className="main-column">
          <div className={"viewer-monitor-layout " + (!draft && selected ? "has-monitor" : "") }>
            <div className="panel viewer-panel">
            <div className="panel-title"><h2>{draft ? "仿真草稿 · 四视角初始状态" : timelineReady ? "主视角录像 · 帧 " + selectedStep : "四视角实时预览"}</h2>{draft ? <span>{draft.id}</span> : selected && <span>{selected.id}</span>}</div>
            <div className={"viewer " + (livePreview ? "multi-camera" : "")} style={{ aspectRatio: viewerAspectRatio }}>
              {draft ? (
                !draft.preview_available ? <div className="viewer-empty">正在加载初始状态预览…</div> : <img
                  key={draft.preview_revision + "-" + draft.preview_status}
                  src={"/api/draft/preview.jpg?revision=" + draft.preview_revision + "&status=" + draft.preview_status}
                  alt="草稿初始状态四视角"
                />
              ) : selected && ACTIVE.has(selected.status) ? (
                <img src={liveStreamUrl ?? undefined} alt="主视角、腕部、左侧和右侧实时画面" />
              ) : timelineReady && selected && videoArtifactUrl ? (
                <video
                  key={selected.id + "-" + mainVideoName}
                  ref={trajectoryVideo}
                  src={videoArtifactUrl}
                  controls
                  playsInline
                  preload="metadata"
                  onLoadedMetadata={() => syncVideoToStep(selectedStep)}
                  onTimeUpdate={(event) => setSelectedStep(videoTimeToStep(
                    event.currentTarget.currentTime,
                    selected.state_count,
                    bootstrap.config.video_fps,
                  ))}
                  onSeeked={(event) => setSelectedStep(videoTimeToStep(
                    event.currentTarget.currentTime,
                    selected.state_count,
                    bootstrap.config.video_fps,
                  ))}
                />
              ) : timelineReady && selected ? (
                <div className="viewer-empty">该历史轨迹没有可播放的主视角视频</div>
              ) : <div className="viewer-empty">选择会话或开始一次仿真</div>}
              {draft && <div className="viewer-overlay"><span>{draft.preview_status}</span><span>{draft.task.level}</span><span>初始状态 0</span></div>}
              {draft && draft.preview_status !== "READY" && <div className="preparation-overlay"><span>{draft.preview_status === "PREPARING" ? "加载场景" : draft.preview_status === "RENDERING" ? "准备预览" : "预览失败：" + draft.error}</span></div>}
              {!draft && selected && <div className="viewer-overlay"><span>{selected.status}</span><span>{selected.current_step}/{selected.max_steps}</span><span>{selected.simulated_duration_seconds.toFixed(2)} s</span></div>}
              {!draft && selected && ACTIVE.has(selected.status) && selected.preparation_message && <div className="preparation-overlay">
                {selected.countdown_remaining !== null && <strong>{selected.countdown_remaining}</strong>}
                <span>{selected.preparation_message}</span>
              </div>}
            </div>
            {!draft && selected && <>
              <div className="progress-track" key={selected.id}><div style={{ width: progress + "%" }} /></div>
              <div className="metric-row">
                <Metric label="控制步" value={String(selected.current_step)} />
                <Metric label="实测频率" value={selected.measured_control_hz ? selected.measured_control_hz.toFixed(2) + " Hz" : "—"} />
                <Metric label="推理次数" value={String(selected.policy_queries)} />
                <Metric label="成功" value={selected.success ? "是" : "否"} />
              </div>
            </>}
            </div>
            {!draft && <SessionMonitor session={selected} />}
          </div>

          {!draft && selected && <div className="panel timeline-panel">
            <div className="panel-title"><h2>回溯与分支</h2><span>{timelineReady ? "0 — " + (selected.state_count - 1) : "结束后启用"}</span></div>
            <input className="timeline" type="range" min={0} max={Math.max(0, selected.state_count - 1)} value={selectedStep} disabled={!timelineReady || Boolean(policyBranchDraft)} onChange={(event) => {
              const step = Number(event.target.value);
              setSelectedStep(step);
              syncVideoToStep(step);
            }} />
            <div className="frame-grid">
              <Metric label="仿真时间" value={frameState ? frameState.time_seconds.toFixed(3) + " s" : "—"} />
              <Metric label="EEF 位置 [m]" value={fixed(frameState?.eef_position_m ?? null)} />
              <Metric label="EEF 轴角 [rad]" value={fixed(frameState?.eef_axis_angle_rad ?? null)} />
              <Metric label="夹爪 qpos" value={fixed(frameState?.gripper_qpos ?? null)} />
              <Metric label="raw action [-]" value={fixed(frameState?.raw_action ?? null, 3)} />
              <Metric label="环境 action [-]" value={fixed(frameState?.env_action ?? null, 3)} />
            </div>
            <div className="button-row branch-buttons">
              <button className="primary" disabled={!selected.branchable || Boolean(active) || busy || Boolean(policyBranchDraft) || selectedStep >= selected.action_count} onClick={configurePolicyBranch}>{policyBranchDraft ? "正在配置二次推理" : "从此帧重新推理"}</button>
              <button
                disabled={!selected.branchable || Boolean(active) || busy || Boolean(policyBranchDraft) || selectedStep >= selected.action_count || !controllerReady}
                title={controllerReady ? "从所选帧开始 SpaceMouse 接管" : "请先连接并校准 SpaceMouse"}
                onClick={() => branch("manual")}
              >SpaceMouse 接管</button>
            </div>
            {!selected.branchable && <p className="hint">分支结果只读，不能继续创建子分支。</p>}
          </div>}

          {!draft && (manualSessionActive || Boolean(selected?.branchable && timelineReady)) && <div className="panel manual-panel">
            <div className="panel-title">
              <h2>{manualSessionActive ? "人工接管 · SpaceMouse" : "SpaceMouse 控制设置"}</h2>
              <span>{controller?.state ?? "DISCONNECTED"}</span>
            </div>
            <div className="gain-grid">
              <Gain label="位移增益" value={translationGain} setValue={setTranslationGain} />
              <Gain label="旋转增益" value={rotationGain} setValue={setRotationGain} />
            </div>
            {!controller?.calibrated && <div className="calibration-row">
              <button className="primary" disabled={Boolean(active) || controller?.state === "CALIBRATING" || !controller?.connected} onClick={calibrate}>校准控制器</button>
              <span>{controller?.message ?? "正在检测控制器"}</span>
              {controller?.state === "CALIBRATING" && <progress max={1} value={controller.calibration_progress} />}
            </div>}
            <p className="hint">控制器只需在连接后手动校准一次。接管前会显示 3 秒倒计时；左键张开夹爪，右键闭合。灵敏度可在运行中实时调整。</p>
          </div>}

          {!draft && selected && Object.keys(selected.artifacts).length > 0 && <div className="panel artifacts-panel">
            <div className="panel-title"><h2>结果文件</h2><span>{Object.keys(selected.artifacts).length}</span></div>
            <div className="artifact-list">
              {Object.keys(selected.artifacts).sort().map((name) => <a key={name} href={"/api/sessions/" + selected.id + "/artifacts/" + encodeURIComponent(name)} target="_blank" rel="noreferrer">{name}</a>)}
            </div>
          </div>}
        </section>
      </section>
      {deleteTarget && <div className="modal-backdrop" role="dialog" aria-modal="true">
        <div className="delete-dialog">
          <h2>永久删除会话？</h2>
          <p>将永久删除 <strong>{deleteTarget.id}</strong> 的整个 UI 结果目录，此操作无法恢复。已有分支已保存独立源轨迹，不会被级联删除。</p>
          <div className="button-row">
            <button onClick={() => setDeleteTarget(null)}>取消</button>
            <button className="danger" disabled={busy} onClick={removeSession}>确认永久删除</button>
          </div>
        </div>
      </div>}
    </section>
  );
}

export default CollectPage;
