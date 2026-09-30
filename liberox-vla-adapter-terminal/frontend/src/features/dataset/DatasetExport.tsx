import { useEffect, useRef, useState } from "react";
import { getDatasetExport, startDatasetExport } from "../run-control/api";
import type { DatasetExportStatus } from "../run-control/types";

const active = (value: DatasetExportStatus | null) => value && ["QUEUED", "RUNNING"].includes(value.status);

export function useDatasetExport(datasetId: string, enabled = true) {
  const [state, setState] = useState<DatasetExportStatus | null>(null);
  const [starting, setStarting] = useState(false);
  const [error, setError] = useState("");
  const generation = useRef(0);
  useEffect(() => {
    const token = ++generation.current;
    let timer: number | undefined;
    setState(null); setError(""); setStarting(false);
    if (!enabled) return;
    const load = () => {
      if (generation.current !== token) return;
      void getDatasetExport(datasetId).then((next) => {
        if (generation.current === token) { setState(next); setError(""); }
      }).catch((err) => {
        if (generation.current === token) { setError(String(err)); timer = window.setTimeout(load, 3000); }
      });
    };
    load();
    return () => { generation.current++; window.clearTimeout(timer); };
  }, [datasetId, enabled]);
  useEffect(() => {
    if (!enabled || !active(state)) return;
    let current = true;
    let timer: number;
    const poll = () => {
      void getDatasetExport(datasetId).then((next) => {
        if (current) { setState(next); setError(""); }
      }).catch((err) => {
        if (current) { setError(String(err)); timer = window.setTimeout(poll, 3000); }
      });
    };
    timer = window.setTimeout(poll, 1500);
    return () => { current = false; window.clearTimeout(timer); };
  }, [datasetId, state, enabled]);
  const start = async () => {
    const token = ++generation.current;
    setStarting(true); setError("");
    try {
      const next = await startDatasetExport(datasetId);
      if (generation.current === token) setState(next);
    } catch (err) { if (generation.current === token) setError(String(err)); }
    finally { if (generation.current === token) setStarting(false); }
  };
  return { state, starting, error, start };
}

type ExportController = ReturnType<typeof useDatasetExport>;

export function DatasetExportButton({ controller, disabled }: { controller: ExportController; disabled: boolean }) {
  const { state, starting, start } = controller;
  return <button disabled={disabled || starting || !!active(state)} onClick={() => void start()}>
      {starting || active(state) ? "正在导出…" : "导出数据集"}
    </button>;
}

export function DatasetExportResult({ controller }: { controller: ExportController }) {
  const { state, error } = controller;
  const pathInput = useRef<HTMLInputElement>(null);
  const [copied, setCopied] = useState<"" | "copied" | "manual">("");
  const copyGeneration = useRef(0);
  useEffect(() => {
    copyGeneration.current++;
    setCopied("");
    return () => { copyGeneration.current++; };
  }, [state?.id, state?.output_path]);
  const copyPath = async () => {
    if (!state?.output_path) return;
    const token = copyGeneration.current;
    try {
      if (!navigator.clipboard?.writeText) throw new Error("Clipboard unavailable");
      await navigator.clipboard.writeText(state.output_path);
      if (copyGeneration.current === token) setCopied("copied");
    } catch {
      if (copyGeneration.current !== token) return;
      pathInput.current?.focus();
      pathInput.current?.select();
      setCopied("manual");
    }
  };
  const message = error || state?.error;
  if (!active(state) && state?.status !== "COMPLETED" && !message) return null;
  return <div className="dataset-export-feedback">
    {active(state) && <div className="dataset-export-result" role="status">
      <span className="dataset-export-indicator" aria-hidden="true" />
      <strong>{state?.stage}</strong><span className="dataset-export-count">{state?.completed_runs}/{state?.total_runs} 条</span>
      {state?.current_file && <span className="dataset-export-file" title={state.current_file}>{state.current_file}</span>}
    </div>}
    {state?.status === "COMPLETED" && <div className="dataset-export-result is-complete" role="status">
      <span className="dataset-export-indicator" aria-hidden="true">✓</span>
      <strong>导出完成</strong>
      {state.output_path ? <><input ref={pathInput} className="dataset-export-path" aria-label="导出目录" readOnly
        title={state.output_path ?? ""} value={state.output_path ?? ""} onFocus={(event) => event.currentTarget.select()} />
      <button className="dataset-export-copy" onClick={() => void copyPath()}>
        {copied === "copied" ? "已复制" : "复制路径"}
      </button></> : <span className="dataset-export-file">导出目录不可用</span>}
      {copied === "manual" && <span className="dataset-export-copy-hint">路径已选中，请手动复制</span>}
    </div>}
    {message && <div className="dataset-export-error" role="alert">{message}</div>}
  </div>;
}
