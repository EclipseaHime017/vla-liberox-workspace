import { useState } from "react";
import { Dialog } from "../../components/ui/Dialog";
import { repairDatasetStorage } from "../run-control/api";

type RepairResult = Awaited<ReturnType<typeof repairDatasetStorage>>;
const layoutLabels: Record<RepairResult["layout"], string> = {
  dated: "旧日期目录", mixed: "新旧目录并存", undated: "无日期目录", empty: "暂无数据",
};

export function StorageRepair({ disabled, onBusyChange, onRepaired }: {
  disabled: boolean;
  onBusyChange: (busy: boolean) => void;
  onRepaired: () => Promise<void>;
}) {
  const [open, setOpen] = useState(false);
  const [running, setRunning] = useState(false);
  const [result, setResult] = useState<RepairResult | null>(null);
  const [error, setError] = useState("");

  const repair = async () => {
    if (running) return;
    setRunning(true); setError(""); onBusyChange(true);
    try {
      const next = await repairDatasetStorage();
      setResult(next);
      if (next.moved > 0) await onRepaired();
    } catch (reason) { setError(String(reason)); }
    finally { setRunning(false); onBusyChange(false); }
  };

  return <div className="storage-repair">
    <button disabled={disabled || running} onClick={() => {
      setResult(null); setError(""); setOpen(true);
    }}>一键修复存储目录</button>
    {open && <Dialog title="修复存储目录" actions={<div className="storage-repair-actions">
      <button disabled={running} onClick={() => setOpen(false)}>{result ? "关闭" : "取消"}</button>
      {!result && <button className="primary" disabled={running} onClick={() => void repair()}>
        {running ? "正在检测并修复…" : "开始检测并修复"}
      </button>}
    </div>}>
      <div className="storage-repair-content">
        {!result && !running && <>
          <p>开始后自动检测当前存储布局：旧记录移至任务目录，遗留空日期目录会被清理；已是新布局则无需修改。</p>
          <p>数据、关键帧和评价保持不变，不重新评价。含其他文件的日期目录会保留。请先停止仿真、取消草稿并结束后台任务。</p>
        </>}
        {running && <p role="status">正在检测当前存储布局并按需修复，请等待完成。</p>}
        {result && <div role="status">
          <p>{result.message}</p>
          <p>检测布局：{layoutLabels[result.layout]}<br />
            已迁移 {result.moved} 条 · 已是新布局 {result.already_current} 条<br />
            已清理 {result.removed_date_dirs} 个空日期目录</p>
          {result.retained_date_dirs.length > 0 && <details>
            <summary>保留 {result.retained_date_dirs.length} 个非空日期目录</summary>
            <ul>{result.retained_date_dirs.map((path) => <li key={path}>{path}</li>)}</ul>
          </details>}
          {result.skipped.length > 0 && <details>
            <summary>{result.skipped.length} 项非标准记录未处理</summary>
            <ul>{result.skipped.map((path) => <li key={path}>{path}</li>)}</ul>
          </details>}
        </div>}
        {error && <p role="alert">{error}</p>}
      </div>
    </Dialog>}
  </div>;
}
