import { useEffect, useState } from "react";
import { getBootstrap } from "../features/run-control/api";
import type { Bootstrap } from "../features/run-control/types";

export function SettingsPage() {
  const [data, setData] = useState<Bootstrap | null>(null);
  useEffect(() => { void getBootstrap().then(setData); }, []);
  return <section className="content-page"><div className="page-heading"><p className="eyebrow">SETTINGS</p><h1>设置</h1><p>第一版保留明确的只读边界，模型接口已解耦但暂不在运行中切换。</p></div>{data && <div className="surface settings-list"><div><span>Checkpoint</span><strong>{data.model.checkpoint}</strong></div><div><span>计算设备</span><strong>{data.model.gpu}</strong></div><div><span>控制频率</span><strong>{data.config.control_hz} Hz</strong></div><div><span>预览</span><strong>{data.config.preview.width}×{data.config.preview.height} · {data.config.preview.fps} fps</strong></div><div><span>任务切换</span><strong>{data.capabilities.task_switching ? "已启用" : "未启用"}</strong></div><div><span>模型切换</span><strong>{data.capabilities.model_switching ? "已启用" : "后端接口已预留"}</strong></div></div>}</section>;
}
