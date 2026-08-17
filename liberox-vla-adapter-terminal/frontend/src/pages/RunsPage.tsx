import { useEffect, useState } from "react";
import { listRuns } from "../features/run-control/api";
import { RunTable } from "../features/dataset/RunTable";
import type { Session } from "../features/run-control/types";

export function RunsPage() {
  const [runs, setRuns] = useState<Session[]>([]);
  useEffect(() => { const load = () => void listRuns().then(setRuns); load(); const timer = window.setInterval(load, 2000); return () => clearInterval(timer); }, []);
  return <section className="content-page"><div className="page-heading"><p className="eyebrow">RUN LIBRARY</p><h1>运行记录</h1><p>集中查看原始仿真与干预分支，所有成功结果沿用 LIBERO 原生判定。</p></div><div className="surface"><RunTable runs={runs} /></div></section>;
}
