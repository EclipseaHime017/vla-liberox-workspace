import { useEffect, useState } from "react";
import { getBootstrap, listRuns } from "../features/run-control/api";
import { RunTable } from "../features/dataset/RunTable";
import type { Session, TaskInfo } from "../features/run-control/types";
import { TaskFilter } from "../features/run-config/TaskSelector";
import { ALL_TASK_SCOPE, filterByTaskScope, type TaskScope } from "../features/run-config/taskHierarchy";

export function RunsPage() {
  const [runs, setRuns] = useState<Session[]>([]);
  const [tasks, setTasks] = useState<TaskInfo[]>([]);
  const [scope, setScope] = useState<TaskScope>(ALL_TASK_SCOPE);
  const [error, setError] = useState("");
  useEffect(() => {
    void getBootstrap().then((value) => setTasks(value.task_catalog)).catch((reason) => setError(String(reason)));
    const load = () => void listRuns().then(setRuns).catch((reason) => setError(String(reason)));
    load(); const timer = window.setInterval(load, 2000); return () => clearInterval(timer);
  }, []);
  return <section className="content-page"><div className="page-heading"><p className="eyebrow">RUN LIBRARY</p><h1>运行记录</h1><p>集中查看原始仿真与干预分支，所有成功结果沿用 LIBERO 原生判定。</p></div>{error && <div className="error-banner">{error}</div>}<div className="surface"><div className="dataset-toolbar"><TaskFilter tasks={tasks} value={scope} onChange={setScope} labelPrefix="运行记录" /></div><RunTable runs={filterByTaskScope(runs, tasks, scope)} /></div></section>;
}
