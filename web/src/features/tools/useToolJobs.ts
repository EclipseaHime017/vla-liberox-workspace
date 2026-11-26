import { useEffect, useRef, useState } from "react";
import type { OfflineJob } from "../run-control/types";

export const pendingStatuses = new Set(["QUEUED", "STARTING", "RUNNING", "STOPPING"]);
export const statusLabels: Record<OfflineJob["status"], string> = {
  QUEUED: "排队中", STARTING: "启动中", RUNNING: "运行中", STOPPING: "停止中",
  COMPLETED: "已完成", FAILED: "失败", CANCELED: "已取消",
};
export const validInteger = (value: number, min: number, max = Number.MAX_SAFE_INTEGER) =>
  Number.isInteger(value) && value >= min && value <= max;

/** Never let an older poll restore a job the user just stopped. */
export const mergeJob = <Job extends OfflineJob>(current: Job | undefined, next: Job): Job => {
  if (current?.id !== next.id) return next;
  if ((!pendingStatuses.has(current.status) && pendingStatuses.has(next.status))
    || (current.status === "STOPPING" && ["QUEUED", "STARTING", "RUNNING"].includes(next.status))) return current;
  return { ...current, ...next };
};

export type JobEndpoints<Job extends OfflineJob> = {
  list: () => Promise<{ jobs: Job[] }>;
  get: (id: string) => Promise<Job>;
  stop: (id: string) => Promise<Job>;
  remove: (id: string) => Promise<unknown>;
};

/** Poll one tool's queue jobs, follow the selected one and stop or delete them.
 *  ``endpoints`` must be a stable (module-level) object. */
export function useToolJobs<Job extends OfflineJob>(endpoints: JobEndpoints<Job>, onError: (message: string) => void) {
  const [jobs, setJobs] = useState<Job[]>([]);
  const [selectedId, setSelectedId] = useState("");
  const [selected, setSelected] = useState<Job | null>(null);
  const [actionId, setActionId] = useState("");
  const selection = useRef("");
  const deleted = useRef(new Set<string>());
  const report = useRef(onError);
  report.current = onError;
  selection.current = selectedId;

  useEffect(() => {
    let current = true;
    let timer: number;
    const refresh = async () => {
      try {
        const { jobs: values } = await endpoints.list();
        if (!current) return;
        const available = values.filter((job) => !deleted.current.has(job.id));
        setJobs((previous) => {
          const byId = new Map(previous.map((job) => [job.id, job]));
          return [...available.map((job) => mergeJob(byId.get(job.id), job)),
            ...previous.filter((job) => pendingStatuses.has(job.status) && !deleted.current.has(job.id)
              && !available.some((value) => value.id === job.id))];
        });
        if (!selection.current && available.length) {
          const next = available.find((job) => pendingStatuses.has(job.status)) ?? available[0];
          setSelectedId(next.id); setSelected(next);
        }
      } catch (reason) { if (current) report.current(String(reason)); }
      finally { if (current) timer = window.setTimeout(() => void refresh(), 3000); }
    };
    void refresh();
    return () => { current = false; window.clearTimeout(timer); };
  }, [endpoints]);

  useEffect(() => {
    if (!selectedId) return;
    let current = true;
    let timer: number;
    const refresh = async () => {
      try {
        const value = await endpoints.get(selectedId);
        if (current && !deleted.current.has(value.id)) setSelected((previous) => mergeJob(previous ?? undefined, value));
      } catch (reason) { if (current) report.current(String(reason)); }
      finally { if (current) timer = window.setTimeout(() => void refresh(), 3000); }
    };
    void refresh();
    return () => { current = false; window.clearTimeout(timer); };
  }, [endpoints, selectedId]);

  const select = (job: Job) => { setSelectedId(job.id); setSelected(job); };
  const add = (job: Job) => { setJobs((values) => [job, ...values.filter((item) => item.id !== job.id)]); select(job); };
  const update = (job: OfflineJob) => {
    setJobs((values) => values.map((item) => item.id === job.id ? mergeJob(item, job as Job) : item));
    if (selection.current === job.id) setSelected((value) => value?.id === job.id ? mergeJob(value, job as Job) : value);
  };
  /** Stop a pending job, or delete a finished one after ``confirmation`` is accepted. */
  const act = async (job: Job, confirmation?: string) => {
    if (confirmation && !window.confirm(confirmation)) return;
    setActionId(job.id); onError("");
    try {
      if (confirmation) {
        await endpoints.remove(job.id);
        deleted.current.add(job.id);
        setJobs((values) => values.filter((item) => item.id !== job.id));
        if (selection.current === job.id) { setSelectedId(""); setSelected(null); }
      } else update(await endpoints.stop(job.id));
    } catch (reason) { onError(String(reason)); }
    finally { setActionId(""); }
  };
  return { jobs, selectedId, selected, actionId, select, add, update, act };
}
