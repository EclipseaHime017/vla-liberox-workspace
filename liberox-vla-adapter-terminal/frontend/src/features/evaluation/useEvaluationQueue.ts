import { useEffect, useRef, useState } from "react";
import { enqueueEvaluation, getEvaluationQueue, getOfflineJob, stopEvaluation } from "../run-control/api";
import type { EvaluationConfig, EvaluationQueueState, OfflineJob } from "../run-control/types";

const active = new Set(["STARTING", "RUNNING", "STOPPING"]);
const terminal = new Set(["COMPLETED", "FAILED", "CANCELED"]);

export function useEvaluationQueue(onChanged: () => void, onError: (message: string) => void) {
  const [queue, setQueue] = useState<EvaluationQueueState>({ jobs: [], waiting_reason: null });
  const [job, setJob] = useState<OfflineJob | null>(null);
  const [following, setFollowing] = useState(true);
  const [stoppingId, setStoppingId] = useState<string | null>(null);
  const jobRef = useRef(job);
  jobRef.current = job;
  const followRef = useRef(true);
  const request = useRef(0);
  const mutation = useRef(0);
  const alive = useRef(true);
  const callbacks = useRef({ onChanged, onError });
  callbacks.current = { onChanged, onError };
  const statuses = useRef(new Map<string, string>());

  const recordStatuses = (jobs: EvaluationQueueState["jobs"]) => {
    let changed = false;
    for (const item of jobs) {
      const previous = statuses.current.get(item.id);
      if (previous && previous !== item.status && terminal.has(item.status)) changed = true;
      statuses.current.set(item.id, item.status);
    }
    if (changed) callbacks.current.onChanged();
  };

  useEffect(() => {
    let current = true;
    let timer: number;
    alive.current = true;
    const refresh = async () => {
      const generation = mutation.current;
      try {
        const next = await getEvaluationQueue();
        if (!current || generation !== mutation.current) return;
        setQueue(next);
        recordStatuses(next.jobs);
        const selected = jobRef.current;
        const latest = next.jobs.find((item) => item.id === selected?.id);
        if (selected && latest && latest.status !== selected.status) setJob({ ...selected, ...latest });
        const running = next.jobs.find((item) => active.has(item.status));
        const target = running ?? (!selected ? next.jobs.find((item) => item.status === "QUEUED") : null);
        if (followRef.current && target && target.id !== selected?.id) {
          const selection = request.current;
          const detail = await getOfflineJob(target.id);
          if (current && followRef.current && selection === request.current && generation === mutation.current) setJob(detail);
        }
      } catch (reason) { if (current) callbacks.current.onError(`测试队列读取失败：${String(reason)}`); }
      finally { if (current) timer = window.setTimeout(() => void refresh(), 3000); }
    };
    void refresh();
    return () => { current = false; alive.current = false; window.clearTimeout(timer); };
  }, []);

  const register = async (config: EvaluationConfig, hash: string) => {
    mutation.current += 1;
    const next = await enqueueEvaluation(config, hash);
    mutation.current += 1;
    if (!alive.current) return;
    statuses.current.set(next.id, next.status);
    setQueue((value) => ({ ...value, jobs: [...value.jobs.filter((item) => item.id !== next.id), next] }));
    if (followRef.current && (!jobRef.current || !active.has(jobRef.current.status))) setJob(next);
    callbacks.current.onChanged();
  };
  const update = (next: OfflineJob) => {
    if (!alive.current || jobRef.current?.id !== next.id) return;
    if (jobRef.current.status !== next.status) mutation.current += 1;
    setJob(next);
    setQueue((value) => ({ ...value, jobs: value.jobs.map((item) => item.id === next.id ? next : item) }));
    recordStatuses([next]);
  };
  const inspect = async (id: string) => {
    followRef.current = false; setFollowing(false);
    const selection = ++request.current;
    try {
      const next = await getOfflineJob(id);
      if (alive.current && selection === request.current) setJob(next);
    } catch (reason) { if (alive.current) callbacks.current.onError(String(reason)); }
  };
  const stop = async (id: string) => {
    mutation.current += 1; setStoppingId(id);
    try {
      const next = await stopEvaluation(id);
      mutation.current += 1;
      if (!alive.current) return;
      setQueue((value) => ({ ...value, jobs: value.jobs.map((item) => item.id === id ? next : item) }));
      if (jobRef.current?.id === id) setJob(next);
      recordStatuses([next]);
    } catch (reason) { if (alive.current) callbacks.current.onError(String(reason)); }
    finally { if (alive.current) setStoppingId(null); }
  };
  const follow = () => { request.current += 1; followRef.current = true; setFollowing(true); };
  const dismiss = () => { request.current += 1; followRef.current = false; setFollowing(false); setJob(null); };
  return { queue, job, following, stoppingId, register, update, inspect, stop, follow, dismiss };
}
