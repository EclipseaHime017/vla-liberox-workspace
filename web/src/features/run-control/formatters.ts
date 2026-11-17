export function formatComputeDevice(value: string): string {
  const cudaDevice = value.match(/^cuda:\d+\s*\((.+)\)$/i);
  return cudaDevice?.[1].trim() || value;
}

type OutcomeRun = { status?: string; success: boolean; success_confirmed?: boolean };

/** A run succeeds only with a confirmed done streak; a lone done is reported as unconfirmed. */
export function runOutcome(run: OutcomeRun): { label: string; tone: "red" | "green" | "neutral"; confirmed: boolean } {
  const confirmed = run.success_confirmed ?? run.success;
  if (run.status === "ERROR") return { label: "错误", tone: "red", confirmed };
  if (confirmed) return { label: "成功", tone: "green", confirmed };
  return { label: run.success ? "未确认" : "失败", tone: "neutral", confirmed };
}
