export function selectMainVideoArtifact(
  artifacts: Record<string, string>,
): string | null {
  const names = Object.keys(artifacts);
  const preferred = [
    "agentview.mp4",
    "intervention_agentview_hd.mp4",
  ];
  for (const preferredName of preferred) {
    const match = names.find((name) => name === preferredName || name.endsWith(`/${preferredName}`));
    if (match) return match;
  }
  return names.find((name) =>
    name.endsWith(".mp4")
    && !name.includes("vla_views")
    && !name.endsWith("intervention.mp4"),
  ) ?? null;
}

export function stepToVideoTime(
  step: number,
  actionCount: number,
  videoFps: number,
): number {
  if (actionCount < 1 || videoFps <= 0) return 0;
  // The trajectory has one more state than encoded action frames. Seeking to
  // duration for that final state intentionally leaves the final video frame
  // displayed instead of snapping the timeline back by one step.
  return Math.min(Math.max(0, step), actionCount) / videoFps;
}

export function videoTimeToStep(
  timeSeconds: number,
  stateCount: number,
  videoFps: number,
): number {
  if (stateCount < 1 || videoFps <= 0) return 0;
  return Math.min(
    stateCount - 1,
    Math.max(0, Math.round(timeSeconds * videoFps)),
  );
}

export function canStartDraft(
  previewReady: boolean,
  hasActiveSession: boolean,
  busy: boolean,
): boolean {
  return previewReady && !hasActiveSession && !busy;
}
