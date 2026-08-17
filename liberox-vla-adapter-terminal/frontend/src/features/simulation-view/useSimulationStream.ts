export function useSimulationStream(sessionId: string | null): string | null {
  return sessionId ? `/api/sessions/${sessionId}/stream.mjpeg` : null;
}
