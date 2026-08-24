export class ApiError extends Error {
  status: number;
  detail: unknown;
  constructor(status: number, detail: unknown, message: string) {
    super(message);
    this.status = status;
    this.detail = detail;
  }
}

export async function api<T>(url: string, options?: RequestInit): Promise<T> {
  const response = await fetch(url, {
    ...options,
    headers: { "Content-Type": "application/json", ...(options?.headers ?? {}) },
  });
  if (!response.ok) {
    let detail: unknown = response.status + " " + response.statusText;
    try {
      const body = await response.json();
      detail = body.detail ?? detail;
    } catch {
      // Keep the HTTP status when the response has no JSON body.
    }
    const message = typeof detail === "string"
      ? detail
      : typeof detail === "object" && detail && "message" in detail
        ? String((detail as { message: unknown }).message)
        : JSON.stringify(detail);
    throw new ApiError(response.status, detail, message);
  }
  return response.json() as Promise<T>;
}
