export function formatComputeDevice(value: string): string {
  const cudaDevice = value.match(/^cuda:\d+\s*\((.+)\)$/i);
  return cudaDevice?.[1].trim() || value;
}
