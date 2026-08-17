export function Info({ label, value, note }: { label: string; value: string; note?: string }) {
  return <div className="info-card"><span>{label}</span><strong>{value}</strong>{note && <small>{note}</small>}</div>;
}

export function Metric({ label, value }: { label: string; value: string }) {
  return <div className="metric"><span>{label}</span><strong>{value}</strong></div>;
}

export function Gain({ label, value, setValue }: { label: string; value: number; setValue: (value: number) => void }) {
  return <label className="gain"><span>{label}</span><strong>{value.toFixed(2)}</strong><input type="range" min={0.05} max={1} step={0.01} value={value} onChange={(event) => setValue(Number(event.target.value))} /></label>;
}
