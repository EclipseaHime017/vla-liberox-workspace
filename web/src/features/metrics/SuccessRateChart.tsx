export function SuccessRateChart({ rate }: { rate: number }) {
  const percentage = Math.max(0, Math.min(100, rate * 100));
  return <div className="success-chart" aria-label={`成功率 ${percentage.toFixed(1)}%`}><div className="success-ring" style={{ "--rate": `${percentage * 3.6}deg` } as React.CSSProperties}><strong>{percentage.toFixed(1)}%</strong><span>成功率</span></div></div>;
}
