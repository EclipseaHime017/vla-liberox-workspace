export function PolicySelector({ checkpoint }: { checkpoint: string }) {
  return <label className="readonly-field"><span>策略模型</span><strong>{checkpoint}</strong><small>当前后端暂不支持切换</small></label>;
}
