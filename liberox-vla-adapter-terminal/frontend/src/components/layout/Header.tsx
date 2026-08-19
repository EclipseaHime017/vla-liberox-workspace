export function Header({ title, buildId }: { title: string; buildId: string }) {
  return <div className="app-header"><div><strong>{title}</strong><span>Franka · LIBERO-X</span></div><div className="privacy-pill" title={"当前后端提供的前端构建：" + buildId}><span />127.0.0.1 · UI {buildId}</div></div>;
}
