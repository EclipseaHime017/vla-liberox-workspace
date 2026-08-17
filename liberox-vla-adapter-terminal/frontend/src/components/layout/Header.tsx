export function Header({ title }: { title: string }) {
  return <div className="app-header"><div><strong>{title}</strong><span>Franka · LIBERO-X</span></div><div className="privacy-pill"><span />127.0.0.1 · Local only</div></div>;
}
