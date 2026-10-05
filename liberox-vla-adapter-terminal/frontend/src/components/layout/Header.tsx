import type { Ref } from "react";

export function Header({ title, buildId, statusRef, showStatus = false }: {
  title: string; buildId: string; statusRef?: Ref<HTMLDivElement>; showStatus?: boolean;
}) {
  return <header className="app-header">
    <div><strong>{title}</strong><span>Franka · LIBERO-X</span></div>
    <div className="app-header-actions">
      <div ref={statusRef} hidden={!showStatus} className="app-header-status" />
      <div className="privacy-pill" title={"当前后端提供的前端构建：" + buildId}><span />127.0.0.1 · UI {buildId}</div>
    </div>
  </header>;
}
