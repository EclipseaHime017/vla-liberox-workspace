export type PageId = "collect" | "runs" | "dataset" | "settings";

const entries: Array<[PageId, string, string]> = [["collect", "采集", "●"], ["runs", "运行记录", "▤"], ["dataset", "数据集", "◫"], ["settings", "设置", "⚙"]];

export function Sidebar({ page, onPage }: { page: PageId; onPage: (page: PageId) => void }) {
  return <aside className="app-sidebar"><div className="brand-mark"><span>LX</span><strong>LIBERO Studio</strong></div><nav>{entries.map(([id, label, icon]) => <button key={id} className={page === id ? "active" : ""} onClick={() => onPage(id)}><i>{icon}</i>{label}</button>)}</nav><div className="local-label"><span />本机工作区</div></aside>;
}
