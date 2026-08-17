import { useState } from "react";
import { Header } from "../components/layout/Header";
import { PageLayout } from "../components/layout/PageLayout";
import { Sidebar, type PageId } from "../components/layout/Sidebar";
import CollectPage from "../pages/CollectPage";
import { DatasetPage } from "../pages/DatasetPage";
import { RunsPage } from "../pages/RunsPage";
import { SettingsPage } from "../pages/SettingsPage";

const titles: Record<PageId, string> = { collect: "仿真与接管", runs: "运行记录", dataset: "数据管理", settings: "系统设置" };

export default function App() {
  const [page, setPage] = useState<PageId>("collect");
  return <div className="app-shell"><Sidebar page={page} onPage={setPage} /><div className="app-body"><Header title={titles[page]} /><PageLayout><div hidden={page !== "collect"}><CollectPage /></div>{page === "runs" && <RunsPage />}{page === "dataset" && <DatasetPage />}{page === "settings" && <SettingsPage />}</PageLayout></div></div>;
}
