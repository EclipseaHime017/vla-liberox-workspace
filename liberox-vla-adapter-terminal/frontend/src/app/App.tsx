import { useEffect, useState } from "react";
import { api } from "../api/client";
import { Header } from "../components/layout/Header";
import { PageLayout } from "../components/layout/PageLayout";
import { Sidebar, type PageId } from "../components/layout/Sidebar";
import CollectPage from "../pages/CollectPage";
import { DatasetPage } from "../pages/DatasetPage";
import { RunsPage } from "../pages/RunsPage";
import { SettingsPage } from "../pages/SettingsPage";
import { TrainingPage } from "../pages/TrainingPage";
import { EvaluationPage } from "../pages/EvaluationPage";
import { ModelsPage } from "../pages/ModelsPage";
import { AnnotationLabPage } from "../pages/AnnotationLabPage";

const titles: Record<PageId, string> = { collect: "LIBERO-X仿真与干预控制台", runs: "运行记录", dataset: "数据管理", "annotation-lab": "自动关键帧实验", models: "模型管理", training: "离线训练", evaluation: "策略测试", settings: "系统设置" };

export default function App() {
  const [page, setPage] = useState<PageId>("collect");
  const [buildId, setBuildId] = useState("checking");
  const [statusTarget, setStatusTarget] = useState<HTMLDivElement | null>(null);
  useEffect(() => {
    api<{ dist_fingerprint: string | null }>("/api/build-info")
      .then((info) => setBuildId(info.dist_fingerprint?.slice(0, 12) ?? "missing"))
      .catch(() => setBuildId("legacy-backend"));
  }, []);
  return <div className="app-shell"><Sidebar page={page} onPage={setPage} /><div className="app-body">
    <Header title={titles[page]} buildId={buildId} statusRef={setStatusTarget} showStatus={page === "collect"} />
    <PageLayout wide={page === "collect"}>
      <div hidden={page !== "collect"}><CollectPage statusTarget={statusTarget} /></div>
      {page === "runs" && <RunsPage />}{page === "dataset" && <DatasetPage />}{page === "models" && <ModelsPage />}{page === "training" && <TrainingPage />}{page === "evaluation" && <EvaluationPage />}{page === "settings" && <SettingsPage />}
      {page === "annotation-lab" && <AnnotationLabPage />}
    </PageLayout>
  </div></div>;
}
