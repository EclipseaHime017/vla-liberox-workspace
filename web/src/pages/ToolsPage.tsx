import { useEffect, useState } from "react";
import { getBootstrap } from "../features/run-control/api";
import type { Bootstrap } from "../features/run-control/types";
import { CriticDiagnosticTool } from "../features/tools/CriticDiagnosticTool";
import { ReturnDistributionTool } from "../features/tools/ReturnDistributionTool";
import "../styles/tools.css";

const toolTabs = [["critic", "IQL Critic 诊断"], ["returns", "VF 回报分布"]] as const;

export function ToolsPage() {
  const [bootstrap, setBootstrap] = useState<Bootstrap | null>(null);
  const [tool, setTool] = useState<(typeof toolTabs)[number][0]>("critic");
  const [error, setError] = useState("");

  useEffect(() => {
    let current = true;
    void getBootstrap().then((value) => { if (current) setBootstrap(value); })
      .catch((reason) => { if (current) setError(String(reason)); });
    return () => { current = false; };
  }, []);

  return <section className="content-page tools-page">
    <nav className="surface tools-toolbar" aria-label="测试工具">{toolTabs.map(([id, label]) =>
      <button key={id} className={tool === id ? "active" : ""} aria-current={tool === id ? "page" : undefined}
        onClick={() => setTool(id)}>{label}</button>)}</nav>
    {error && <div className="error-banner" role="alert"><span>{error}</span><button onClick={() => setError("")}>关闭</button></div>}
    {tool === "critic" ? <CriticDiagnosticTool bootstrap={bootstrap} /> : <ReturnDistributionTool bootstrap={bootstrap} />}
  </section>;
}
