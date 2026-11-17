import type { DatasetIntegrityReport } from "../run-control/types";
import { Badge } from "../../components/ui/Badge";
import { rewardSourceLabels } from "./rewardVersions";

const scopeLabels = { global: "全局", dataset: "数据集专属", effective: "实际可用" };

export function DatasetIntegrityResult({ report }: { report: DatasetIntegrityReport }) {
  return <div className="dataset-integrity-result" aria-label="数据集完整性检查结果">
    <p><strong>{report.dataset_name}</strong> · {report.member_count} 条轨迹 · 原始文件
      <Badge tone={report.integrity_status === "HEALTHY" ? "green" : "red"}>
        {report.integrity_status === "HEALTHY" ? "完整" : "异常"}</Badge></p>
    {report.integrity_error && <p className="error-banner" role="alert">{report.integrity_error}</p>}
    <div className="dataset-integrity-table"><table>
      <thead><tr><th>评价 / 奖励</th>{Object.values(scopeLabels).map((label) => <th key={label}>{label}</th>)}</tr></thead>
      <tbody>{report.sources.map((item) => <tr key={item.source}>
        <th scope="row">{item.source === "keyframes" ? "Stage 关键帧" : rewardSourceLabels[item.source]}</th>
        {(["global", "dataset", "effective"] as const).map((scope) => <td key={scope}>
          <Badge tone={item[scope].ready === item[scope].total ? "green" : scope === "dataset" && !item[scope].invalid ? "neutral" : "red"}>
            {item[scope].ready} / {item[scope].total} 可用</Badge>
          {item[scope].invalid > 0 && <span> · {item[scope].invalid} 无效</span>}
          {scope === "effective" && item.inherited > 0 && <span> · 复用全局 {item.inherited}</span>}
        </td>)}
      </tr>)}</tbody>
    </table></div>
    <p className="field-hint">缺少某类型评价时不能使用该奖励；不影响其他奖励或 BC 训练。Robometer 仅用于诊断。</p>
    {report.sources.filter((item) => item.issues.length).map((item) => <details key={item.source}>
      <summary>{item.source === "keyframes" ? "Stage 关键帧" : rewardSourceLabels[item.source]} · 缺失或异常明细</summary>
      <ul>{item.issues.map((issue, index) => <li key={index}>
        <code>{issue.run_id}</code> · {scopeLabels[issue.scope]}：{issue.error}
      </li>)}</ul>
    </details>)}
  </div>;
}
