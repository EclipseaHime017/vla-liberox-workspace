export type AppRoute =
  | "collect"
  | "runs"
  | "dataset"
  | "training"
  | "evaluation"
  | "tools"
  | "settings";
export const DEFAULT_ROUTE: AppRoute = "collect";
