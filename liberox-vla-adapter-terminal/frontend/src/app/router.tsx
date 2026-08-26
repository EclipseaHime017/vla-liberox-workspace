export type AppRoute =
  | "collect"
  | "runs"
  | "dataset"
  | "training"
  | "evaluation"
  | "settings";
export const DEFAULT_ROUTE: AppRoute = "collect";
