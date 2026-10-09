export type AppRoute =
  | "collect"
  | "runs"
  | "dataset"
  | "annotation-lab"
  | "training"
  | "evaluation"
  | "settings";
export const DEFAULT_ROUTE: AppRoute = "collect";
