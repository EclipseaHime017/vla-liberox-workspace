import type { PropsWithChildren } from "react";
export function PageLayout({ children, wide = false }: PropsWithChildren<{ wide?: boolean }>) {
  return <div className={`page-layout${wide ? " page-layout-wide" : ""}`}>{children}</div>;
}
