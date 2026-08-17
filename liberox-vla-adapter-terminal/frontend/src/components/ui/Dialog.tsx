import type { PropsWithChildren, ReactNode } from "react";

export function Dialog({ title, children, actions }: PropsWithChildren<{ title: string; actions?: ReactNode }>) {
  return <div className="modal-backdrop" role="dialog" aria-modal="true"><div className="delete-dialog"><h2>{title}</h2>{children}{actions}</div></div>;
}
