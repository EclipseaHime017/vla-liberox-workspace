"""One held, cross-process lease for expensive platform work."""
from __future__ import annotations

import fcntl
from pathlib import Path

from ..core.exceptions import ConflictError


class ResourceLease:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.stream = path.open("a+")
        try:
            fcntl.flock(self.stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            self.stream.close()
            raise ConflictError("工作资源正忙，请等待当前任务完成；人工接管不会排队。",
                                code="WORK_RESOURCE_BUSY", context={"severity": "warning"}) from exc

    def close(self) -> None:
        # Closing the parent's copy after pass_fds must NOT unlock the child's
        # shared open-file-description. The last holder releases the lease.
        self.stream.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
