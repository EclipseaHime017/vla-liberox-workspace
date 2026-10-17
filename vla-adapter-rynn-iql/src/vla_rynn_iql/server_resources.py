"""One account-wide training lease, retained by the launched supervisor."""
from __future__ import annotations

import fcntl
from contextlib import contextmanager
import json
import os
from pathlib import Path
import uuid

from .io import atomic_json

LEASE_ROOT = Path("/tmp") / f"liberox-server-{os.getuid()}"


def process_identity(pid: int) -> dict | None:
    try:
        fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
        if fields[0] == "Z":
            return None
        return {"pid": pid, "start": fields[19],
                "boot": Path("/proc/sys/kernel/random/boot_id").read_text().strip()}
    except (FileNotFoundError, ProcessLookupError):
        return None


def process_alive(identity: dict | None) -> bool:
    return bool(identity and process_identity(identity["pid"]) == identity)


class TrainingBusy(RuntimeError):
    pass


class TrainingCleanupError(RuntimeError):
    """Training may still be active; the queue must not finalize or advance."""


@contextmanager
def _members(*, write=True):
    with (LEASE_ROOT / "members.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        path = LEASE_ROOT / "members.json"
        state = json.loads(path.read_text()) if path.exists() else {}
        yield state
        if write:
            atomic_json(path, state)


def register_worker(token: str):
    """Join before any CUDA work; stale launchers cannot join a newer lease."""
    with _members() as state:
        if not token or state.get("token") != token or not state.get("accepting", True):
            raise TrainingBusy("Training lease changed before worker startup; refusing GPU work")
        state.setdefault("workers", []).append(process_identity(os.getpid()))


def signal_worker(identity, command):
    """Use a pidfd so a recycled PID can never receive a training signal."""
    import ctypes
    if not process_alive(identity):
        return
    # Some Conda Python builds omit os.pidfd_open despite running on Linux.
    libc = ctypes.CDLL(None, use_errno=True)
    descriptor = libc.pidfd_open(identity["pid"], 0)
    if descriptor < 0:
        error = ctypes.get_errno()
        if error == 3:  # ESRCH: worker already exited.
            return
        raise OSError(error, os.strerror(error))
    try:
        if process_alive(identity) and libc.pidfd_send_signal(descriptor, command, None, 0) < 0:
            error = ctypes.get_errno()
            if error != 3:
                raise OSError(error, os.strerror(error))
    finally:
        os.close(descriptor)


class TrainingLease:
    def __init__(self):
        self.token = None
        root = LEASE_ROOT
        root.mkdir(mode=0o700, exist_ok=True)
        if root.is_symlink() or root.stat().st_uid != os.getuid():
            raise ValueError("Unsafe server training lease directory")
        self.fd = os.open(root / "training.lock", os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
        try:
            fcntl.flock(self.fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            with _members() as state:
                if any(process_alive(item) for item in [state.get("owner"), *state.get("workers", [])]):
                    raise TrainingBusy("Previous server worker is still active; waiting prevents overlapping GPU jobs")
                self.token = uuid.uuid4().hex
                state.clear()
                state.update(token=self.token, owner=process_identity(os.getpid()), workers=[], accepting=True)
        except BlockingIOError:
            self.close()
            raise TrainingBusy("Another server training is active for this local account") from None
        except BaseException:
            self.close()
            raise

    def workers(self):
        with _members(write=False) as state:
            if state.get("token") != self.token:
                return []
            return [item for item in state.get("workers", []) if process_alive(item)]

    def retire(self):
        with _members() as state:
            if state.get("token") == self.token:
                state["accepting"] = False

    def close(self):
        if self.fd is not None:
            try:
                if self.token is not None:
                    with _members() as state:
                        if state.get("token") == self.token:
                            state["owner"] = None
                            state["accepting"] = False
            finally:
                os.close(self.fd)
                self.fd = None

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
