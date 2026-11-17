"""Account-wide exclusion, with opt-in disjoint GPU leases for PBS experiments."""
from __future__ import annotations

import fcntl
from contextlib import contextmanager
import json
import os
from pathlib import Path
import re
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
        record = _record(state, token)
        if not record or not record.get("accepting", True):
            raise TrainingBusy("Training lease changed before worker startup; refusing GPU work")
        record.setdefault("workers", []).append(process_identity(os.getpid()))


def _record(state, token):
    if not token:
        return None
    return state if state.get("token") == token else state.get("scopes", {}).get(token)


def _active(record):
    return any(process_alive(item) for item in [record.get("owner"), *record.get("workers", [])])


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
    def __init__(self, devices=None):
        if devices is not None:
            devices = tuple(sorted(devices))
            if (not devices or len(set(devices)) != len(devices) or any(
                    not isinstance(item, str) or not re.fullmatch(
                        r"GPU-[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}", item)
                    for item in devices)):
                raise ValueError("Shared training leases require distinct full physical GPU UUIDs")
        self.devices = devices
        self.token = None
        self._gpu_fds = []
        root = LEASE_ROOT
        root.mkdir(mode=0o700, exist_ok=True)
        if root.is_symlink() or root.stat().st_uid != os.getuid():
            raise ValueError("Unsafe server training lease directory")
        self.fd = os.open(root / "training.lock", os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
        try:
            fcntl.flock(self.fd, (fcntl.LOCK_SH if devices else fcntl.LOCK_EX) | fcntl.LOCK_NB)
            for device in devices or ():
                descriptor = os.open(root / f"{device}.lock", os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
                self._gpu_fds.append(descriptor)
                fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
            with _members() as state:
                records = [state, *state.get("scopes", {}).values()]
                for record in records:
                    overlap = devices is None or record.get("devices") is None or set(devices) & set(record["devices"])
                    if overlap and _active(record):
                        raise TrainingBusy("Previous server worker is still active on the requested resources")
                self.token = uuid.uuid4().hex
                record = dict(token=self.token, owner=process_identity(os.getpid()), workers=[],
                              accepting=True, devices=devices)
                if devices is None:
                    state.clear()
                    state.update(record)
                else:
                    scopes = {token: value for token, value in state.get("scopes", {}).items() if _active(value)}
                    scopes[self.token] = record
                    state["scopes"] = scopes
        except BlockingIOError:
            self.close()
            raise TrainingBusy("Another server training holds the requested resources") from None
        except BaseException:
            self.close()
            raise

    def workers(self):
        with _members(write=False) as state:
            record = _record(state, self.token) or {}
            return [item for item in record.get("workers", []) if process_alive(item)]

    def retire(self):
        with _members() as state:
            record = _record(state, self.token)
            if record is not None:
                record["accepting"] = False

    @property
    def fds(self):
        return (self.fd, *self._gpu_fds)

    def close(self):
        if self.fd is not None:
            try:
                if self.token is not None:
                    with _members() as state:
                        record = _record(state, self.token)
                        if record is not None:
                            record["owner"] = None
                            record["accepting"] = False
            finally:
                for descriptor in self._gpu_fds:
                    os.close(descriptor)
                self._gpu_fds.clear()
                os.close(self.fd)
                self.fd = None

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
