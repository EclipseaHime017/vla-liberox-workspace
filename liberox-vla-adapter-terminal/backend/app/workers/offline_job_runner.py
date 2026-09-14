#!/usr/bin/env python3
"""Detached, restart-safe runner for UI-created offline RL jobs.

The FastAPI process only creates a validated job specification. This runner owns
the cross-process GPU lock and writes all durable state without importing UI or
MuJoCo code.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import signal
import subprocess
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class Runner:
    def __init__(self, job_dir: Path):
        self.job_dir = job_dir.resolve()
        self.job_path = self.job_dir / "job.json"
        self.log_path = self.job_dir / "job.log"
        self.lock = threading.RLock()
        self.stop = threading.Event()
        self.child: subprocess.Popen[str] | None = None
        self.payload = json.loads(self.job_path.read_text(encoding="utf-8"))

    def persist(self, **changes: Any) -> None:
        with self.lock:
            self.payload.update(changes)
            temporary = self.job_path.with_name(".job.json.tmp")
            with temporary.open("w", encoding="utf-8") as stream:
                json.dump(self.payload, stream, ensure_ascii=False, indent=2)
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, self.job_path)

    def log(self, level: str, message: str) -> None:
        line = json.dumps(
            {"time": utc_now(), "level": level, "message": message},
            ensure_ascii=False,
        )
        with self.log_path.open("a", encoding="utf-8") as stream:
            stream.write(line + "\n")
            stream.flush()

    def signal(self, signum: int, _frame: Any) -> None:
        self.stop.set()
        self.log("WARN", f"收到停止信号 {signum}，正在结束当前阶段")
        child = self.child
        if child is not None and child.poll() is None:
            try:
                child.terminate()
            except ProcessLookupError:
                pass

    def heartbeat(self) -> None:
        while not self.stop.wait(1.0):
            self.persist(heartbeat_at=utc_now())

    def run(self) -> int:
        signal.signal(signal.SIGTERM, self.signal)
        signal.signal(signal.SIGINT, self.signal)
        lock_stream = None
        if self.payload.get("requires_gpu", True):
            gpu_lock_path = Path(self.payload["gpu_lock_path"])
            gpu_lock_path.parent.mkdir(parents=True, exist_ok=True)
            lock_stream = gpu_lock_path.open("a+")
            try:
                fcntl.flock(lock_stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                lock_stream.close()
                self.persist(status="FAILED", completed_at=utc_now(),
                             error="Another GPU task already holds the platform lock")
                return 2
        self.persist(
            status="RUNNING", pid=os.getpid(), process_group_id=os.getpgrp(),
            started_at=utc_now(), heartbeat_at=utc_now(), error=None,
        )
        heartbeat = threading.Thread(target=self.heartbeat, daemon=True)
        heartbeat.start()
        self.log("INFO", f"任务开始 · {self.payload['kind']} · PID {os.getpid()}")
        try:
            for stage in self.payload["stages"]:
                if self.stop.is_set():
                    break
                self.persist(stage=stage["id"], stage_label=stage["label"])
                self.log(
                    "INFO",
                    f"{stage['label']} · conda 环境 {stage['environment']}",
                )
                command = [
                    "conda", "run", "--no-capture-output", "-n",
                    stage["environment"], *stage["argv"],
                ]
                with self.log_path.open("a", encoding="utf-8") as output:
                    output.write(json.dumps({
                        "time": utc_now(), "level": "COMMAND",
                        "message": " ".join(command),
                    }, ensure_ascii=False) + "\n")
                    output.flush()
                    self.child = subprocess.Popen(
                        command,
                        cwd=stage.get("cwd") or None,
                        stdout=output,
                        stderr=subprocess.STDOUT,
                        text=True,
                    )
                    return_code = self.child.wait()
                self.child = None
                if self.stop.is_set():
                    break
                if return_code != 0:
                    raise RuntimeError(
                        f"Stage {stage['label']} exited with code {return_code}"
                    )
                self.log("DONE", f"{stage['label']}完成")
            if self.stop.is_set():
                self.persist(status="CANCELED", completed_at=utc_now(), stage="canceled")
                self.log("WARN", "任务已取消；已完成的缓存和 checkpoint 保留")
                return 130
            self.persist(status="COMPLETED", completed_at=utc_now(), stage="completed")
            self.log("DONE", "任务完成")
            return 0
        except BaseException as exc:
            self.persist(
                status="FAILED", completed_at=utc_now(), stage="failed",
                error=f"{type(exc).__name__}: {exc}",
            )
            self.log("ERROR", f"{type(exc).__name__}: {exc}")
            return 1
        finally:
            self.stop.set()
            self.persist(heartbeat_at=utc_now())
            if lock_stream is not None:
                fcntl.flock(lock_stream.fileno(), fcntl.LOCK_UN)
                lock_stream.close()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--job-dir", type=Path, required=True)
    args = parser.parse_args()
    return Runner(args.job_dir).run()


if __name__ == "__main__":
    raise SystemExit(main())
