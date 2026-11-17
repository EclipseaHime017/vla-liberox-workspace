#!/usr/bin/env python3
"""Run local parallel simulations using SSH-forwarded remote checkpoint inference."""
import argparse
import os
from pathlib import Path
import sys

for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ[name] = "1"
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True, help="Local server_evaluation.yaml")
    parser.add_argument("--endpoint", action="append", required=True, help="Loopback SSH tunnel URL; repeat for replicas")
    parser.add_argument("--token-file", type=Path, required=True)
    parser.add_argument("--policy-sha256", required=True, help="Expected remote policy.yaml SHA-256")
    parser.add_argument("--concurrency", type=int, default=1)
    args = parser.parse_args()
    from vla_rynn_iql.server_evaluation import load_evaluation
    from vla_rynn_iql.remote_evaluation import run_remote
    return run_remote(load_evaluation(args.config), args.endpoint, args.token_file, args.policy_sha256, args.concurrency)


if __name__ == "__main__":
    raise SystemExit(main())
