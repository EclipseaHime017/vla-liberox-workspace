"""Server and HPC commands: ``python -m cluster <command>``."""
from core.cli import dispatch

COMMANDS = {
    "train": ("cluster.commands.train_server", "YAML-driven single-node multi-GPU training"),
    "queue": ("cluster.commands.train_queue", "Persistent FIFO training queue"),
    "pbs": ("cluster.commands.train_pbs", "Prepare or submit a PBS job with parallel experiments"),
    "evaluate": ("cluster.commands.evaluate_server", "Closed-loop evaluation of exported checkpoints"),
    "evaluate-remote": ("cluster.commands.evaluate_remote", "Local simulation against remote inference"),
    "serve": ("cluster.commands.serve_evaluation", "Serve one checkpoint for remote evaluation"),
    "train-worker": ("cluster.commands.train_distributed", "torchrun worker (started by `train`)"),
    "transfer-source": ("cluster.transfer.source", "Publish the allowlisted completed-model catalog"),
    "transfer-scan": ("cluster.transfer.scan", "Scan a pinned source for new models"),
    "transfer-dispatch": ("cluster.transfer.dispatch", "Download, verify, publish and evaluate within a window"),
    "transfer-pbs": ("cluster.transfer.pbs", "Publish a frozen PBS terminal catalog"),
    "transfer-seed": ("cluster.transfer.seed", "Seed a receiver payload from local files"),
}

if __name__ == "__main__":
    raise SystemExit(dispatch("cluster", COMMANDS))
