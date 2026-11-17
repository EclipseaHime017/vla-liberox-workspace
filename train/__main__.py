"""Training commands: ``python -m train <command>``."""
from core.cli import dispatch

COMMANDS = {
    "prepare-dataset": ("train.commands.prepare_dataset", "Freeze a recorded selection into replay chunks"),
    "train": ("train.commands.train_policy", "Train one policy with the configured model and method"),
    "pipeline": ("train.commands.pipeline", "Single-GPU prepare → rewards → train pipeline"),
    "export-checkpoint": ("train.commands.export_checkpoint", "Export a training checkpoint as a policy"),
    "export-policy": ("train.commands.export_policy", "Copy a policy for deployment with a new base path"),
    "setup-pi05": ("train.commands.setup_pi05", "Install pinned OpenPI into the dedicated pi05 environment"),
    "prepare-pi05": ("train.commands.prepare_pi05", "Convert π₀.₅ base weights and bind normalization"),
}

if __name__ == "__main__":
    raise SystemExit(dispatch("train", COMMANDS))
