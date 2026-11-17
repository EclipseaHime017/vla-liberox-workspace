"""Reward commands: ``python -m rewards <command>``."""
from core.cli import dispatch

COMMANDS = {
    "annotate": ("rewards.commands.annotate", "Run RynnValue on a prepared dataset (rynnvalue environment)"),
    "materialize": ("rewards.commands.materialize", "Derive sparse/PBRS/Stage/fusion rewards from saved signals"),
    "verify": ("rewards.commands.verify", "Check the pinned RynnValue checkout and environment"),
    "robometer-evaluate": ("rewards.commands.robometer_evaluate", "Score trajectories with Robometer (robometer environment)"),
    "robometer-verify": ("rewards.commands.robometer_verify", "Check the pinned Robometer runtime"),
}

if __name__ == "__main__":
    raise SystemExit(dispatch("rewards", COMMANDS))
