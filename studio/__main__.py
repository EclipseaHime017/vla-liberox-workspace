"""Platform commands: ``python -m studio`` starts the Web UI; ``--help`` lists the rest."""
from core.cli import dispatch

COMMANDS = {
    "serve": ("studio.commands.serve", "Start the simulation, takeover and data Web UI (default)"),
    "evaluate-policies": ("studio.commands.evaluate_policies", "Run one batch policy evaluation"),
    "critic-diagnostic": ("studio.commands.critic_diagnostic", "Run one critic diagnostic phase"),
    "migrate-run-layout": ("studio.commands.migrate_run_layout", "Preview, apply or roll back the recording layout"),
    "setup-factr": ("studio.commands.setup_factr", "Install or check the pinned FACTR runtime"),
    "check-factr": ("studio.commands.check_factr", "Interactive FACTR calibration and control check"),
    "check-spacemouse": ("studio.commands.check_spacemouse", "SpaceMouse input and manual-control check"),
    "check-model": ("studio.commands.check_model", "Smoke-test one model through the running platform"),
}

if __name__ == "__main__":
    raise SystemExit(dispatch("studio", COMMANDS, default="serve"))
