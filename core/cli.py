"""Subcommand dispatch for ``python -m <package> <command>`` entry points."""
from __future__ import annotations

import importlib
import sys


def dispatch(package: str, commands: dict[str, tuple[str, str]], argv: list[str] | None = None,
             *, default: str | None = None) -> int:
    """Run ``commands[name]``'s ``main()`` with the remaining arguments."""
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv and default:
        argv = [default]
    if not argv or argv[0] in {"-h", "--help"}:
        width = max(map(len, commands))
        print(f"usage: python -m {package} <command> [options]\n\ncommands:")
        for name, (_, summary) in commands.items():
            print(f"  {name.ljust(width)}  {summary}")
        print(f"\nRun `python -m {package} <command> --help` for a command's options.")
        return 0 if argv else 2
    name, *rest = argv
    if name not in commands:
        print(f"Unknown command {name!r}; run `python -m {package} --help`.", file=sys.stderr)
        return 2
    module = importlib.import_module(commands[name][0])
    sys.argv = [f"python -m {package} {name}", *rest]
    result = module.main()
    return 0 if result is None else int(result)
