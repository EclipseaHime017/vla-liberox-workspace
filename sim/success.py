"""LIBERO success semantics.

LIBERO exposes benchmark success as the environment ``done`` flag. All configured
tasks use the same evaluation semantics, including composite tasks.
"""


class LiberoEvaluator:
    @staticmethod
    def success(done: bool) -> bool:
        return bool(done)
