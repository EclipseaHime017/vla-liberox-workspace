"""LIBERO success semantics.

LIBERO exposes benchmark success as the environment ``done`` flag. No task is
special-cased, including the three selectable LEVEL1 tasks.
"""


class LiberoEvaluator:
    @staticmethod
    def success(done: bool) -> bool:
        return bool(done)
