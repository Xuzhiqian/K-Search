"""Task adapters for K-Search."""

from k_search.tasks.task_base import BuildSpec, EvalResult, Solution, SourceFile, SupportedLanguages, Task, code_from_solution

__all__ = [
    "BuildSpec",
    "EvalResult",
    "Solution",
    "SourceFile",
    "SupportedLanguages",
    "Task",
    "code_from_solution",
]

try:  # pragma: no cover
    from k_search.tasks.arm_cpu_task import ArmCpuTask

    __all__.append("ArmCpuTask")
except Exception:
    ArmCpuTask = None  # type: ignore


