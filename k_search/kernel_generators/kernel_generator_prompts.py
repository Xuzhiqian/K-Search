"""Prompt templates for Armv8 CPU kernel generation."""

from __future__ import annotations


CPU_OPTIMIZATION_HINTS = """
Optimization guidance:
- Target Armv8/AArch64 CPU execution. Do not use CUDA, Triton, MLX, Metal, GPU APIs, warp/block terminology, tensor cores, or shared-memory GPU concepts.
- Prefer C++17 with a C ABI entrypoint declared in kernel.h and implemented in kernel.cpp.
- Consider cache locality, loop ordering, loop tiling, NEON SIMD, optional SVE when requested, alignment, prefetch, branch reduction, unrolling, and avoiding unnecessary memory traffic.
- Use OpenMP or pthreads only if the task definition explicitly allows threading.
- Keep numerical behavior within the requested tolerance and preserve the exact entrypoint signature.
"""


CPP_CODE_FORMAT = """
Return exactly these XML blocks and no markdown:
<header_file name="kernel.h">
// declarations only
</header_file>
<source_file name="kernel.cpp">
// optimized implementation
</source_file>
<main_file name="main.cpp">
// optional helper/wrapper code; do not define main()
</main_file>
"""


CPP_PROMPT = """You are a CPU kernel performance engineer. Generate a C++17 kernel optimized for {target_cpu}.

Specification:
{definition}

{per_task_requirement}

{hints}

{code_format}

Generate the implementation:"""


CPP_OPTIMIZATION_PROMPT = """You are optimizing a C++17 Armv8 CPU kernel for {target_cpu}.
The current implementation may be incorrect, may fail to compile, or may be slower than desired.

Original Specification:
{definition}

Current Implementation Status:
{trace_logs}

Current Implementation:
{current_code}

{per_task_requirement}

{hints}

{extra_context}

{code_format}

Generate the corrected and optimized implementation:"""


def _normalize_language(language: str) -> str:
    lang = str(language or "").strip().lower()
    if lang in ("c++", "cpp"):
        return "cpp"
    if lang == "c":
        return "c"
    if lang == "python":
        return "python"
    return lang


def get_prompt_from_definition_text(
    language: str,
    definition_text: str,
    target_gpu: str = "armv8-a",
    *,
    per_task_requirement: str = "",
) -> str:
    """Task-agnostic prompt builder for CPU code generation."""
    lang = _normalize_language(language)
    if lang not in ("cpp", "c"):
        raise ValueError(f"Unsupported language for Arm CPU generation: {language}")
    return CPP_PROMPT.format(
        definition=str(definition_text or "").strip(),
        target_cpu=str(target_gpu or "armv8-a"),
        per_task_requirement=str(per_task_requirement or "").strip(),
        hints=CPU_OPTIMIZATION_HINTS.strip(),
        code_format=CPP_CODE_FORMAT.strip(),
    )


def get_optimization_prompt_from_definition_text(
    language: str,
    *,
    definition_text: str,
    trace_logs: str,
    current_code: str,
    target_gpu: str = "armv8-a",
    current_best: str | None = None,
    previous_round_summary: str | None = None,
    per_task_requirement: str = "",
) -> str:
    """Task-agnostic optimization prompt builder for CPU code generation."""
    lang = _normalize_language(language)
    if lang not in ("cpp", "c"):
        raise ValueError(f"Unsupported language for Arm CPU optimization: {language}")
    extra_context = _build_extra_context(
        current_best=current_best,
        previous_round_summary=previous_round_summary,
    )
    return CPP_OPTIMIZATION_PROMPT.format(
        definition=str(definition_text or "").strip(),
        trace_logs=str(trace_logs or "").strip(),
        current_code=str(current_code or "").strip(),
        target_cpu=str(target_gpu or "armv8-a"),
        per_task_requirement=str(per_task_requirement or "").strip(),
        hints=CPU_OPTIMIZATION_HINTS.strip(),
        extra_context=extra_context,
        code_format=CPP_CODE_FORMAT.strip(),
    )


def _build_extra_context(
    *,
    current_best: str | None,
    previous_round_summary: str | None,
) -> str:
    parts: list[str] = []
    if current_best and current_best.strip():
        parts.append("Current Best Solution So Far:\n" + current_best.strip())
    if previous_round_summary and previous_round_summary.strip():
        parts.append("Last Round Summary:\n" + previous_round_summary.strip())
    if not parts:
        return ""
    return "\n\n" + "\n\n".join(parts)
