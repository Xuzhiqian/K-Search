"""World-model codegen prompts for Armv8 CPU kernels."""

from __future__ import annotations

from .kernel_generator_prompts import CPU_OPTIMIZATION_HINTS, CPP_CODE_FORMAT


CPU_ACTION_PROMPT = """You are implementing a SPECIFIC NEXT ACTION on top of a known-good Armv8 CPU C++ baseline for {target_cpu}.

Original Specification:
{definition}

Known-Good Base Implementation:
{base_code}

Chosen Next Action:
{action_text}

{code_format}

Rules:
- Implement ONLY the chosen action; keep unrelated code as close as possible to the base implementation.
- Preserve the exact C ABI function signature declared in kernel.h.
- Do not use CUDA, Triton, MLX, Metal, GPU APIs, tensor cores, warps, GPU thread blocks, or GPU shared memory.
- Return only the full updated XML blocks.

{hints}

Generate the updated implementation:"""


CPU_DEBUG_PROMPT = """You are in a debug-and-improve loop for an Armv8 CPU C++ kernel.
The current implementation may be buggy OR already correct-but-slower-than-desired.

Original Specification:
{definition}

Known-Good Base Implementation:
{base_code}

Current Implementation:
{buggy_code}

Performance Summary:
{perf_summary}

Failure Logs:
{trace_logs}

Chosen Next Action:
{action_text}

Debug-and-improve round: {debug_round}/{max_rounds}

{code_format}

Rules:
- If the current implementation FAILED: fix compile, link, runtime, or correctness issues first.
- If the current implementation PASSED: improve CPU performance while preserving correctness.
- Keep changes minimal and aligned with the chosen action.
- Return only the full corrected XML blocks.

{hints}

Generate the corrected implementation:"""


CPU_IMPROVE_PROMPT = """You are improving an Armv8 CPU C++ kernel.
The current implementation may be correct-but-slower-than-desired, or it may have regressed.

Original Specification:
{definition}

Cycle-Best Base Implementation:
{base_code}

Current Implementation:
{current_code}

Performance Summary:
{perf_summary}

Recent Logs:
{trace_logs}

Improve round: {debug_round}/{max_rounds}

{code_format}

Rules:
- If the current implementation FAILED: fix compile, link, runtime, or correctness issues first.
- If the current implementation PASSED: improve CPU performance while preserving correctness.
- Optimize around cache locality, SIMD/vectorization, branch behavior, register pressure, and memory bandwidth.
- Return only the full corrected XML blocks.

{hints}

Generate the improved implementation:"""


def _require_cpu_language(language: str) -> None:
    lang = str(language or "").strip().lower()
    if lang not in ("cpp", "c++", "c"):
        raise ValueError(f"Unsupported language for Arm CPU prompt: {language}")


def get_generate_code_from_action_prompt_from_text(
    language: str,
    *,
    definition_text: str,
    base_code: str,
    action_text: str,
    code_format: str = "",
    target_gpu: str = "armv8-a",
) -> str:
    _require_cpu_language(language)
    return CPU_ACTION_PROMPT.format(
        definition=str(definition_text or "").strip(),
        base_code=str(base_code or "").strip(),
        action_text=str(action_text or "").strip(),
        target_cpu=str(target_gpu or "armv8-a"),
        code_format=(str(code_format or "").strip() or CPP_CODE_FORMAT.strip()),
        hints=CPU_OPTIMIZATION_HINTS.strip(),
    )


def get_generate_code_from_spec_with_action_prompt_from_text(
    language: str,
    *,
    definition_text: str,
    action_text: str,
    code_format: str = "",
    target_gpu: str = "armv8-a",
) -> str:
    _require_cpu_language(language)
    return get_generate_code_from_action_prompt_from_text(
        language,
        definition_text=definition_text,
        base_code="(no base code; start from the specification)",
        action_text=action_text,
        code_format=code_format,
        target_gpu=target_gpu,
    )


def get_debug_and_improve_from_spec_prompt_from_text(
    language: str,
    *,
    definition_text: str,
    trace_logs: str,
    current_code: str,
    action_text: str,
    code_format: str = "",
    debug_round: int,
    max_rounds: int = 5,
    target_gpu: str = "armv8-a",
    perf_summary: str = "",
    base_code: str = "(no base code; start from spec)",
) -> str:
    return get_debug_generated_code_prompt_from_text(
        language,
        definition_text=definition_text,
        trace_logs=trace_logs,
        base_code=base_code,
        buggy_code=current_code,
        action_text=action_text,
        code_format=code_format,
        debug_round=debug_round,
        max_rounds=max_rounds,
        target_gpu=target_gpu,
        perf_summary=perf_summary,
    )


def get_debug_generated_code_prompt_from_text(
    language: str,
    *,
    definition_text: str,
    trace_logs: str,
    base_code: str,
    buggy_code: str,
    action_text: str,
    code_format: str = "",
    debug_round: int,
    max_rounds: int = 5,
    target_gpu: str = "armv8-a",
    perf_summary: str = "",
) -> str:
    _require_cpu_language(language)
    dr, mr = _bounded_rounds(debug_round, max_rounds)
    return CPU_DEBUG_PROMPT.format(
        definition=str(definition_text or "").strip(),
        base_code=str(base_code or "").strip(),
        buggy_code=str(buggy_code or "").strip(),
        perf_summary=str(perf_summary or "").strip() or "(none)",
        trace_logs=str(trace_logs or "").strip() or "(no logs)",
        action_text=str(action_text or "").strip(),
        debug_round=dr,
        max_rounds=mr,
        target_cpu=str(target_gpu or "armv8-a"),
        code_format=(str(code_format or "").strip() or CPP_CODE_FORMAT.strip()),
        hints=CPU_OPTIMIZATION_HINTS.strip(),
    )


def get_improve_from_spec_prompt_from_text(
    language: str,
    *,
    definition_text: str,
    trace_logs: str,
    current_code: str,
    code_format: str = "",
    debug_round: int,
    max_rounds: int = 5,
    target_gpu: str = "armv8-a",
    perf_summary: str = "",
    base_code: str = "(no base code; start from spec)",
) -> str:
    return get_improve_generated_code_prompt_from_text(
        language,
        definition_text=definition_text,
        trace_logs=trace_logs,
        base_code=base_code,
        current_code=current_code,
        code_format=code_format,
        debug_round=debug_round,
        max_rounds=max_rounds,
        target_gpu=target_gpu,
        perf_summary=perf_summary,
    )


def get_improve_generated_code_prompt_from_text(
    language: str,
    *,
    definition_text: str,
    trace_logs: str,
    base_code: str,
    current_code: str,
    code_format: str = "",
    debug_round: int,
    max_rounds: int = 5,
    target_gpu: str = "armv8-a",
    perf_summary: str = "",
) -> str:
    _require_cpu_language(language)
    dr, mr = _bounded_rounds(debug_round, max_rounds)
    return CPU_IMPROVE_PROMPT.format(
        definition=str(definition_text or "").strip(),
        base_code=str(base_code or "").strip(),
        current_code=str(current_code or "").strip(),
        perf_summary=str(perf_summary or "").strip() or "(none)",
        trace_logs=str(trace_logs or "").strip() or "(no logs)",
        debug_round=dr,
        max_rounds=mr,
        target_cpu=str(target_gpu or "armv8-a"),
        code_format=(str(code_format or "").strip() or CPP_CODE_FORMAT.strip()),
        hints=CPU_OPTIMIZATION_HINTS.strip(),
    )


def _bounded_rounds(debug_round: int, max_rounds: int) -> tuple[int, int]:
    try:
        dr = int(debug_round)
    except Exception:
        dr = 1
    try:
        mr = int(max_rounds)
    except Exception:
        mr = 1
    if mr < 1:
        mr = 1
    if dr < 1:
        dr = 1
    if dr > mr:
        dr = mr
    return dr, mr
