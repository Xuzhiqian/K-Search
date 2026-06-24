"""Compile and evaluate generated Armv8 CPU C/C++ kernels.

The evaluator intentionally uses a simple harness contract. A task supplies a
`harness_source` C++ translation unit that includes `kernel.h`, runs correctness
checks and timing, then prints machine-readable lines:

KSEARCH_STATUS=passed|failed
KSEARCH_KERNEL_MS=<float>
KSEARCH_REFERENCE_MS=<float>
KSEARCH_LOG=<short text>
"""

from __future__ import annotations

from dataclasses import dataclass, field
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
from typing import Any


@dataclass(frozen=True)
class ArmCpuEvalConfig:
    cxx: str = "c++"
    cxxflags: list[str] = field(
        default_factory=lambda: ["-O3", "-std=c++17", "-march=armv8-a+simd", "-ffast-math", "-fno-exceptions"]
    )
    timeout_seconds: int = 300
    keep_tmp: bool = False
    max_output_chars: int = 6000
    vectorization_report: bool = False


@dataclass(frozen=True)
class ArmCpuEvalSummary:
    status: str
    latency_ms: float | None = None
    reference_latency_ms: float | None = None
    speedup_factor: float | None = None
    log_excerpt: str = ""
    compile_command: list[str] = field(default_factory=list)
    workdir: str | None = None


def evaluate_arm_cpu_solution(
    *,
    sources: dict[str, str],
    harness_source: str,
    config: ArmCpuEvalConfig,
) -> ArmCpuEvalSummary:
    tmp_ctx = tempfile.TemporaryDirectory(prefix="ksearch-armcpu-")
    tmpdir = Path(tmp_ctx.name)
    keep_tmp = bool(config.keep_tmp)
    try:
        _write_sources(tmpdir=tmpdir, sources=sources, harness_source=harness_source)
        cmd = _compile_command(tmpdir=tmpdir, config=config)
        compile_proc = subprocess.run(
            cmd,
            cwd=str(tmpdir),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=int(config.timeout_seconds),
        )
        if compile_proc.returncode != 0:
            return ArmCpuEvalSummary(
                status="compile_error",
                log_excerpt=_bounded_output(compile_proc.stdout, compile_proc.stderr, config.max_output_chars),
                compile_command=cmd,
                workdir=(str(tmpdir) if keep_tmp else None),
            )

        exe = tmpdir / "bench"
        run_proc = subprocess.run(
            [str(exe)],
            cwd=str(tmpdir),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=int(config.timeout_seconds),
        )
        out = (run_proc.stdout or "") + "\n" + (run_proc.stderr or "")
        if run_proc.returncode != 0:
            return ArmCpuEvalSummary(
                status="runtime_error",
                log_excerpt=_truncate(out, config.max_output_chars),
                compile_command=cmd,
                workdir=(str(tmpdir) if keep_tmp else None),
            )

        parsed = _parse_harness_output(out)
        status = str(parsed.get("status") or "failed")
        kernel_ms = parsed.get("kernel_ms")
        ref_ms = parsed.get("reference_ms")
        speedup = None
        if isinstance(kernel_ms, float) and kernel_ms > 0 and isinstance(ref_ms, float) and ref_ms > 0:
            speedup = ref_ms / kernel_ms
        log = str(parsed.get("log") or "").strip()
        if not log:
            log = _truncate(out, config.max_output_chars)
        return ArmCpuEvalSummary(
            status=("passed" if status.lower() == "passed" else "failed"),
            latency_ms=kernel_ms,
            reference_latency_ms=ref_ms,
            speedup_factor=speedup,
            log_excerpt=_truncate(log, config.max_output_chars),
            compile_command=cmd,
            workdir=(str(tmpdir) if keep_tmp else None),
        )
    except subprocess.TimeoutExpired as e:
        return ArmCpuEvalSummary(
            status="timeout",
            log_excerpt=f"Timed out after {int(config.timeout_seconds)} seconds: {e}",
            workdir=(str(tmpdir) if keep_tmp else None),
        )
    except Exception as e:
        return ArmCpuEvalSummary(
            status="evaluator_error",
            log_excerpt=f"{type(e).__name__}: {e}",
            workdir=(str(tmpdir) if keep_tmp else None),
        )
    finally:
        if keep_tmp:
            tmp_ctx.cleanup = lambda: None  # type: ignore[method-assign]
        else:
            tmp_ctx.cleanup()


def _write_sources(*, tmpdir: Path, sources: dict[str, str], harness_source: str) -> None:
    required = ("kernel.h", "kernel.cpp", "main.cpp")
    for name in required:
        content = str(sources.get(name, "") or "")
        if name != "main.cpp" and not content.strip():
            raise ValueError(f"Missing required generated source: {name}")
        (tmpdir / name).write_text(content, encoding="utf-8")
    (tmpdir / "harness.cpp").write_text(str(harness_source or ""), encoding="utf-8")


def _compile_command(*, tmpdir: Path, config: ArmCpuEvalConfig) -> list[str]:
    cxx = str(config.cxx or "c++")
    flags = [str(x) for x in (config.cxxflags or []) if str(x).strip()]
    if config.vectorization_report:
        flags.extend(["-fopt-info-vec-optimized", "-fopt-info-vec-missed"])
    sources = ["kernel.cpp", "main.cpp", "harness.cpp"]
    return [cxx, *flags, *sources, "-o", str(tmpdir / "bench")]


def _parse_harness_output(text: str) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for raw in str(text or "").splitlines():
        line = raw.strip()
        if not line.startswith("KSEARCH_") or "=" not in line:
            continue
        key, val = line.split("=", 1)
        key = key.strip().lower()
        val = val.strip()
        if key == "ksearch_status":
            out["status"] = val
        elif key == "ksearch_kernel_ms":
            out["kernel_ms"] = _float_or_none(val)
        elif key == "ksearch_reference_ms":
            out["reference_ms"] = _float_or_none(val)
        elif key == "ksearch_log":
            out["log"] = val
    return out


def _float_or_none(s: str) -> float | None:
    try:
        return float(s)
    except Exception:
        return None


def _bounded_output(stdout: str, stderr: str, max_chars: int) -> str:
    parts = []
    if stdout:
        parts.append("[stdout]\n" + stdout)
    if stderr:
        parts.append("[stderr]\n" + stderr)
    return _truncate("\n\n".join(parts), max_chars)


def _truncate(text: str, max_chars: int) -> str:
    s = str(text or "")
    n = int(max_chars or 0)
    if n <= 0 or len(s) <= n:
        return s
    return s[:n] + "...<truncated>..."
