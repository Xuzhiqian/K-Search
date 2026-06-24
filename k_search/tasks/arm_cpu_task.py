"""Armv8 CPU task adapter for K-Search."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Dict

from k_search.kernel_generators.kernel_generator_prompts import CPP_CODE_FORMAT, CPU_OPTIMIZATION_HINTS
from k_search.tasks.arm_cpu.evaluator import ArmCpuEvalConfig, evaluate_arm_cpu_solution
from k_search.tasks.task_base import (
    BuildSpec,
    EvalResult,
    Solution,
    SourceFile,
    SupportedLanguages,
    load_ksearch_solution_json,
    solution_from_json_dict,
)


@dataclass(frozen=True)
class ArmCpuTaskConfig:
    task_path: str
    target_cpu: str = "armv8-a"
    cpu_features: list[str] | None = None
    cxx: str = "c++"
    cxxflags: list[str] | None = None
    warmup_runs: int = 10
    iterations: int = 100
    num_trials: int = 3
    timeout_seconds: int = 300
    keep_tmp: bool = False
    vectorization_report: bool = False
    max_failure_excerpt_chars: int = 6000


class ArmCpuTask:
    """Task wrapper for generated C/C++ kernels on Armv8/AArch64 CPUs."""

    def __init__(
        self,
        *,
        task_path: str,
        target_cpu: str = "armv8-a",
        cpu_features: list[str] | None = None,
        cxx: str = "c++",
        cxxflags: list[str] | None = None,
        warmup_runs: int = 10,
        iterations: int = 100,
        num_trials: int = 3,
        timeout_seconds: int = 300,
        artifacts_dir: str | None = None,
        keep_tmp: bool = False,
        vectorization_report: bool = False,
    ) -> None:
        self._cfg = ArmCpuTaskConfig(
            task_path=str(task_path),
            target_cpu=str(target_cpu or "armv8-a"),
            cpu_features=list(cpu_features or []),
            cxx=str(cxx or "c++"),
            cxxflags=list(cxxflags or _default_cxxflags(target_cpu=target_cpu, cpu_features=cpu_features)),
            warmup_runs=int(warmup_runs),
            iterations=int(iterations),
            num_trials=int(num_trials),
            timeout_seconds=int(timeout_seconds),
            keep_tmp=bool(keep_tmp),
            vectorization_report=bool(vectorization_report),
        )
        self._ksearch_artifacts_dir = str(artifacts_dir) if artifacts_dir else None
        self._definition = _load_task_definition(Path(task_path))
        self._name = str(self._definition.get("name") or Path(task_path).stem or "armcpu_task")
        self._solutions: dict[str, Solution] = {}
        self._last_round_trace_logs_for_prompt = ""
        self._last_round_passed_count = 0
        self._last_round_total_workloads = 1
        self._last_round_summary_line = ""

    @property
    def name(self) -> str:
        return self._name

    def get_definition_text(self, language: str | None = None) -> str:
        signature = str(self._definition.get("signature") or "").strip()
        description = str(self._definition.get("description") or "").strip()
        constraints = self._definition.get("constraints")
        reference = str(self._definition.get("reference_source") or "").strip()
        includes = self._definition.get("allowed_includes")
        allow_threads = bool(self._definition.get("allow_threads", False))

        lines = [
            "# Armv8 CPU Kernel Task",
            f"Name: {self._name}",
            f"Target CPU: {self._cfg.target_cpu}",
            f"CPU features: {', '.join(self._cfg.cpu_features or []) or 'neon'}",
            "",
            "## Objective",
            description or "Generate a high-performance C++17 CPU kernel that matches the reference behavior.",
        ]
        if signature:
            lines.extend(["", "## Required C ABI Signature", signature])
        if constraints:
            lines.extend(["", "## Constraints", _render_json_or_text(constraints)])
        lines.extend(
            [
                "",
                "## Generated File Contract",
                "- Return kernel.h, kernel.cpp, and main.cpp XML blocks.",
                "- kernel.h must declare the required extern \"C\" entrypoint.",
                "- kernel.cpp must implement the optimized kernel.",
                "- main.cpp may contain helper code but must not define main().",
                "- Do not use CUDA, Triton, MLX, Metal, GPU APIs, tensor cores, warps, GPU blocks, or GPU shared memory.",
            ]
        )
        if allow_threads:
            lines.append("- Threading is allowed only when it does not change numerical results.")
        else:
            lines.append("- Do not use OpenMP, pthreads, std::thread, or other threading APIs.")
        if includes:
            lines.extend(["", "## Allowed Includes", _render_json_or_text(includes)])
        if reference:
            lines.extend(["", "Reference Implementation:", reference])
        lines.extend(
            [
                "",
                "## Evaluation Harness Contract",
                "The harness includes kernel.h, runs correctness and timing checks, and prints KSEARCH_STATUS, KSEARCH_KERNEL_MS, and KSEARCH_REFERENCE_MS.",
                f"Warmup runs: {self._cfg.warmup_runs}",
                f"Iterations: {self._cfg.iterations}",
                f"Trials: {self._cfg.num_trials}",
            ]
        )
        return "\n".join(lines).strip() + "\n"

    def get_generation_prompt(self, *, language: str, target_gpu: str) -> str:
        return (
            f"{self.get_definition_text(language=language).strip()}\n\n"
            f"{CPU_OPTIMIZATION_HINTS.strip()}\n\n"
            f"{CPP_CODE_FORMAT.strip()}\n"
        )

    def get_optimization_prompt(
        self,
        *,
        language: str,
        target_gpu: str,
        trace_logs: str,
        current_code: str,
        current_best: str | None = None,
        previous_round_summary: str | None = None,
    ) -> str:
        parts = [
            self.get_definition_text(language=language).strip(),
            "Current implementation:\n" + str(current_code or "").strip(),
        ]
        if previous_round_summary:
            parts.append("Previous round summary:\n" + str(previous_round_summary).strip())
        if trace_logs:
            parts.append("Compiler/runtime/correctness feedback:\n" + str(trace_logs).strip())
        if current_best:
            parts.append("Current best:\n" + str(current_best).strip())
        parts.extend([CPU_OPTIMIZATION_HINTS.strip(), CPP_CODE_FORMAT.strip()])
        return "\n\n".join(p for p in parts if p.strip())

    def get_code_format_text(self, *, language: str, target_gpu: str) -> str:
        return CPP_CODE_FORMAT.strip()

    def get_per_task_requirement_text(self, *, language: str, target_gpu: str, phase: str = "") -> str:
        return (
            "Arm CPU requirements:\n"
            f"- Target CPU/features: {self._cfg.target_cpu} {', '.join(self._cfg.cpu_features or [])}\n"
            "- Preserve the exact C ABI signature and generated file contract.\n"
            "- Prioritize cache locality, SIMD-friendly loops, and low memory traffic."
        )

    def get_baseline_targets_text(self) -> str:
        return "Primary objective: maximize reference_latency_ms / kernel_latency_ms while passing correctness."

    def make_solution_from_generated_code(
        self,
        *,
        cleaned_code: Any,
        raw_code: Any,
        round_num: int,
        model_name: str,
        target_gpu: str,
        language: str,
    ) -> Solution:
        if not isinstance(cleaned_code, dict):
            cleaned_code = {}
        files = {
            "kernel.h": str(cleaned_code.get("kernel.h", "") or ""),
            "kernel.cpp": str(cleaned_code.get("kernel.cpp", "") or ""),
            "main.cpp": str(cleaned_code.get("main.cpp", "") or ""),
        }
        missing = [name for name, content in files.items() if name != "main.cpp" and not content.strip()]
        if missing:
            raise ValueError(f"Generated C++ solution is missing required files: {missing}")

        safe_model = str(model_name or "model").replace("/", "_").replace("\\", "_")
        sol_name = f"{safe_model}_{self._name}_cpp_r{int(round_num)}"
        sol = Solution(
            name=sol_name,
            definition=self._name,
            author=str(model_name),
            spec=BuildSpec(
                language=SupportedLanguages.CPP,
                target_hardware=[self._cfg.target_cpu, *(self._cfg.cpu_features or [])],
                entry_point="kernel.cpp::kernel_entry",
                dependencies=[],
            ),
            sources=[SourceFile(path=k, content=v) for k, v in files.items()],
            description=f"Arm CPU C++ kernel for {self._name} (round {round_num})",
        )
        self._solutions[sol_name] = sol
        return sol

    def get_solution(self, solution_name: str) -> Solution | None:
        name = str(solution_name or "")
        if name in self._solutions:
            return self._solutions[name]
        try:
            d = load_ksearch_solution_json(
                solution_ref=name,
                definition_name=self._name,
                artifacts_dir=self._ksearch_artifacts_dir,
            )
            sol = solution_from_json_dict(d)
            if sol.definition != self._name:
                return None
            self._solutions[sol.name] = sol
            return sol
        except Exception:
            return None

    def code_for_world_model_from_raw(self, *, raw: Any, language: str) -> str:
        return str(raw or "")

    def seed_eval_for_base_solution(self, *, base_solution: Solution, config: Any = None) -> EvalResult:
        return self.run_benchmark(solution=base_solution, config=config, dump_traces=False, round_num=0)

    def run_benchmark(
        self,
        *,
        solution: Solution,
        config: Any = None,
        dump_traces: bool = False,
        round_num: int | None = None,
    ) -> EvalResult:
        sources = {sf.path: sf.content for sf in (solution.sources or [])}
        harness = str(self._definition.get("harness_source") or "").strip()
        if not harness:
            return self._failed_eval("Task definition must provide harness_source for evaluation.", round_num)

        eval_cfg = ArmCpuEvalConfig(
            cxx=self._cfg.cxx,
            cxxflags=list(self._cfg.cxxflags or []),
            timeout_seconds=int(self._cfg.timeout_seconds),
            keep_tmp=bool(self._cfg.keep_tmp),
            max_output_chars=int(self._cfg.max_failure_excerpt_chars),
            vectorization_report=bool(self._cfg.vectorization_report),
        )
        summary = evaluate_arm_cpu_solution(sources=sources, harness_source=harness, config=eval_cfg)
        passed = str(summary.status).lower() == "passed"
        self._last_round_passed_count = 1 if passed else 0
        self._last_round_total_workloads = 1
        self._last_round_trace_logs_for_prompt = _format_feedback(summary)
        speedup = summary.speedup_factor if passed else None
        score = float(speedup) if isinstance(speedup, (int, float)) and speedup > 0 else None
        status = "passed" if passed else str(summary.status or "failed")
        er = EvalResult(
            status=status,
            latency_ms=summary.latency_ms,
            reference_latency_ms=summary.reference_latency_ms,
            mean_vs_baseline_factor=speedup,
            speedup_factor=speedup,
            log_excerpt=self._last_round_trace_logs_for_prompt,
            metrics={
                "score_name": "speedup_vs_reference",
                "score": score,
                "compile_command": " ".join(summary.compile_command),
                "workdir": summary.workdir,
            },
        )
        self._last_round_summary_line = _summary_line(self._name, round_num, er)
        print(self._last_round_summary_line, flush=True)
        if not passed and self._last_round_trace_logs_for_prompt:
            print(self._last_round_trace_logs_for_prompt, flush=True)
        return er

    def run_final_evaluation(
        self,
        *,
        solutions: list[Solution],
        config: Any = None,
        dump_traces: bool = False,
        workload_limit: int | None = None,
    ) -> dict[str, Any]:
        rows: list[dict[str, Any]] = []
        for sol in solutions or []:
            er = self.run_benchmark(solution=sol, config=config, dump_traces=False, round_num=None)
            rows.append(
                {
                    "solution": sol.name,
                    "status": er.status,
                    "latency_ms": er.latency_ms,
                    "reference_latency_ms": er.reference_latency_ms,
                    "speedup": er.speedup_factor,
                    "score": er.metrics.get("score") if isinstance(er.metrics, dict) else None,
                    "log_excerpt": er.log_excerpt,
                }
            )
        return {
            "task": self._name,
            "target_cpu": self._cfg.target_cpu,
            "cpu_features": list(self._cfg.cpu_features or []),
            "cxx": self._cfg.cxx,
            "cxxflags": list(self._cfg.cxxflags or []),
            "solutions": rows,
        }

    def get_last_round_trace_logs_for_prompt(self) -> str:
        return self._last_round_trace_logs_for_prompt

    def get_last_round_passed_count(self) -> int:
        return int(self._last_round_passed_count)

    def get_last_round_total_workloads(self) -> int:
        return int(self._last_round_total_workloads)

    def get_config_for_logging(self) -> Dict[str, Any]:
        return {
            "task_source": "armcpu",
            "task_path": self._cfg.task_path,
            "task_name": self._name,
            "target_cpu": self._cfg.target_cpu,
            "cpu_features": list(self._cfg.cpu_features or []),
            "cxx": self._cfg.cxx,
            "cxxflags": list(self._cfg.cxxflags or []),
            "warmup_runs": self._cfg.warmup_runs,
            "iterations": self._cfg.iterations,
            "num_trials": self._cfg.num_trials,
            "timeout_seconds": self._cfg.timeout_seconds,
        }

    def _failed_eval(self, message: str, round_num: int | None) -> EvalResult:
        self._last_round_passed_count = 0
        self._last_round_total_workloads = 1
        self._last_round_trace_logs_for_prompt = str(message)
        er = EvalResult(
            status="failed",
            latency_ms=None,
            reference_latency_ms=None,
            mean_vs_baseline_factor=None,
            speedup_factor=None,
            log_excerpt=str(message),
            metrics={"score_name": "speedup_vs_reference", "score": None},
        )
        self._last_round_summary_line = _summary_line(self._name, round_num, er)
        print(self._last_round_summary_line, flush=True)
        print(message, flush=True)
        return er


def _load_task_definition(path: Path) -> dict[str, Any]:
    p = path.expanduser()
    if p.is_dir():
        for name in ("task.json", "task.yaml", "task.yml"):
            candidate = p / name
            if candidate.exists():
                p = candidate
                break
    if not p.exists():
        raise FileNotFoundError(f"Arm CPU task definition not found: {p}")
    text = p.read_text(encoding="utf-8")
    if p.suffix.lower() == ".json":
        obj = json.loads(text)
    elif p.suffix.lower() in (".yaml", ".yml"):
        try:
            import yaml  # type: ignore
        except Exception as e:
            raise RuntimeError("YAML task files require PyYAML; use JSON or install PyYAML.") from e
        obj = yaml.safe_load(text)
    else:
        raise ValueError(f"Unsupported task definition extension: {p.suffix}")
    if not isinstance(obj, dict):
        raise TypeError("Arm CPU task definition must be a JSON/YAML object")
    return obj


def _default_cxxflags(*, target_cpu: str, cpu_features: list[str] | None) -> list[str]:
    features = {str(x).strip().lower() for x in (cpu_features or []) if str(x).strip()}
    march = str(target_cpu or "armv8-a").strip()
    if "neon" in features or "simd" in features:
        if "+simd" not in march:
            march += "+simd"
    if "sve" in features and "+sve" not in march:
        march += "+sve"
    return ["-O3", "-std=c++17", f"-march={march}", "-ffast-math", "-fno-exceptions"]


def _render_json_or_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, indent=2, ensure_ascii=False)
    except Exception:
        return str(value)


def _format_feedback(summary: Any) -> str:
    lines = [f"status={summary.status}"]
    if summary.latency_ms is not None:
        lines.append(f"kernel_latency_ms={summary.latency_ms:.6f}")
    if summary.reference_latency_ms is not None:
        lines.append(f"reference_latency_ms={summary.reference_latency_ms:.6f}")
    if summary.speedup_factor is not None:
        lines.append(f"speedup_vs_reference={summary.speedup_factor:.6f}x")
    if summary.compile_command:
        lines.append("compile_command=" + " ".join(summary.compile_command))
    if summary.log_excerpt:
        lines.append("log_excerpt:\n" + str(summary.log_excerpt).strip())
    return "\n".join(lines).strip()


def _summary_line(task_name: str, round_num: int | None, er: EvalResult) -> str:
    rn = str(round_num) if round_num is not None else "final"
    sp = er.speedup_factor
    lat = er.latency_ms
    sp_s = f"{float(sp):.3f}x" if isinstance(sp, (int, float)) else "-"
    lat_s = f"{float(lat):.6f}ms" if isinstance(lat, (int, float)) else "-"
    return f"[{task_name}] Round {rn}: status={er.status} | latency={lat_s} | speedup={sp_s}"
