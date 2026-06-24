from __future__ import annotations

import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import shlex
from typing import Any, Optional
import uuid


def _persist_ksearch_solution(
    solution: Any, *, definition_name: str, artifacts_dir: Optional[str]
) -> Optional[Path]:
    try:
        from k_search.tasks.task_base import Solution as KSearchSolution
        from k_search.utils.paths import get_ksearch_artifacts_dir

        root = get_ksearch_artifacts_dir(base_dir=artifacts_dir, task_name=str(definition_name or "")).resolve()
        out_dir = root / "solutions" / str(definition_name or "__unknown__")
        out_dir.mkdir(parents=True, exist_ok=True)
        name = str(getattr(solution, "name", "") or "solution")
        safe_name = "".join(c if (c.isalnum() or c in ("-", "_", ".")) else "_" for c in name).strip("_")
        if not safe_name:
            safe_name = "solution"
        dest = out_dir / f"{safe_name}.json"
        obj = solution.to_dict() if isinstance(solution, KSearchSolution) else getattr(solution, "__dict__", {"solution": str(solution)})
        dest.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
        return dest
    except Exception as e:
        print(f"Error saving k-search solution: {e}")
        return None


def _persist_ksearch_eval_report(
    report: dict[str, Any],
    *,
    definition_name: str,
    solution_name: Optional[str],
    artifacts_dir: Optional[str],
) -> Optional[Path]:
    try:
        from k_search.utils.paths import get_ksearch_artifacts_dir

        root = get_ksearch_artifacts_dir(base_dir=artifacts_dir, task_name=str(definition_name or "")).resolve()
        out_dir = root / "eval" / str(definition_name or "__unknown__")
        out_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        sn = str(solution_name or "").strip()
        safe_sn = "".join(c if (c.isalnum() or c in ("-", "_", ".")) else "_" for c in sn) if sn else ""
        suffix = f"_{safe_sn}" if safe_sn else ""
        dest = out_dir / f"eval_report_{ts}{suffix}.json"
        dest.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        return dest
    except Exception as e:
        print(f"Error saving eval report: {e}")
        return None


def generate_and_evaluate(
    *,
    task: Any,
    model_name: str,
    base_url: Optional[str],
    api_key: Optional[str],
    language: str,
    target_cpu: str,
    max_opt_rounds: int,
    save_solutions: bool,
    save_results: bool,
    continue_from_solution: Optional[str] = None,
    continue_from_world_model: Optional[str] = None,
    enable_wandb: bool = False,
    wandb_project: Optional[str] = None,
    run_name: Optional[str] = None,
    enable_world_model: bool = False,
    wm_stagnation_window: int = 5,
    wm_max_difficulty: Optional[int] = None,
    artifacts_dir: Optional[str] = None,
) -> None:
    try:
        import wandb  # type: ignore
    except Exception:
        wandb = None

    wb_run = None
    if enable_wandb and wandb is not None:
        try:
            task_cfg = task.get_config_for_logging()
        except Exception:
            task_cfg = {}
        wb_run = wandb.init(
            project=wandb_project or os.getenv("WANDB_PROJECT", "ksearch-armcpu"),
            name=run_name or os.getenv("RUN_NAME"),
            config={
                "task": task_cfg,
                "generator": {
                    "model_name": model_name,
                    "language": language,
                    "target_cpu": target_cpu,
                },
                "max_opt_rounds": int(max_opt_rounds),
                "continue_from_solution": continue_from_solution,
                "continue_from_world_model": continue_from_world_model,
                "enable_world_model": bool(enable_world_model),
                "wm_stagnation_window": int(wm_stagnation_window),
                "wm_max_difficulty": wm_max_difficulty,
                "save_results": bool(save_results),
                "save_solutions": bool(save_solutions),
                "artifacts_dir": artifacts_dir,
            },
            reinit=True,
        )

    if enable_world_model:
        from k_search.kernel_generators.kernel_generator_world_model import WorldModelKernelGeneratorWithBaseline

        generator = WorldModelKernelGeneratorWithBaseline(
            model_name=model_name,
            language=language,
            target_gpu=target_cpu,
            api_key=api_key,
            base_url=base_url,
            artifacts_dir=artifacts_dir,
            wm_max_difficulty=wm_max_difficulty,
        )
        solution = generator.generate(
            task=task,
            max_opt_rounds=max_opt_rounds,
            wm_stagnation_window=int(wm_stagnation_window),
            continue_from_solution=continue_from_solution,
            continue_from_world_model=continue_from_world_model,
        )
    else:
        from k_search.kernel_generators.kernel_generator import KernelGenerator

        generator = KernelGenerator(
            model_name=model_name,
            language=language,
            target_gpu=target_cpu,
            api_key=api_key,
            base_url=base_url,
        )
        solution = generator.generate(
            task=task,
            max_opt_rounds=max_opt_rounds,
            continue_from_solution=continue_from_solution,
        )

    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    uid = uuid.uuid4().hex[:8]
    solution.name = f"{solution.name}_{ts}_{uid}"
    try:
        solution.description = (solution.description or "") + f" (generated {ts} uid={uid})"
    except Exception:
        pass

    def_name = str(getattr(task, "name", "") or "")
    if save_solutions:
        saved_path = _persist_ksearch_solution(solution, definition_name=def_name, artifacts_dir=artifacts_dir)
        if saved_path:
            print(f"[{def_name}] Saved solution to: {saved_path}")

    print(f"[{def_name}] Generated solution: {solution.name}")

    report = task.run_final_evaluation(solutions=[solution], config=None, dump_traces=bool(save_results))
    if save_results:
        saved = _persist_ksearch_eval_report(
            report,
            definition_name=def_name,
            solution_name=str(getattr(solution, "name", "") or ""),
            artifacts_dir=artifacts_dir,
        )
        if saved:
            print(f"[{def_name}] Saved eval report to: {saved}")

    if wb_run is not None and wandb is not None:
        try:
            wandb.finish()
        except Exception:
            pass


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate and evaluate Armv8 CPU kernels with K-Search.")
    parser.add_argument("--task-source", choices=["armcpu"], default="armcpu", help="Task backend to use.")
    parser.add_argument("--task-path", required=True, help="Path to an Arm CPU task JSON/YAML file or directory containing task.json.")
    parser.add_argument("--definition", default=None, help="Reserved for compatibility; Arm CPU task name comes from task definition.")
    parser.add_argument("--model-name", required=True, help="LLM model name.")
    parser.add_argument("--base-url", default=None, help="OpenAI-compatible API base URL.")
    parser.add_argument("--api-key", default=None, help="API key; if omitted, uses LLM_API_KEY.")
    parser.add_argument("--language", default="cpp", choices=["cpp", "c++", "c"], help="Target generated language.")
    parser.add_argument("--target-cpu", default="armv8-a", help="Target CPU architecture hint, e.g. armv8-a or armv8.2-a.")
    parser.add_argument("--cpu-features", default="neon", help="Comma-separated CPU features, e.g. neon,sve.")
    parser.add_argument("--cxx", default="c++", help="C++ compiler command on the Arm CPU machine.")
    parser.add_argument("--cxxflags", default=None, help="Override compiler flags as a shell-style string.")
    parser.add_argument("--warmup-runs", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--num-trials", type=int, default=3)
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--keep-tmp", action="store_true", help="Keep evaluator temp directories on the Arm CPU machine.")
    parser.add_argument("--vectorization-report", action="store_true", help="Ask compiler for vectorization reports when supported.")
    parser.add_argument("--max-opt-rounds", type=int, default=5, help="Max optimization rounds.")
    parser.add_argument("--no-save-results", action="store_true", help="Do not persist final eval report.")
    parser.add_argument("--save-solutions", action="store_true", help="Persist generated solution JSON.")
    parser.add_argument("--artifacts-dir", default=".ksearch", help="Base directory for K-Search artifacts.")
    parser.add_argument("--continue-from-solution", default=None, help="Resume from a persisted solution name or path.")
    parser.add_argument("--continue-from-world-model", default=None, help="Resume world-model JSON from 'auto' or path.")
    parser.add_argument("--world-model", action="store_true", help="Enable world-model prompting.")
    parser.add_argument("--wm-stagnation-window", type=int, default=5)
    parser.add_argument("--wm-max-difficulty", type=int, default=None)
    parser.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging.")
    parser.add_argument("--wandb-project", default=os.getenv("WANDB_PROJECT"), help="W&B project.")
    parser.add_argument("--run-name", default=os.getenv("RUN_NAME"), help="W&B run name.")

    args = parser.parse_args()
    api_key = args.api_key or os.getenv("LLM_API_KEY")
    if not api_key:
        raise ValueError("API key is required (pass --api-key or set LLM_API_KEY)")

    cpu_features = [x.strip() for x in str(args.cpu_features or "").split(",") if x.strip()]
    cxxflags = shlex.split(args.cxxflags) if args.cxxflags else None

    from k_search.tasks.arm_cpu_task import ArmCpuTask

    task = ArmCpuTask(
        task_path=args.task_path,
        target_cpu=args.target_cpu,
        cpu_features=cpu_features,
        cxx=args.cxx,
        cxxflags=cxxflags,
        warmup_runs=args.warmup_runs,
        iterations=args.iterations,
        num_trials=args.num_trials,
        timeout_seconds=args.timeout,
        artifacts_dir=args.artifacts_dir,
        keep_tmp=args.keep_tmp,
        vectorization_report=args.vectorization_report,
    )

    generate_and_evaluate(
        task=task,
        model_name=args.model_name,
        base_url=args.base_url,
        api_key=api_key,
        language=("cpp" if args.language == "c++" else args.language),
        target_cpu=args.target_cpu,
        max_opt_rounds=args.max_opt_rounds,
        save_solutions=args.save_solutions,
        save_results=not args.no_save_results,
        continue_from_solution=args.continue_from_solution,
        continue_from_world_model=args.continue_from_world_model,
        enable_world_model=args.world_model,
        wm_stagnation_window=args.wm_stagnation_window,
        wm_max_difficulty=args.wm_max_difficulty,
        artifacts_dir=args.artifacts_dir,
        enable_wandb=args.wandb,
        wandb_project=args.wandb_project,
        run_name=args.run_name,
    )


if __name__ == "__main__":
    main()
