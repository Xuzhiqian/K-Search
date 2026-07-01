# K-Search
<!-- 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 -->
K-Search is an LLM-driven kernel engineering loop for Armv8/AArch64 CPU kernels. It generates C/C++ implementations, evaluates them with a task-provided CPU harness, and iteratively improves them with an optional co-evolving world model.

The current codebase targets pure CPU platforms. CUDA, Triton, FlashInfer, GPUMode, KernelBench, MLX, Metal, GPU benchmark assets, and GPU baseline integrations have been removed.

## What It Keeps

- OpenAI-compatible LLM generation.
- Iterative optimize-and-evaluate loops.
- World-model-guided action selection.
- Solution JSON persistence and world model snapshots.
- Optional Weights & Biases logging.

## Arm CPU Task Format

Pass `--task-path` as either a JSON/YAML file or a directory containing `task.json`.

The evaluator expects the task to provide a C++ `harness_source`. The harness includes `kernel.h`, runs correctness and timing checks, and prints:

```text
KSEARCH_STATUS=passed
KSEARCH_KERNEL_MS=0.123
KSEARCH_REFERENCE_MS=0.456
KSEARCH_LOG=optional details
```

Minimal task shape:

```json
{
  "name": "vector_add_f32",
  "description": "Add two float arrays into an output array.",
  "signature": "extern \"C\" void kernel_entry(const float* a, const float* b, float* out, int n);",
  "constraints": {
    "dtype": "float32",
    "n": 1048576,
    "tolerance": 0.0001
  },
  "allow_threads": false,
  "reference_source": "for (int i = 0; i < n; ++i) out[i] = a[i] + b[i];",
  "harness_source": "// C++ harness that includes kernel.h and prints KSEARCH_* metrics"
}
```

Generated solutions must contain:

- `kernel.h`: C ABI declaration.
- `kernel.cpp`: optimized implementation.
- `main.cpp`: optional helpers/wrappers, without `main()`.

## Run

```bash
TASK_PATH=/path/to/task.json \
LLM_API_KEY=... \
bash scripts/armcpu_wm.sh
```

Useful environment variables:

| Variable | Default | Description |
| --- | --- | --- |
| `MODEL_NAME` | `gpt-5.2` | LLM model |
| `BASE_URL` | `https://api.openai.com/v1` | OpenAI-compatible API base URL |
| `TARGET_CPU` | `armv8-a` | CPU architecture hint |
| `CPU_FEATURES` | `neon` | Comma-separated feature hints, e.g. `neon,sve` |
| `CXX` | `c++` | C++ compiler on the Arm machine |
| `CXXFLAGS` | auto | Optional full compiler flag override |
| `MAX_OPT_ROUNDS` | `50` | Optimization rounds |
| `WM` | `1` | Enable world model |
| `ARTIFACTS_DIR` | `.ksearch-output-armcpu` | Output directory |

CLI equivalent:

```bash
python generate_kernels_and_eval.py \
  --task-source armcpu \
  --task-path /path/to/task.json \
  --model-name gpt-5.2 \
  --api-key "$LLM_API_KEY" \
  --language cpp \
  --target-cpu armv8-a \
  --cpu-features neon \
  --world-model \
  --save-solutions
```

## Local Development Constraint

On lightweight local machines, use only static checks such as:

```bash
python -m py_compile generate_kernels_and_eval.py k_search/**/*.py
python generate_kernels_and_eval.py --help
```

Run generated kernel evaluation on the intended Armv8 CPU server.
