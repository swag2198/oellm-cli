# Evaluating Megatron-LM Checkpoints

## Overview

`oellm schedule-eval` can evaluate Megatron-LM distributed checkpoints directly (no `.safetensors` conversion step) by delegating to [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness)'s built-in `megatron_lm` model backend. Evaluation produces the same lm-eval JSONs as the HF flow and plugs into the existing `oellm collect-results` pipeline.

This document explains the user-facing API, the internal architecture changes, and the design trade-offs that were made. It assumes you already know the basic `oellm` workflow — if not, read [`README.md`](../README.md) and [`docs/VENV.md`](VENV.md) first.

## Quick start

```bash
# Run on Leonardo with a Megatron-LM dist-checkpoint
export CKPT_DIR=/leonardo_work/OELLM_prod2026/users/shaldar0/oellm-autoexp/results/moe_bsz256_lr_sparsity_grid_A130M_120BT/moe_abl_nexp_8_lr0.001_gbsz256_seed1234_decay120BT/checkpoints
export CKPT_STEP=114441
export VOCAB_FILE=/leonardo_work/OELLM_prod2026/models/EleutherAI/gpt-neox-20b/vocab.json
export MERGE_FILE=/leonardo_work/OELLM_prod2026/models/EleutherAI/gpt-neox-20b/merges.txt

oellm schedule-eval \
  --model_type megatron_lm \
  --model_args "load=${CKPT_DIR},ckpt_step=${CKPT_STEP},ckpt_format=torch_dist,vocab_file=${VOCAB_FILE},merge_file=${MERGE_FILE},tokenizer_type=GPT2BPETokenizer" \
  --task_groups "open-sci-0.01" \
  --slurm_template_var '{"ACCOUNT":"OELLM_prod2026","TIME":"02:00:00"}' \
  --venv_path /leonardo_work/OELLM_prod2026/users/shaldar0/oellm-cli/.venv-megatron \
  --modules "gcc/12.2.0,cuda/12.6"
```

Collect the results once the Slurm array finishes:

```bash
OUT=/leonardo_work/OELLM_prod2026/users/shaldar0/oellm-evals/outputs/<timestamp>
oellm collect-results --results_dir "$OUT" --output_csv "$OUT/eval_results.csv"
```

The resulting CSV uses a short, human-readable model id derived from the checkpoint path (e.g. `moe_abl_nexp_8_lr0.001_gbsz256_seed1234_decay120BT__step114441`) rather than lm-eval's internal random token — see [Friendly id for results](#friendly-id-for-results).

## New CLI flags on `oellm schedule-eval`

| Flag | Type | Purpose |
|---|---|---|
| `--model_type` | `str`, default `hf` | lm-eval `--model` backend. `hf` (default) preserves the existing HF flow; `megatron_lm` enables the new path. |
| `--model_args` | `str`, optional | Fully-resolved lm-eval `--model_args` string (e.g. `load=...,ckpt_step=...,vocab_file=...,...`). Only valid when `model_type != "hf"`. For HF, the args are still built automatically from the model path and this flag must not be set. |
| `--modules` | `str`, optional | Comma-separated list of environment modules to `module load` inside each job, e.g. `"gcc/12.2.0,cuda/12.6"`. Applied only when `--venv_path` is used (containers don't have `module`). Generic — not Megatron-specific. |

### Validation rules

When `--model_type megatron_lm` is used, `schedule-eval` enforces:

- `--model_args` **required**: lm-eval's `megatron_lm` backend has mandatory fields (`load`, `ckpt_format`, tokenizer bits) that have no sensible defaults.
- `--venv_path` **required**: Megatron-LM expects `module load`able GCC/CUDA and the custom dist-checkpoint stack, neither of which fit cleanly in the shared Singularity image. We don't try to host them there.
- `--models` **forbidden**: the checkpoint lives entirely inside `--model_args`, and mixing conflicts with `--models`' CSV-split semantics.
- `--eval_csv_path` **forbidden**: the CSV schema today doesn't carry `model_type`/`model_args`. Heterogeneous batches go through repeated `schedule-eval` invocations.
- `--tasks` or `--task_groups` **required**: obvious, but checked early for a clean error message.

When `--model_type hf` (the default), `--model_args` must be unset — HF model args are still auto-built from the model path to preserve prior behavior and avoid two ways to say the same thing.

## Architecture

### `EvaluationJob` dataclass

Two new fields, both with sensible defaults so HF rows don't have to touch them:

```python
@dataclass
class EvaluationJob:
    model_path: Path | str
    task_path: str
    n_shot: int
    eval_suite: str
    model_type: str = "hf"
    model_args: str = ""
```

- `model_type` picks the lm-eval backend (`hf`, `megatron_lm`, ...).
- `model_args` holds the **fully-resolved** lm-eval `--model_args` string for that row. For HF rows it stays empty and the sbatch builds `pretrained=$model_path,trust_remote_code=True` at runtime. For Megatron rows it's copied verbatim from `--model_args`.

For Megatron rows, `model_path` carries a short *display id* derived from the args string (parent directory of `load=` + `step<ckpt_step>`). This id is what shows up in logs, `jobs.tsv`, and the collected CSV — see [Friendly id for results](#friendly-id-for-results).

### Jobs file: `jobs.tsv` (was `jobs.csv`)

The worker-facing jobs manifest is now **tab-separated** and lives at `<evals_dir>/jobs.tsv` with six columns in a fixed order:

```
model_path  task_path  n_shot  eval_suite  model_type  model_args
```

Why TSV:

- Megatron's `model_args` string unavoidably contains commas (`load=...,ckpt_step=...,...`). Pandas quotes comma-containing CSV fields correctly, but the sbatch row parser used `IFS=,` and didn't handle quoted CSV. Bumping to TSV is the minimal fix and removes a latent bug that would also have bitten anyone writing `revision=...` HF args via `eval_csv_path`.
- The file is generated by `oellm` and only consumed by the sbatch in the same run, so there are no external backwards-compat concerns. `oellm collect-results --check` reads `jobs.tsv` when present and still falls back to legacy `jobs.csv` for older runs.

### Sbatch template (`oellm/resources/template.sbatch`)

Three focused edits, each scoped narrowly enough not to change the HF/lighteval/evalchemy code paths:

1. **TSV reader**. The per-row loop now reads six tab-separated fields:

   ```bash
   while IFS=$'\t' read -r model_path task_path n_shot eval_suite model_type model_args
   ```

2. **Generic `modules` loop** inside `run_python`, gated on `VENV_PATH`:

   ```bash
   MODULES_TO_LOAD="{modules}"
   run_python() {
       if [ -n "$VENV_PATH" ]; then
           source "$VENV_PATH/bin/activate"
           if [ -n "$MODULES_TO_LOAD" ]; then
               IFS=',' read -ra MODS <<< "$MODULES_TO_LOAD"
               for mod in "${MODS[@]}"; do
                   mod="$(echo "$mod" | tr -d '[:space:]')"
                   [ -z "$mod" ] && continue
                   module load "$mod"
               done
           fi
           python "$@"
       else
           singularity exec ...
       fi
   }
   ```

   `module` isn't available inside our Singularity images, so the modules loop is a no-op in container mode. Lives in `run_python` rather than at the top of the sbatch so the venv is activated first — module `LD_LIBRARY_PATH` edits win over the venv's.

3. **Unified `lm_eval` case** that dispatches on `$model_type`:

   ```bash
   case "$model_type_normalized" in
       megatron_lm|megatron-lm)
           export MEGATRON_PATH="${MEGATRON_PATH:-/leonardo_work/.../Megatron-LM}"
           export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
           # per-task unique MASTER_PORT — see "Port collisions" below
           if [ -z "${MASTER_PORT:-}" ]; then
               _array_job_id="${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-0}}"
               _array_task_id="${SLURM_ARRAY_TASK_ID:-0}"
               export MASTER_PORT=$(( 20000 + (_array_job_id + _array_task_id * 37) % 20000 ))
           fi
           LM_EVAL_MODEL="$model_type_normalized"
           LM_EVAL_MODEL_ARGS="$model_args"
           LM_EVAL_EXTRA_ARGS+=(--batch_size 1)
           ;;
       hf|"")
           LM_EVAL_MODEL="hf"
           LM_EVAL_MODEL_ARGS="pretrained=$model_path,trust_remote_code=True"
           ;;
       *)
           LM_EVAL_MODEL="$model_type_normalized"
           LM_EVAL_MODEL_ARGS="$model_args"
           ;;
   esac

   run_python -m lm_eval --model "$LM_EVAL_MODEL" \
       --model_args "$LM_EVAL_MODEL_ARGS" \
       --tasks "$task_path" \
       --num_fewshot "$n_shot" \
       --output_path "<evals_dir>/$(openssl rand -hex 5).json" \
       --trust_remote_code \
       ${LM_EVAL_INCLUDE_PATH:+--include_path $LM_EVAL_INCLUDE_PATH} \
       ${LIMIT:+--limit $LIMIT} \
       "${LM_EVAL_EXTRA_ARGS[@]}"
   ```

   The `case "$suite_normalized" in lm_eval|lm-eval|lm-eval-harness)` header is unchanged — Megatron jobs still report `eval_suite=lm_eval` because they are lm-eval runs, just with a non-HF backend. The lighteval and evalchemy branches are untouched.

### Port collisions (`MASTER_PORT`)

Each array task requests `--gres=gpu:1`, and Leonardo's Booster nodes carry 4 A100s, so Slurm packs up to four array tasks per node. lm-eval's `megatron_lm` backend calls `torch.distributed.init_process_group`, which `bind()`s a TCPStore. With a single hardcoded `MASTER_PORT=29500`, co-located tasks raced for the same port; the first one bound and the rest died with:

```
torch.distributed.DistNetworkError: The server socket has failed to listen on
any local network address. port: 29500, ..., EADDRINUSE, address already in use
```

We observed this pattern clearly — in a 12-task array, tasks that landed alone on a node succeeded and tasks that co-located failed in proportion to the pack size on their node.

The fix derives a deterministic per-task port:

```bash
export MASTER_PORT=$(( 20000 + (SLURM_ARRAY_JOB_ID + SLURM_ARRAY_TASK_ID * 37) % 20000 ))
```

Properties:

- Range 20000–39999 — well above privileged ports, below the common ephemeral range (`/proc/sys/net/ipv4/ip_local_port_range` typically 32768–60999, but collisions there are transient and rare enough in practice).
- Indices 0..3 within one job map to ports that differ by ≥37, so four co-located tasks never collide.
- Deterministic per `(array_job_id, array_task_id)` pair → reproducible, easy to log/debug.
- An explicit `MASTER_PORT=<N>` export on the submitter side is respected (the formula is only applied when `MASTER_PORT` is unset).
- Falls back to `SLURM_JOB_ID` (non-array runs) and `0` when the variables are completely absent, so the script also works outside Slurm.

A kernel-assigned ephemeral port via `python -c 'import socket; s=socket.socket(); s.bind(("", 0)); print(s.getsockname()[1])'` would be collision-free by construction, but introduces a (small) TOCTOU window and makes logs non-reproducible. We chose the deterministic formula as the simplest good-enough option and can upgrade later if needed.

### Friendly id for results

lm-eval's `megatron_lm` backend writes a **random 8‑char token** to `model_name` in the result JSON (e.g. `vdujuxrm`) and stores the real arguments under `config.model_args` as a **dict**:

```json
{
  "model_source": "megatron_lm",
  "model_name": "vdujuxrm",
  "config": {
    "model": "megatron_lm",
    "model_args": {
      "load": "/.../moe_abl_nexp_8_..._decay120BT/checkpoints",
      "ckpt_step": 114441,
      "ckpt_format": "torch_dist",
      ...
    }
  }
}
```

This is different from the HF backend, where `model_name` is the pretrained path. To keep `eval_results.csv` readable and to let `collect-results --check` match completions against `jobs.tsv`, `collect-results` now detects Megatron results via `model_source == "megatron_lm"` (or `config.model` starting with `megatron`), pulls `config.model_args`, and collapses it with `_parse_megatron_model_id` into:

```
<run_dir_basename>__step<ckpt_step>
```

For example: `moe_abl_nexp_8_lr0.001_gbsz256_seed1234_decay120BT__step114441`. The same helper runs at schedule time to populate the `model_path` column in `jobs.tsv`, so both sides use the exact same id. The helper accepts either the CLI comma-string (used at schedule time) or the result-JSON dict (used at collect time).

A string-based fallback (`"load=" in model_name and "ckpt_step=" in model_name`) remains as defensive handling in case a future lm-eval version inlines the args string into `model_name`.

## Design decisions

### Why `model_type`/`model_args` and not a single `--model` flag

An earlier draft used a single `--model` flag that would take either a backend name (`megatron_lm`) or an HF path. It was rejected because:

- The existing CLI already has `--models` (plural) for HF paths. A new `--model` (singular) with drastically different semantics is a usability trap.
- `model_type` is orthogonal (which backend?) while `model_args` is orthogonal (what does the backend need?). Splitting them makes the validation rules and the sbatch dispatch trivial.
- A generic `--modules` flag fell out naturally and is useful beyond Megatron (anyone using `--venv_path` can now load arbitrary environment modules).

### Why only one Megatron checkpoint per invocation

The HF flow uses `--models "a,b,c"` + a shared task list to cross-product jobs. Reusing that for Megatron would require either (a) treating `--models` as a list of checkpoint paths and assuming a uniform template with `{ckpt_step}` etc. placeholders, or (b) accepting a JSON list of per-model args strings.

Both options got complicated fast (each Megatron run in the real autoexp grids has a different `ckpt_step`), and supporting multiple Megatron models adds no value over just calling `oellm schedule-eval` several times. So the constraint is: **one `--model_args` per invocation**. If you need to evaluate N Megatron checkpoints, you submit N times. Because each invocation is cheap and independent, this is strictly simpler than the alternatives.

### Why TSV instead of CSV

Pandas' `to_csv` will happily quote the comma-containing `model_args` field, but the bash row parser in the sbatch template was `while IFS=, read ...`, which doesn't understand CSV quoting. Options considered:

- Switch bash to real CSV parsing (`csvkit`, `miller`, `python -c`): heavier runtime dependency, brittle under `set -e`.
- Encode the `model_args` field (e.g. base64 or replace `,`): leaks encoding into a file that users inspect.
- **Switch the manifest to TSV**: zero extra dependencies, one-line change in both the writer and the reader, and solves a latent bug that also affects HF args with commas in `--eval_csv_path`.

The TSV choice also makes the file trivially greppable (`awk -F'\t' '$5=="megatron_lm"'` etc.) which is useful when debugging.

### Why container mode isn't supported for Megatron

Megatron-LM depends on a specific GCC/CUDA toolchain and on custom modules (`MEGATRON_PATH`) that are cleanest to set up via `module load` + pip-installed torch in a venv. The shared Singularity image isn't built for it and `module` isn't available inside Apptainer anyway. Rather than half-support it, `--model_type megatron_lm` hard-requires `--venv_path` with a clear error message. This mirrors the existing `evalchemy` constraint.

### Why `eval_suite=lm_eval` for Megatron rows

`eval_suite` historically selects the harness that runs the eval (lm-eval, lighteval, evalchemy). Megatron runs are still lm-eval runs — we just pass a different `--model` backend to it. So the suite stays `lm_eval` and the per-row dispatch happens via the inner `model_type` switch. Keeping the two orthogonal avoids an explosion of `{suite × backend}` cases.

## `jobs.tsv` schema

Written to `<evals_dir>/jobs.tsv`; tab-separated; one row per (model, task, n_shot) triple:

| column | HF row | Megatron row |
|---|---|---|
| `model_path` | hub id or local dir, e.g. `EleutherAI/pythia-160m` | friendly id, e.g. `moe_abl_nexp_8_..._decay120BT__step114441` |
| `task_path` | lm-eval task name or YAML path | same |
| `n_shot` | `int` | same |
| `eval_suite` | `lm_eval` / `lighteval` / `evalchemy` | `lm_eval` |
| `model_type` | `hf` | `megatron_lm` |
| `model_args` | *(empty)* | the full `load=...,ckpt_step=...,...` string |

`collect-results --check` reads `jobs.tsv` when present and falls back to `jobs.csv` for runs scheduled before this change.

## Environment variables and defaults

The sbatch sets Megatron-only defaults inside the `megatron_lm` case (not at the top of the script), so other suites are unaffected. All three respect an override from the submitter's environment:

| Variable | Default | How to override |
|---|---|---|
| `MEGATRON_PATH` | `/leonardo_work/OELLM_prod2026/users/shaldar0/oellm-autoexp/submodules/Megatron-LM` | export before `oellm schedule-eval`, or pass via `--slurm_template_var` |
| `MASTER_ADDR` | `127.0.0.1` | export before `oellm schedule-eval` |
| `MASTER_PORT` | deterministic per array task (see above) | export an explicit `MASTER_PORT=...` to pin it |

Caller-supplied modules are passed through `--modules "gcc/12.2.0,cuda/12.6"` and loaded after the venv is activated.

## Re-running failed evaluations

`oellm collect-results --check` diffs `jobs.tsv` against the produced JSONs and writes a `_missing.csv` with the unfinished rows. For HF flows you can feed that straight back into `--eval_csv_path`. For Megatron flows, the `--eval_csv_path` schema currently doesn't carry `model_type`/`model_args`, so the simplest recipe is to re-invoke `schedule-eval` with only the missing tasks listed in `--tasks`:

```bash
oellm schedule-eval \
  --model_type megatron_lm \
  --model_args "$MODEL_ARGS" \
  --tasks "copa,arc_easy,hellaswag,lambada_openai"  # whichever failed \
  --n_shot 0 \
  --venv_path .venv-megatron \
  --modules "gcc/12.2.0,cuda/12.6" \
  --slurm_template_var '{"ACCOUNT":"OELLM_prod2026","TIME":"02:00:00"}'
```

Extending `--eval_csv_path` to natively carry `model_type`/`model_args` columns is a small follow-up if needed.

## Files changed

| File | Change |
|---|---|
| `oellm/main.py` | `EvaluationJob` gains `model_type`/`model_args`; `schedule_evals` gains `--model_type`/`--model_args`/`--modules` + validation; Megatron job-construction branch; `_parse_megatron_model_id` helper accepting either a string or a dict; HF-only pathways guarded on `model_type == "hf"`; jobs file written as `jobs.tsv`; `collect_results` reads `jobs.tsv` (falls back to `jobs.csv`) and normalizes Megatron `model_name` via `config.model_args`. |
| `oellm/resources/template.sbatch` | `IFS=$'\t'` row parser with two new columns; generic `modules` loop inside `run_python`; unified `lm_eval` case dispatching on `$model_type`; Megatron-only `MEGATRON_PATH`/`MASTER_ADDR`/`MASTER_PORT` defaults with per-task port derivation; `--batch_size 1` for Megatron only. |
| `docs/MEGATRON.md` | This document. |
