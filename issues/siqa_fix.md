# Social IQA and lm_eval offline support in oellm-cli

This document describes why Social IQA needed special handling, what the code does, and how the pieces fit together. It reflects the work discussed when fixing `schedule-eval` and Slurm jobs for `social_iqa` on HPC systems (e.g. Leonardo) with `HF_HUB_OFFLINE=1` on compute nodes.

---

## Is `custom_lm_eval_tasks` under each run directory deliberate?

**Yes.** When you do not pass a custom `lm_eval_include_path`, `schedule_evals` calls `_materialize_lm_eval_include_path`, which:

1. Creates **`{run_dir}/custom_lm_eval_tasks/`** (where `run_dir` is the timestamped folder under `EVAL_OUTPUT_DIR`, e.g. `.../outputs/2026-04-05-12-52-54/`).
2. **Copies** every `*.yaml` / `*.yml` from the bundled `oellm/resources/custom_lm_eval_tasks/` into that directory.
3. **Patches** `social_iqa.yaml` in place (see below) so `dataset_kwargs.data_files` points at **local parquet paths** under your HF cache.

**Why not use the checkout path directly?**

- The Slurm/Singularity template typically **bind-mounts** paths under something like `EVAL_BASE_DIR` (and `HF_HOME`, etc.).
- The oellm-cli **install path** may lie **outside** those mounts, so the container cannot see `.../oellm-cli/oellm/resources/custom_lm_eval_tasks/`.
- Materializing under **`EVAL_OUTPUT_DIR`** (which is under the eval workspace) guarantees `--include_path` points at a directory the job can read.

So the folder next to `jobs.csv`, `submit_evals.sbatch`, and `results/` is **expected** and is the canonical copy used for that scheduled run.

---

## What was wrong with stock Social IQA?

### 1. lm_eval’s built-in task and Hugging Face “scripts”

The default lm_eval task name is **`social_iqa`**. Under the hood, many configs resolve to the Hugging Face dataset **`allenai/social_i_qa`**.

That dataset’s **loading script** (not the parquet conversion) downloads a **zip from Google Cloud Storage** (`storage.googleapis.com`). On many HPC login nodes:

- TLS to GCS fails (corporate proxy, incomplete CA bundle, etc.), or
- You want **no** dependency on that URL at schedule or eval time.

So **`load_dataset("allenai/social_i_qa")` via the script** is a fragile path for automation.

### 2. `HF_HUB_OFFLINE=1` on compute nodes

Batch jobs often set **`HF_HUB_OFFLINE=1`** so workers do not call the Hub at runtime. If the task YAML still referenced **remote** Hub URLs for data, **`datasets`** could try to resolve them and fail when offline.

### 3. Task name vs. YAML file path (`--tasks`)

`lm_eval --tasks social_iqa` goes through the **merged task index**. A **built-in** task can **shadow** an override from `--include_path` depending on resolution order.

Passing **`--tasks /absolute/path/to/social_iqa.yaml`** forces TaskManager to load **that** config file, which is reliable for overrides.

---

## Code changes (what and why)

### A. Bundled task YAML: `oellm/resources/custom_lm_eval_tasks/social_iqa.yaml`

- Defines the task in a way aligned with **parquet**-backed loading (and Hub revision `refs/convert/parquet` conceptually).
- At **schedule** time, the copy under the run directory is **rewritten** so `dataset_kwargs.data_files` contains **absolute paths** to parquet shards already in the cache (see patching below).

### B. `_patch_social_iqa_parquet_yaml` in `oellm/main.py`

- Calls **`hf_hub_download`** for the two parquet files (`default/train/0000.parquet`, `default/validation/0000.parquet`) at revision **`refs/convert/parquet`**.
- Writes those **local** paths into the materialized YAML with PyYAML.
- **Why:** So eval uses **only filesystem paths** that exist under `HF_HOME` (or your cache), consistent with **`HF_HUB_OFFLINE=1`** on workers (no Hub fetch for the dataset at run time).

### C. `_materialize_lm_eval_include_path` in `oellm/main.py`

- Copies bundled YAMLs into **`{evals_dir}/custom_lm_eval_tasks`** and runs the Social IQA patch on `social_iqa.yaml`.
- Returns the **string path** passed to the batch template as **`lm_eval_include_path`** (`--include_path` for lm_eval).

### D. `_rewrite_lm_eval_task_paths_to_yaml` in `oellm/main.py`

- After materialization, for **lm_eval** suite rows, if `task_path` is a bare name (e.g. `social_iqa`) and **`{include_dir}/{task}.yaml`** exists, replaces **`task_path`** in the jobs dataframe with the **resolved absolute path** to that YAML.
- **`jobs.csv`** then contains the path; the batch script uses **`--tasks "$task_path"`** without extra shell logic.

**Why:** Avoids built-in shadowing and keeps the Slurm template **generic** (no `LM_EVAL_TASK_ARG` workaround in bash).

### E. `oellm/resources/template.sbatch`

- **`lm_eval`** invocation uses **`--tasks "$task_path"`** as read from the CSV (which may already be an absolute `.yaml` path after rewrite).

### F. `oellm/utils.py`: `_pre_download_social_iqa_parquet_shards` + `_pre_download_datasets_from_specs`

- For dataset spec **`allenai/social_i_qa`**, **do not** call **`load_dataset`** (script → GCS zip).
- Instead, **`hf_hub_download`** the same parquet shards as the patch step (Hub **`refs/convert/parquet`**), optionally respecting **`HF_HOME`** for cache layout.

**Why:** `schedule-eval` runs **dataset pre-download** when checks are enabled. The registry in `task-groups.yaml` maps task names like `social_iqa` → **`allenai/social_i_qa`**. Without this branch, pre-download would trigger the **GCS** path and fail on the same SSL issues as before. Parquet pre-download aligns with how eval actually loads data.

### G. `oellm/resources/task-groups.yaml`

- Entries that list **`dataset: allenai/social_i_qa`** drive that pre-download spec.
- If you **remove** `social_iqa` from the registry, pre-download **skips** it (workaround) but you lose the explicit “cache this dataset” step unless you rely on the parquet branch or manual cache population.

---

## Operational summary

| Concern | Mitigation |
|--------|------------|
| GCS zip / SSL on login node | Parquet via Hub in `utils.py` pre-download + YAML patch in `main.py` |
| Offline eval on GPU nodes | Materialized YAML with **local** `data_files` paths |
| Built-in task shadows override | **`jobs.csv` stores absolute path** to `social_iqa.yaml` |
| Container cannot see repo YAMLs | **Copy** to `{run}/custom_lm_eval_tasks` under `EVAL_OUTPUT_DIR` |

---

## References in the codebase

- `oellm/main.py`: `_patch_social_iqa_parquet_yaml`, `_materialize_lm_eval_include_path`, `_rewrite_lm_eval_task_paths_to_yaml`, `schedule_evals` ordering (materialize → rewrite → shuffle → `jobs.csv`).
- `oellm/utils.py`: `_pre_download_social_iqa_parquet_shards`, special case in `_pre_download_datasets_from_specs`.
- `oellm/resources/custom_lm_eval_tasks/social_iqa.yaml`: bundled task definition.
- `oellm/resources/template.sbatch`: `lm_eval` case, `--tasks` from CSV.
- `oellm/task_groups.py` / `oellm/resources/task-groups.yaml`: task → dataset mapping for pre-download.

---

## Chat context (condensed)

The issues surfaced when running `oellm schedule-eval` with **`--tasks social_iqa`**: pre-download used **`load_dataset("allenai/social_i_qa")`**, which hit **GCS** and failed with **SSL certificate** errors on the cluster. Separately, offline evals required **local parquet paths** in the task YAML, and **Singularity** needed task files under **mounted** paths. The solution combined **Hub parquet** for caching and YAML patching, **CSV rewriting** so `lm_eval` loads the override file by path, and an optional **utils** special case so **pre-download** does not use the dataset script. Commenting out Social IQA in `task-groups.yaml` “fixed” scheduling only by **disabling** that pre-download lookup—not by fixing the underlying GCS path.
