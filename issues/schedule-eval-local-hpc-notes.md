# `schedule-eval --local` on HPC: discussion summary

This note captures the investigation around running `oellm schedule-eval` with `--local` on Leonardo (and similar clusters): how it behaves, what broke, what we changed in code, and what we fixed operationally (PyTorch / environment).

---

## 1. What `--local` does (no code mystery)

- **`--local true`** does **not** call `sbatch`. It writes the same generated script as SLURM jobs (`submit_evals.sbatch`), then runs it with **`bash`**, with fake SLURM variables (`SLURM_ARRAY_TASK_ID=0`, etc.).
- Work runs on **whatever machine invoked `oellm`** (login node, interactive GPU node, laptop). The tool does **not** require a “compute node” by itself—you need whatever resources the eval needs (e.g. a GPU if you want GPU eval).
- **`--limit N`** is passed to **`lm_eval` as `--limit`** (and to lighteval as `--max-samples`): it caps **examples per task**, useful for smoke tests. It is not about job count.

Relevant implementation: `schedule_evals()` in `oellm/main.py` (local branch runs `subprocess.run(["bash", str(sbatch_script_path)], ...)` instead of `sbatch`).

---

## 2. Eval dependencies (venv)

For local runs you need a venv with the eval stack, e.g.:

```bash
uv pip install lm-eval torch transformers accelerate "datasets<4.0.0"
```

- **`datasets<4.0.0`** is a real dependency for Hugging Face–based tasks, not only for resolver conflicts.
- Use **`uv pip install`** from the **repository root** where `.venv` lives, or pass **`--python /absolute/path/to/oellm-cli/.venv/bin/python`** so `uv` resolves the correct environment.

---

## 3. Problem A: Hub / cache on air-gapped compute nodes

### Symptoms

- Errors such as **`Network is unreachable`** when loading tokenizer/model from the Hub, or **`OSError: Can't load tokenizer`**, even though models were already cached under a shared **`HF_HOME`**.

### Why SLURM jobs worked but `--local` failed

| Aspect | Non-local (`sbatch`) | `--local` (before fixes) |
|--------|----------------------|---------------------------|
| **Cluster env** | `_load_cluster_env()` loads `clusters.yaml` and sets paths (e.g. `EVAL_*`, and often your **`HF_HOME`** via environment/modules). | **`_load_cluster_env()` is skipped.** |
| **`HF_HOME` default** | Usually set by cluster setup. | **`setdefault("HF_HOME", ~/.cache/huggingface)`** if unset—often **not** your shared 1 TB cache. |
| **`HF_HUB_OFFLINE` in script** | Template embedded **`hf_hub_offline=1`** → **offline** Hub use on workers. | Template embedded **`hf_hub_offline=0`** for local → tries **online** Hub access. |
| **Export `HF_HUB_OFFLINE=1` before `oellm`** | N/A for generation in the same way; batch script still got **1** from template. | **Insufficient by itself**: the generated script contained **`export HF_HUB_OFFLINE=0`**, which **overwrote** the parent environment when `bash` ran the script. |

So: **`export HF_HOME=...`** before `oellm` fixed wrong cache **roots** (because `setdefault` does not override an already-set variable). **`export HF_HUB_OFFLINE=1`** alone did **not** fix offline behavior until the code respected it when **generating** the script.

### Code change: `_resolve_hf_hub_offline()`

**File:** `oellm/main.py`

A helper **`_resolve_hf_hub_offline(local: bool) -> int`** was added. It decides the integer baked into the template as **`HF_HUB_OFFLINE`**:

- If **`HF_HUB_OFFLINE`** is set in the environment when **`oellm` runs** (non-empty), that value is parsed as an integer and used.
- Otherwise: **`0`** for `--local` (typical laptop dev, allow Hub downloads) and **`1`** for SLURM (air-gapped workers).

The template still contains:

```bash
export HF_HUB_OFFLINE={hf_hub_offline}
```

but **`{hf_hub_offline}`** is now filled from **`_resolve_hf_hub_offline(local)`** instead of a hardcoded **`0 if local else 1`** that ignored a user export.

**Documentation:** `README.md` under **Running Locally (without SLURM)** documents exporting **`HF_HOME`** and **`HF_HUB_OFFLINE=1`** on air-gapped nodes before `--local`.

---

## 4. Problem B: PyTorch CUDA build vs NVIDIA driver

### Symptoms

- **`RuntimeError: The NVIDIA driver on your system is too old (found version 12020)`** (driver API compatible with **CUDA 12.x**), while PyTorch was installed as **`torch==2.11.0`** with the **CUDA 13** wheel stack (**`+cu130`**, `nvidia-cudnn-cu13`, etc.).

That combination is invalid: the **driver** on the node must support the **CUDA version** your **PyTorch binaries** expect.

### Fix (operational, not an `oellm` code change)

On a **CUDA 12.2** driver (e.g. **NVIDIA-SMI** reporting **CUDA Version: 12.2**), reinstall **torch** from the **cu12** PyTorch index, e.g.:

```bash
cd /path/to/oellm-cli   # directory that contains .venv

uv pip install --python .venv/bin/python \
  --upgrade --force-reinstall \
  torch \
  --index-url https://download.pytorch.org/whl/cu124
```

- **`torchvision` / `torchaudio`** are optional for typical text **`lm_eval`** workflows; **`torch`** alone is enough unless you need those libraries.
- **`lm-eval`**, **`transformers`**, **`accelerate`**, **`datasets`** usually stay as-is; only **torch** (and its NVIDIA dependency packages) were replaced.

### Pitfall: wrong working directory

If **`uv`** reports no venv for **`.venv/bin/python`**, the shell is not the repo root (or the path is wrong). **`cd`** to **`oellm-cli`** or use the **absolute** path to **`.venv/bin/python`**.

### Verification

On a **GPU** node:

```bash
.venv/bin/python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

Expect something like **`2.6.0+cu124`**, **`True`**, and the GPU name, with **no** “driver is too old” error.

---

## 5. Successful smoke test (reference)

With correct **`HF_HOME`**, offline flag when needed, matching **torch `+cu124`**, and **`--venv_path .venv --local true --limit 1`**, a minimal run can:

- Load **`EleutherAI/pythia-160m`** from cache,
- Use **`cuda:0`**,
- Use cached **`openai/gsm8k`** with **offline mode** enabled in logs,
- Finish **`lm_eval`** and write aggregated results under the run’s output directory (see CLI logs for **`EVAL_OUTPUT_DIR`** / **`oellm-output`** layout).

---

## 6. Files touched in the repository (HF Hub offline behavior)

| File | Change |
|------|--------|
| `oellm/main.py` | Added **`_resolve_hf_hub_offline()`**; **`sbatch_template.format(..., hf_hub_offline=_resolve_hf_hub_offline(local))`**. |
| `README.md` | Air-gapped **`HF_HOME` / `HF_HUB_OFFLINE`** example for **`--local`**. |

The eval template **`oellm/resources/template.sbatch`** still exports **`HF_HOME`**, **`HF_DATASETS_CACHE`**, and **`HF_HUB_OFFLINE`**; only the **numeric value** for **`HF_HUB_OFFLINE`** for local runs is now derived as above.

---

## 7. Related internal doc

Social IQA / parquet / **`HF_HUB_OFFLINE`** context is described separately in **`siqa_fix.md`** (repository root).
