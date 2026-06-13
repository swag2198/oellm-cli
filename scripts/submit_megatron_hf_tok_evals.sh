#!/usr/bin/env bash
# Submit `oellm schedule-eval` for Megatron-LM dist-checkpoints that use a
# HuggingFace tokenizer (instead of the legacy GPT2BPETokenizer with
# vocab.json + merges.txt).
#
# Differs from submit_megatron_evals.sh only in tokenizer-related model_args:
#     tokenizer_type=HuggingFaceTokenizer
#     tokenizer_model=<HF tokenizer directory>   # passed to AutoTokenizer
# No vocab_file / merge_file are needed.
#
# One oellm submission per (CKPT_DIR, CKPT_STEP) pair (see docs/MEGATRON.md
# for why one checkpoint per invocation is the intended pattern).
#
# Usage:
#   bash scripts/submit_megatron_hf_tok_evals.sh              # submit all
#   DRY_RUN=1 bash scripts/submit_megatron_hf_tok_evals.sh    # preview, submit nothing
#   TASK_GROUPS=open-sci-0.01 TIME=04:00:00 bash scripts/submit_megatron_hf_tok_evals.sh
#   TOKENIZER_MODEL=/path/to/another-hf-tokenizer bash scripts/submit_megatron_hf_tok_evals.sh
set -euo pipefail

# --- shared (overridable) config -------------------------------------------
# Exported vars propagate to the SLURM job via sbatch's default --export=ALL
# and are consumed by oellm/resources/template.sbatch (see docs/MEGATRON.md).
export MEGATRON_PATH="${MEGATRON_PATH:-/leonardo_work/OELLM_prod2026/users/shaldar0/oellm-autoexp/submodules/Megatron-LM}"
export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"

# HF tokenizer to pair with the checkpoints below. `tokenizer_model` should
# point to a HuggingFace tokenizer *directory* (contains tokenizer_config.json,
# tokenizer.model / tokenizer.json, special_tokens_map.json, ...). lm-eval's
# megatron_lm backend loads it via AutoTokenizer.from_pretrained.
TOKENIZER_TYPE="${TOKENIZER_TYPE:-HuggingFaceTokenizer}"
TOKENIZER_MODEL="${TOKENIZER_MODEL:-/leonardo_work/OELLM_prod2026/users/shaldar0/tokenizers/openeurollm-tokenizer-256k}"

VENV_PATH="${VENV_PATH:-/leonardo_work/OELLM_prod2026/users/shaldar0/oellm-cli/.venv-megatron}"
TASK_GROUPS="${TASK_GROUPS:-open-sci-0.01}"
ACCOUNT="${ACCOUNT:-OELLM_prod2026}"
TIME="${TIME:-06:00:00}"
MODULES="${MODULES:-gcc/12.2.0,cuda/12.6}"
DRY_RUN="${DRY_RUN:-0}"

# --- checkpoints to evaluate -----------------------------------------------
# "CKPT_DIR CKPT_STEP" per line; blank lines / lines starting with '#' are ignored.
# CKPT_DIR is the parent `checkpoints/` directory that contains iter_NNNNN
# subdirectories (NOT the iter_NNNNN dir itself).
CHECKPOINTS=$(cat <<'EOF'
# Multilingual scaling: dense Qwen3 0.9B with HF (openeurollm-256k) tokenizer
/leonardo_work/OELLM_prod2026/experiments/multilingual_scaling/0.9B_ne/training/qwen3_dense_0.9B_ne_lr0.001_gbsz32_stable12BT/checkpoints 16000
EOF
)

# --- submit loop -----------------------------------------------------------
n_total=0
n_submitted=0
n_skipped=0

while IFS= read -r line; do
    # strip comments and trim whitespace
    line="${line%%#*}"
    line="$(echo "$line" | awk '{$1=$1};1')"
    [ -z "$line" ] && continue

    read -r CKPT_DIR CKPT_STEP <<<"$line"
    n_total=$((n_total + 1))

    if [ ! -d "$CKPT_DIR" ]; then
        echo "[skip] $CKPT_DIR does not exist" >&2
        n_skipped=$((n_skipped + 1))
        continue
    fi
    if [ ! -d "$CKPT_DIR/iter_$(printf '%07d' "$CKPT_STEP")" ]; then
        echo "[warn] $CKPT_DIR has no iter_$(printf '%07d' "$CKPT_STEP")/ subdir; proceeding anyway" >&2
    fi

    MODEL_ARGS="load=${CKPT_DIR},ckpt_step=${CKPT_STEP},ckpt_format=torch_dist,tokenizer_type=${TOKENIZER_TYPE},tokenizer_model=${TOKENIZER_MODEL}"

    echo "======================================================================"
    echo "[$((n_submitted + 1))] $(basename "$(dirname "$CKPT_DIR")") @ step=${CKPT_STEP}"
    echo "    CKPT_DIR=$CKPT_DIR"
    echo "    TOKENIZER_TYPE=$TOKENIZER_TYPE"
    echo "    TOKENIZER_MODEL=$TOKENIZER_MODEL"
    echo "    MEGATRON_PATH=$MEGATRON_PATH"
    echo "    MASTER_ADDR=$MASTER_ADDR"
    echo "======================================================================"

    if [ "$DRY_RUN" = "1" ]; then
        printf '    [dry-run] would run:\n      oellm schedule-eval \\\n        --model_type megatron_lm \\\n        --model_args "%s" \\\n        --task_groups "%s" \\\n        --slurm_template_var %s \\\n        --venv_path %s \\\n        --modules "%s"\n' \
            "$MODEL_ARGS" "$TASK_GROUPS" "'{\"ACCOUNT\":\"${ACCOUNT}\",\"TIME\":\"${TIME}\"}'" "$VENV_PATH" "$MODULES"
    else
        oellm schedule-eval \
            --model_type megatron_lm \
            --model_args "${MODEL_ARGS}" \
            --task_groups "${TASK_GROUPS}" \
            --slurm_template_var "{\"ACCOUNT\":\"${ACCOUNT}\",\"TIME\":\"${TIME}\"}" \
            --venv_path "${VENV_PATH}" \
            --modules "${MODULES}"
    fi

    n_submitted=$((n_submitted + 1))
done <<<"$CHECKPOINTS"

echo
echo "Done. total=${n_total}  submitted=${n_submitted}  skipped(missing dir)=${n_skipped}"
