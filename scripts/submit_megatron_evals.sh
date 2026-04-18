#!/usr/bin/env bash
# Submit `oellm schedule-eval` for a list of Megatron-LM dist-checkpoints.
# One oellm submission per (CKPT_DIR, CKPT_STEP) pair (see docs/MEGATRON.md for
# why one checkpoint per invocation is the intended pattern).
#
# Usage:
#   bash scripts/submit_megatron_evals.sh              # submit all
#   DRY_RUN=1 bash scripts/submit_megatron_evals.sh    # print what would run, submit nothing
#   TASK_GROUPS=open-sci-0.01 TIME=04:00:00 bash scripts/submit_megatron_evals.sh
set -euo pipefail

# --- shared (overridable) config -------------------------------------------
# Exported vars propagate to the SLURM job via sbatch's default --export=ALL
# and are consumed by oellm/resources/template.sbatch (see docs/MEGATRON.md).
export VOCAB_FILE="${VOCAB_FILE:-/leonardo_work/OELLM_prod2026/models/EleutherAI/gpt-neox-20b/vocab.json}"
export MERGE_FILE="${MERGE_FILE:-/leonardo_work/OELLM_prod2026/models/EleutherAI/gpt-neox-20b/merges.txt}"
export TOKENIZER_TYPE="${TOKENIZER_TYPE:-GPT2BPETokenizer}"
export MEGATRON_PATH="${MEGATRON_PATH:-/leonardo_work/OELLM_prod2026/users/shaldar0/oellm-autoexp/submodules/Megatron-LM}"
export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"

VENV_PATH="${VENV_PATH:-/leonardo_work/OELLM_prod2026/users/shaldar0/oellm-cli/.venv-megatron}"
TASK_GROUPS="${TASK_GROUPS:-open-sci-0.01}"
ACCOUNT=OELLM_prod2026
TIME="${TIME:-06:00:00}"
MODULES="${MODULES:-gcc/12.2.0,cuda/12.6}"
DRY_RUN="${DRY_RUN:-0}"

# --- checkpoints to evaluate -----------------------------------------------
# "CKPT_DIR CKPT_STEP" per line; blank lines / lines starting with '#' are ignored.
CHECKPOINTS=$(cat <<'EOF'
# MoE without GQA (bsz256, lr0.001, decay120BT)
# /leonardo_work/OELLM_prod2026/users/shaldar0/oellm-autoexp/results/moe_bsz256_lr_sparsity_grid_A130M_120BT/moe_abl_nexp_16_lr0.001_gbsz256_seed1234_decay120BT/checkpoints 114441
# /leonardo_work/OELLM_prod2026/users/shaldar0/oellm-autoexp/results/moe_bsz256_lr_sparsity_grid_A130M_120BT/moe_abl_nexp_32_lr0.001_gbsz256_seed1234_decay120BT/checkpoints 114441
# /leonardo_work/OELLM_prod2026/users/shaldar0/oellm-autoexp/results/moe_bsz256_lr_sparsity_grid_A130M_120BT/moe_abl_nexp_64_lr0.001_gbsz256_seed1234_decay120BT/checkpoints 114441
# /leonardo_work/OELLM_prod2026/users/shaldar0/oellm-autoexp/results/moe_bsz256_lr_sparsity_grid_A130M_120BT/moe_abl_nexp_8_lr0.001_gbsz256_seed1234_decay80BT/checkpoints 76294
# /leonardo_work/OELLM_prod2026/users/shaldar0/oellm-autoexp/results/moe_bsz256_lr_sparsity_grid_A130M_120BT/moe_abl_nexp_16_lr0.001_gbsz256_seed1234_decay80BT/checkpoints 76294
# /leonardo_work/OELLM_prod2026/users/shaldar0/oellm-autoexp/results/moe_bsz256_lr_sparsity_grid_A130M_120BT/moe_abl_nexp_32_lr0.001_gbsz256_seed1234_decay80BT/checkpoints 76294
# /leonardo_work/OELLM_prod2026/users/shaldar0/oellm-autoexp/results/moe_bsz256_lr_sparsity_grid_A130M_120BT/moe_abl_nexp_64_lr0.001_gbsz256_seed1234_decay80BT/checkpoints 76294


# MoE with GQA
# /leonardo_work/OELLM_prod2026/users/shaldar0/oellm-autoexp/results/moe_GQA_bsz256_lr_sparsity_grid_A130M_120BT/moe_GQA_nexp_8_lr0.001_gbsz256_seed1234_decay120BT/checkpoints 114441
# /leonardo_work/OELLM_prod2026/users/shaldar0/oellm-autoexp/results/moe_GQA_bsz256_lr_sparsity_grid_A130M_120BT/moe_GQA_nexp_16_lr0.001_gbsz256_seed1234_decay120BT/checkpoints 114441
# /leonardo_work/OELLM_prod2026/users/shaldar0/oellm-autoexp/results/moe_GQA_bsz256_lr_sparsity_grid_A130M_120BT/moe_GQA_nexp_32_lr0.001_gbsz256_seed1234_decay120BT/checkpoints 114441
# /leonardo_work/OELLM_prod2026/users/shaldar0/oellm-autoexp/results/moe_GQA_bsz256_lr_sparsity_grid_A130M_120BT/moe_GQA_nexp_64_lr0.001_gbsz256_seed1234_decay120BT/checkpoints 114441

# /leonardo_work/OELLM_prod2026/users/shaldar0/oellm-autoexp/results/moe_GQA_bsz256_lr_sparsity_grid_A130M_120BT/moe_GQA_nexp_8_lr0.001_gbsz256_seed1234_decay80BT/checkpoints 76294
# /leonardo_work/OELLM_prod2026/users/shaldar0/oellm-autoexp/results/moe_GQA_bsz256_lr_sparsity_grid_A130M_120BT/moe_GQA_nexp_16_lr0.001_gbsz256_seed1234_decay80BT/checkpoints 76294
# /leonardo_work/OELLM_prod2026/users/shaldar0/oellm-autoexp/results/moe_GQA_bsz256_lr_sparsity_grid_A130M_120BT/moe_GQA_nexp_32_lr0.001_gbsz256_seed1234_decay80BT/checkpoints 76294
# /leonardo_work/OELLM_prod2026/users/shaldar0/oellm-autoexp/results/moe_GQA_bsz256_lr_sparsity_grid_A130M_120BT/moe_GQA_nexp_64_lr0.001_gbsz256_seed1234_decay80BT/checkpoints 76294

# Diana's dense Qwen3 300M baselines with GQA
# /leonardo_work/OELLM_prod2026/users/donutu00/oellm-autoexp/results/qwen3_300M_gpt_neox/dense_qwen3_300M_gpt_neox_lr0.002_gbsz256_beta20.95_decay120BT/checkpoints 114441
# /leonardo_work/OELLM_prod2026/users/donutu00/oellm-autoexp/results/qwen3_300M_gpt_neox/dense_qwen3_300M_gpt_neox_lr0.002_gbsz512_beta20.95_decay200BT/checkpoints 95368
# /leonardo_work/OELLM_prod2026/users/donutu00/oellm-autoexp/results/qwen3_300M_gpt_neox/dense_qwen3_300M_gpt_neox_lr0.002_gbsz512_beta20.95_decay300BT/checkpoints 143052
# /leonardo_work/OELLM_prod2026/users/donutu00/oellm-autoexp/results/qwen3_300M_gpt_neox/dense_qwen3_300M_gpt_neox_lr0.002_gbsz256_beta20.95_decay50BT/checkpoints 47684
# /leonardo_work/OELLM_prod2026/users/donutu00/oellm-autoexp/results/qwen3_300M_gpt_neox/dense_qwen3_300M_gpt_neox_lr0.002_gbsz256_beta20.95_decay80BT/checkpoints 76294


# MoE no GQA replication (just to check random variations)
# /leonardo_work/OELLM_prod2026/users/shaldar0/oellm-autoexp/results/moe_no_GQA_bsz256_lr_sparsity_grid_A130M_120BT/moe_no_GQA_nexp_8_lr0.001_gbsz256_seed1234_decay80BT/checkpoints 76294
# /leonardo_work/OELLM_prod2026/users/shaldar0/oellm-autoexp/results/moe_no_GQA_bsz256_lr_sparsity_grid_A130M_120BT/moe_no_GQA_nexp_8_lr0.001_gbsz256_seed1234_decay120BT/checkpoints 114441
# /leonardo_work/OELLM_prod2026/users/shaldar0/oellm-autoexp/results/moe_no_GQA_bsz256_lr_sparsity_grid_A130M_120BT/moe_no_GQA_nexp_32_lr0.001_gbsz256_seed1234_decay80BT/checkpoints 76294
# /leonardo_work/OELLM_prod2026/users/shaldar0/oellm-autoexp/results/moe_no_GQA_bsz256_lr_sparsity_grid_A130M_120BT/moe_no_GQA_nexp_32_lr0.001_gbsz256_seed1234_decay120BT/checkpoints 114441

# Diana's dense Qwen3 300M baselines without GQA
/leonardo_work/OELLM_prod2026/users/donutu00/oellm-autoexp/results/old_no_gqa_qwen/qwen3_300M_gpt_neox/dense_qwen3_300M_gpt_neox_lr0.002_gbsz256_beta20.95_decay120BT/checkpoints 114441
/leonardo_work/OELLM_prod2026/users/donutu00/oellm-autoexp/results/old_no_gqa_qwen/qwen3_300M_gpt_neox/dense_qwen3_300M_gpt_neox_lr0.002_gbsz512_beta20.95_decay300BT/checkpoints 143052
/leonardo_work/OELLM_prod2026/users/donutu00/oellm-autoexp/results/old_no_gqa_qwen/qwen3_300M_gpt_neox/dense_qwen3_300M_gpt_neox_lr0.002_gbsz512_beta20.95_decay200BT/checkpoints 95368
/leonardo_work/OELLM_prod2026/users/donutu00/oellm-autoexp/results/old_no_gqa_qwen/qwen3_300M_gpt_neox/dense_qwen3_300M_gpt_neox_lr0.002_gbsz256_beta20.95_decay80BT/checkpoints 76294
/leonardo_work/OELLM_prod2026/users/donutu00/oellm-autoexp/results/old_no_gqa_qwen/qwen3_300M_gpt_neox/dense_qwen3_300M_gpt_neox_lr0.002_gbsz256_beta20.95_decay50BT/checkpoints 47684
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

    MODEL_ARGS="load=${CKPT_DIR},ckpt_step=${CKPT_STEP},ckpt_format=torch_dist,vocab_file=${VOCAB_FILE},merge_file=${MERGE_FILE},tokenizer_type=${TOKENIZER_TYPE}"

    echo "======================================================================"
    echo "[$((n_submitted + 1))] $(basename "$(dirname "$CKPT_DIR")") @ step=${CKPT_STEP}"
    echo "    CKPT_DIR=$CKPT_DIR"
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
