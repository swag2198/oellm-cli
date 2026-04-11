#!/usr/bin/env bash
# cd oellm-cli
source .venv-megatron/bin/activate
module load gcc/12.2.0; module load cuda/12.6


export MEGATRON_PATH=/leonardo_work/OELLM_prod2026/users/shaldar0/oellm-autoexp/submodules/Megatron-LM
export VOCAB_FILE=/leonardo_work/OELLM_prod2026/models/EleutherAI/gpt-neox-20b/vocab.json
export MERGE_FILE=/leonardo_work/OELLM_prod2026/models/EleutherAI/gpt-neox-20b/merges.txt
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=29500


# CKPT_DIR / CKPT_STEP: optional args (defaults below). Example:
# bash scripts/lm_eval_megatron.sh /path/to/.../checkpoints 57221

# 256, 0.001, 120BT -- 4 sparsity configs
# bash scripts/lm_eval_megatron.sh /leonardo_work/OELLM_prod2026/users/shaldar0/oellm-autoexp/results/moe_bsz256_lr_sparsity_grid_A130M_120BT/moe_abl_nexp_8_lr0.001_gbsz256_seed1234_decay120BT/checkpoints 114441 -- started
# bash scripts/lm_eval_megatron.sh /leonardo_work/OELLM_prod2026/users/shaldar0/oellm-autoexp/results/moe_bsz256_lr_sparsity_grid_A130M_120BT/moe_abl_nexp_16_lr0.001_gbsz256_seed1234_decay120BT/checkpoints 114441 -- started
# bash scripts/lm_eval_megatron.sh /leonardo_work/OELLM_prod2026/users/shaldar0/oellm-autoexp/results/moe_bsz256_lr_sparsity_grid_A130M_120BT/moe_abl_nexp_32_lr0.001_gbsz256_seed1234_decay120BT/checkpoints 114441 -- started
# bash scripts/lm_eval_megatron.sh /leonardo_work/OELLM_prod2026/users/shaldar0/oellm-autoexp/results/moe_bsz256_lr_sparsity_grid_A130M_120BT/moe_abl_nexp_64_lr0.001_gbsz256_seed1234_decay120BT/checkpoints 114441 -- started

DEFAULT_CKPT_DIR=/leonardo_work/OELLM_prod2026/users/shaldar0/oellm-autoexp/results/moe_bsz256_lr_sparsity_grid_A130M_120BT/moe_abl_nexp_8_lr0.001_gbsz256_seed1234_decay120BT/checkpoints
DEFAULT_CKPT_STEP=114441
export CKPT_DIR="${1:-${DEFAULT_CKPT_DIR}}"
export CKPT_STEP="${2:-${DEFAULT_CKPT_STEP}}"

# Output (same layout as lm_eval_megatron_batch.sh):
#   OUTPUT_BASE/<grid>__<decay>__step<CKPT_STEP>/<run_dir>/<task>/
# CKPT_DIR must be .../<run_dir>/checkpoints; decay is parsed from <run_dir> (e.g. ..._decay80BT).
CKPT_GRID="$(basename "$(dirname "$(dirname "${CKPT_DIR}")")")"
CKPT_RUN="$(basename "$(dirname "${CKPT_DIR}")")"
DECAY="decay_unknown"
if [[ "${CKPT_RUN}" =~ _(decay[0-9]+BT)$ ]]; then
  DECAY="${BASH_REMATCH[1]}"
fi
MODEL_ID="${CKPT_GRID}__${DECAY}__step${CKPT_STEP}"

OUTPUT_BASE=/leonardo_work/OELLM_prod2026/users/shaldar0/oellm-evals/outputs/lmeval_megatron_batch

MODEL_ARGS="load=${CKPT_DIR},ckpt_step=${CKPT_STEP},ckpt_format=torch_dist,vocab_file=${VOCAB_FILE},merge_file=${MERGE_FILE},tokenizer_type=GPT2BPETokenizer"

# Usage: run_lm_eval <lm_eval_task_name> <num_fewshot>
run_lm_eval() {
  local task_name=$1
  local nshot=$2
  local out="${OUTPUT_BASE}/${MODEL_ID}/${CKPT_RUN}/${task_name}"
  mkdir -p "$out"
  lm_eval --model megatron_lm --model_args "${MODEL_ARGS}" \
    --tasks "${task_name}" \
    --output_path "${out}" \
    --num_fewshot "${nshot}" \
    --batch_size 1
}

# Task names must match lm-eval-harness registry for --tasks.
# num_fewshot per task follows your eval config; dataset or subset affects HF loading only.
# run_lm_eval copa 0
# run_lm_eval boolq 10
# run_lm_eval openbookqa 0
# run_lm_eval lambada_openai 0
# run_lm_eval winogrande 0
# run_lm_eval arc_easy 10
# run_lm_eval arc_challenge 10
# run_lm_eval commonsense_qa 10
# run_lm_eval piqa 10
# run_lm_eval hellaswag 10
run_lm_eval social_iqa 0
run_lm_eval mmlu 5

