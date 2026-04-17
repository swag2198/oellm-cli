#!/bin/bash
#SBATCH --job-name=oellm-eval
#SBATCH --time={time_limit}
#SBATCH --gres=gpu:$GPUS_PER_NODE
#SBATCH --output={log_dir}/%x-%A-%a.out
#SBATCH --partition=$PARTITION
#SBATCH --account=$ACCOUNT
#SBATCH --error={log_dir}/%x-%A-%a.err
#SBATCH --array=0-{array_limit}%{max_array_len}


CSV_PATH="{csv_path}"
NUM_JOBS={num_jobs}
TOTAL_EVALS={total_evals}
LIMIT="{limit}"
VENV_PATH="{venv_path}"
LM_EVAL_INCLUDE_PATH="{lm_eval_include_path}"
MODULES_TO_LOAD="{modules}"

# avoiding crashes due to compute nodes not having access to the internet
export HF_HOME=$HF_HOME
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export HF_HUB_OFFLINE={hf_hub_offline}

# Path to the shared Singularity image that contains all runtime deps (container mode)
export EVAL_SIF_PATH="$EVAL_BASE_DIR/$EVAL_CONTAINER_IMAGE"

echo "Running eval on $CSV_PATH with $NUM_JOBS array jobs distributing $TOTAL_EVALS evaluations"
echo "SLURM_JOB_ID: $SLURM_JOB_ID"
echo "SLURM_ARRAY_JOB_ID: $SLURM_ARRAY_JOB_ID"
echo "SLURM_ARRAY_TASK_ID: $SLURM_ARRAY_TASK_ID"

if [ "$TOTAL_EVALS" -lt 1 ]; then
    echo "No evaluations to run. Exiting."
    exit 0
fi

# Calculate how many evaluations each array job should handle
EVALS_PER_JOB=$(((TOTAL_EVALS + NUM_JOBS - 1) / NUM_JOBS))

START_INDEX=$((SLURM_ARRAY_TASK_ID * EVALS_PER_JOB + 1))
END_INDEX=$(((SLURM_ARRAY_TASK_ID + 1) * EVALS_PER_JOB))

if [ "$END_INDEX" -gt "$TOTAL_EVALS" ]; then
    END_INDEX=$TOTAL_EVALS
fi

echo "This job (array task $SLURM_ARRAY_TASK_ID) will process evaluations from index $START_INDEX to $END_INDEX."
echo "Each array job handles approximately $EVALS_PER_JOB evaluations."

if [ "$START_INDEX" -gt "$END_INDEX" ]; then
    echo "No evaluations for this job to run. Exiting."
    exit 0
fi

# Use `tail` and `head` to slice the jobs.tsv file for the tasks assigned to this job.
# The +1 on START_INDEX accounts for the header row. Fields are tab-separated so that
# `model_args` (which contains commas) round-trips cleanly through bash.
tail -n +$((START_INDEX + 1)) "$CSV_PATH" | head -n $((END_INDEX - START_INDEX + 1)) | \
while IFS=$'\t' read -r model_path task_path n_shot eval_suite model_type model_args
do
    # Remove trailing carriage returns if script is edited on Windows
    model_path=$(echo "$model_path" | tr -d '\r')
    task_path=$(echo "$task_path" | tr -d '\r')
    n_shot=$(echo "$n_shot" | tr -d '\r')
    eval_suite=$(echo "${{eval_suite:-lm_eval}}" | tr -d '\r')
    model_type=$(echo "${{model_type:-hf}}" | tr -d '\r')
    model_args=$(echo "$model_args" | tr -d '\r')

    # Skip empty lines
    if [ -z "$model_path" ]; then
        continue
    fi

    echo "----------------------------------------------------"
    echo "Starting evaluation for:"
    echo "  Model: $model_path"
    echo "  Task: $task_path"
    echo "  N-shot: $n_shot"
    echo "  Suite: $eval_suite"
    echo "  Model type: $model_type"
    if [ -n "$model_args" ]; then
        echo "  Model args: $model_args"
    fi
    echo "----------------------------------------------------"

    # Build bind paths: always mount the shared eval directory, and additionally
    # bind the directory that contains the model checkpoint when the path exists
    BIND_PATHS="$EVAL_BASE_DIR:$EVAL_BASE_DIR,$HF_HOME:$HF_HOME,$HF_DATASETS_CACHE:$HF_DATASETS_CACHE"

    if [ -e "$model_path" ]; then
        # If the model_path is a file, bind its parent directory; otherwise bind the dir itself
        MODEL_DIR="$model_path"
        if [ ! -d "$MODEL_DIR" ]; then
            MODEL_DIR="$(dirname "$MODEL_DIR")"
        fi
        # Avoid adding duplicate bind if it is already covered by EVAL_BASE_DIR
        if [[ $MODEL_DIR != $EVAL_BASE_DIR* ]]; then
            BIND_PATHS="$BIND_PATHS,$MODEL_DIR:$MODEL_DIR"
        fi
    fi

    # When using --contain, provide a writable home for prediction caches
    SINGULARITY_HOME_ARG=""
    SINGULARITY_ENV_ARGS=""
    if [[ "$SINGULARITY_ARGS" == *"--contain"* ]]; then
        JOB_HOME="$EVAL_BASE_DIR/$USER/container_home/$SLURM_JOB_ID"
        mkdir -p "$JOB_HOME"
        SINGULARITY_HOME_ARG="--home $JOB_HOME"
        SINGULARITY_ENV_ARGS="--env HF_HOME=$HF_HOME --env HF_DATASETS_CACHE=$HF_DATASETS_CACHE --env HF_HUB_OFFLINE=$HF_HUB_OFFLINE --env TRANSFORMERS_OFFLINE=$TRANSFORMERS_OFFLINE --env NLTK_DATA=$NLTK_DATA"
    fi

    GPU_DEVICES=$(seq -s, 0 $(($GPUS_PER_NODE - 1)))

    MULTI_GPU_FLAG=""
    if [ "$GPUS_PER_NODE" -gt 1 ]; then
        MULTI_GPU_FLAG="--multi-gpu"
    fi

    suite_normalized=$(echo "$eval_suite" | tr '[:upper:]' '[:lower:]')

    # Helper function to run Python commands in the appropriate environment.
    # When running under a venv, optionally `module load` any modules requested via
    # the CLI (e.g. gcc/cuda for Megatron-LM); `module` isn't available inside the
    # Singularity container path, so it's a no-op there.
    run_python() {{
        if [ -n "$VENV_PATH" ]; then
            source "$VENV_PATH/bin/activate"
            if [ -n "$MODULES_TO_LOAD" ]; then
                IFS=',' read -ra MODS <<< "$MODULES_TO_LOAD"
                for mod in "${{MODS[@]}}"; do
                    mod="$(echo "$mod" | tr -d '[:space:]')"
                    [ -z "$mod" ] && continue
                    module load "$mod"
                done
            fi
            python "$@"
        else
            singularity exec $SINGULARITY_ARGS \
                $SINGULARITY_ENV_ARGS \
                $SINGULARITY_HOME_ARG \
                --bind $BIND_PATHS \
                $EVAL_SIF_PATH \
                env CUDA_VISIBLE_DEVICES=$GPU_DEVICES \
                python "$@"
        fi
    }}

    case "$suite_normalized" in
        lm_eval|lm-eval|lm-eval-harness)
            model_type_normalized=$(echo "${{model_type:-hf}}" | tr '[:upper:]' '[:lower:]')
            LM_EVAL_EXTRA_ARGS=()
            case "$model_type_normalized" in
                megatron_lm|megatron-lm)
                    if [ -z "$VENV_PATH" ]; then
                        echo "[error] model_type=$model_type requires --venv_path."
                        exit 1
                    fi
                    # Sensible defaults for Leonardo; can be overridden via
                    # --slurm_template_var or by exporting these at submit time.
                    export MEGATRON_PATH="${{MEGATRON_PATH:-/leonardo_work/OELLM_prod2026/users/shaldar0/oellm-autoexp/submodules/Megatron-LM}}"
                    export MASTER_ADDR="${{MASTER_ADDR:-127.0.0.1}}"
                    # Derive a unique MASTER_PORT per array task so co-located jobs
                    # (Slurm packs up to 4 single-GPU tasks onto one 4xA100 node on
                    # Leonardo) don't collide on the torch.distributed TCPStore and
                    # fail with EADDRINUSE. Respect an explicit override if set.
                    if [ -z "${{MASTER_PORT:-}}" ]; then
                        _array_job_id="${{SLURM_ARRAY_JOB_ID:-${{SLURM_JOB_ID:-0}}}}"
                        _array_task_id="${{SLURM_ARRAY_TASK_ID:-0}}"
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
                --output_path "{evals_dir}/$(openssl rand -hex 5).json" \
                --trust_remote_code \
                ${{LM_EVAL_INCLUDE_PATH:+--include_path $LM_EVAL_INCLUDE_PATH}} \
                ${{LIMIT:+--limit $LIMIT}} \
                "${{LM_EVAL_EXTRA_ARGS[@]}}"
            ;;
        lighteval|light-eval)
            LIGHT_TASK="$task_path"

            if [[ -f "$LIGHT_TASK" ]]; then
                LIGHT_TASK_ARG="$LIGHT_TASK"
            else
                last_segment="${{LIGHT_TASK##*|}}"
                if [[ "$LIGHT_TASK" == *"|"* && "$last_segment" =~ ^[0-9]+$ ]]; then
                    if [[ -n "$n_shot" && "$last_segment" != "$n_shot" ]]; then
                        LIGHT_TASK_ARG="${{LIGHT_TASK%|*}}|$n_shot"
                    else
                        LIGHT_TASK_ARG="$LIGHT_TASK"
                    fi
                else
                    LIGHT_TASK_ARG="${{LIGHT_TASK}}|$n_shot"
                fi
            fi

            RESULTS_SUBDIR="{evals_dir}/$(openssl rand -hex 5)"
            mkdir -p "$RESULTS_SUBDIR"

            if [ -n "$VENV_PATH" ]; then
                source "$VENV_PATH/bin/activate"
                lighteval accelerate \
                    "model_name=$model_path,{lighteval_model_args}" \
                    "$LIGHT_TASK_ARG" \
                    --load-tasks-multilingual \
                    --output-dir "$RESULTS_SUBDIR" \
                    ${{LIMIT:+--max-samples $LIMIT}}
            else
                singularity exec $SINGULARITY_ARGS \
                    $SINGULARITY_ENV_ARGS \
                    $SINGULARITY_HOME_ARG \
                    --bind $BIND_PATHS \
                    $EVAL_SIF_PATH \
                    env CUDA_VISIBLE_DEVICES=$GPU_DEVICES \
                    lighteval accelerate \
                        "model_name=$model_path,{lighteval_model_args}" \
                        "$LIGHT_TASK_ARG" \
                        --load-tasks-multilingual \
                        --output-dir "$RESULTS_SUBDIR" \
                        ${{LIMIT:+--max-samples $LIMIT}}
            fi
            ;;
        evalchemy)
            EVALCHEMY_WORK_DIR="{evalchemy_dir}"
            RESULTS_SUBDIR="{evals_dir}/$(openssl rand -hex 5)"
            if [ -n "$VENV_PATH" ]; then
                source "$VENV_PATH/bin/activate"
                cd "$EVALCHEMY_WORK_DIR"
                accelerate launch --num-processes "$GPUS_PER_NODE" --num-machines 1 \
                    $MULTI_GPU_FLAG -m eval.eval \
                    --model hf \
                    --tasks "$task_path" \
                    --model_args "trust_remote_code=True,pretrained=$model_path" \
                    --batch_size auto \
                    --output_path "$RESULTS_SUBDIR" \
                    ${{LIMIT:+--limit $LIMIT}}
            else
                echo "[error] evalchemy suite requires --venv_path (not supported in container mode)."
                exit 1
            fi
            ;;
        *)
            echo "[warning] Unknown evaluation suite '$eval_suite'. Skipping."
            ;;
    esac

    echo "Evaluation finished for model: $model_path"

done

echo "Job $SLURM_ARRAY_TASK_ID finished."
