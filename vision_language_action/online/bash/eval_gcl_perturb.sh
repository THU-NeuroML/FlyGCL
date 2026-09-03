#!/usr/bin/env bash
set -euo pipefail

CHECKPOINT_DIR="${1:-./outputs/libero_10_gcl_clare_expand_v2/checkpoints/last}"
BENCHMARK="${2:-libero_10}"
TASK_PREFIX="${3:-Libero_10_Task}"
DATASET_PREFIX="${4:-continuallearning/libero_10_image_task}"
DATASET_ROOT="${5:-./data/continuallearning/libero_10_image_task_0}"
MAX_TASK="${6:-9}"

export HF_HOME="./.cache/huggingface"
export HF_LEROBOT_HOME="./data"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export MUJOCO_GL=egl
export CUDA_VISIBLE_DEVICES=0

cd .

ALL_TASKS=""
for ((i=0; i<=MAX_TASK; i++)); do
  if [[ -n "$ALL_TASKS" ]]; then
    ALL_TASKS="${ALL_TASKS},"
  fi
  ALL_TASKS="${ALL_TASKS}${TASK_PREFIX}_${i}"
done

PERTURB_BENCHMARK="${BENCHMARK}_task"

echo "评估 checkpoint: $CHECKPOINT_DIR"
echo "扰动 benchmark: $PERTURB_BENCHMARK"
echo "数据路径: $DATASET_ROOT"

python lerobot_lsy/src/lerobot/scripts/eval_peft_multitask.py \
  --policy.path="${CHECKPOINT_DIR}/pretrained_model" \
  --policy.push_to_hub=false \
  --env.type=libero \
  --env.benchmark=${PERTURB_BENCHMARK} \
  --env.task="${ALL_TASKS}" \
  --eval.batch_size=10 \
  --eval.n_episodes=20 \
  --output_dir="${CHECKPOINT_DIR}/eval_results_perturb" \
  --peft_weight_path="${CHECKPOINT_DIR}/adapter" \
  --dataset.repo_id="${DATASET_PREFIX}_0" \
  --dataset.root="${DATASET_ROOT}" \
  2>&1 | tee ./logs/eval_gcl_perturb.log
