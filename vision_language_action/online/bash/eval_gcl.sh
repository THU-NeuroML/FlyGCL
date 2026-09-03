#!/usr/bin/env bash
set -euo pipefail

CHECKPOINT_DIR="${1:-./outputs/libero_10_gcl_clare/checkpoints/last}"
BENCHMARK="${2:-libero_10}"
TASK_PREFIX="${3:-Libero_10_Task}"
DATASET_PREFIX="${4:-continuallearning/libero_10_image_task}"
MAX_TASK="${5:-9}"

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

echo "评估 checkpoint: $CHECKPOINT_DIR"
echo "任务列表: $ALL_TASKS"

python lerobot_lsy/src/lerobot/scripts/eval_peft_multitask.py \
  --policy.path="${CHECKPOINT_DIR}/pretrained_model" \
  --policy.push_to_hub=false \
  --env.type=libero \
  --env.benchmark=${BENCHMARK} \
  --env.task="${ALL_TASKS}" \
  --eval.batch_size=10 \
  --eval.n_episodes=20 \
  --output_dir="${CHECKPOINT_DIR}/eval_results" \
  --peft_weight_path="${CHECKPOINT_DIR}/adapter" \
  --dataset.repo_id="${DATASET_PREFIX}_0" \
  --dataset.root="./data/continuallearning/libero_10_image_task_0" \
  2>&1 | tee ./logs/eval_gcl.log
