#!/bin/bash
#SBATCH --job-name=llm_judge
#SBATCH --output=/data/calfaro/LLMs-profiling/results/logs/llm_judge_%j.out
#SBATCH --error=/data/calfaro/LLMs-profiling/results/logs/llm_judge_%j.err
#SBATCH --partition=nukwa
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --exclude=nukwa-00.cnca,nukwa-01.cnca,nukwa-02.cnca,nukwa-03.cnca
#SBATCH --propagate=NONE

set -euo pipefail

# Scores the stored short-answer and open-ended answers (2,160 judgments:
# 180 questions x 6 models x 2 conditions) with the LLM judge. Nothing is
# re-generated; see src/scripts/llm_judge.py.
#
# NOTE: --time=12:00:00 and --mem=48G are estimates, not measurements. A
# 24B model loaded through bitsandbytes 8-bit decodes slowly, and each
# prompt carries a full course document (~2-3K tokens). Run a timing
# smoke test first, e.g.:
#     sbatch run_llm_judge.slurm --limit 20
# and size --time from its reported s/item. The script is resumable: rows
# already in results/metrics/llm_judge_scores.csv are skipped, so if the
# time limit is hit, just resubmit the same command. If 8-bit is too slow,
# pass --model mistral-small-24b-judge-4bit.
#
# The weights (~47GB bf16) must already be in $HF_HOME. Download only the
# model shards on a node with internet access (skip consolidated.safetensors
# to avoid doubling the download; mind the /data Lustre quota that blocked
# Qwen2.5-32B earlier):
#     huggingface-cli download mistralai/Mistral-Small-24B-Instruct-2501 \
#         --include "model-*.safetensors" "*.json" "tokenizer*" "tekken.json"

PROJECT_DIR=/data/calfaro/LLMs-profiling
PYTHON=/data/calfaro/miniconda/envs/llm-profiling/bin/python

cd "$PROJECT_DIR"
mkdir -p results/logs results/metrics

export HF_HOME=/data/calfaro/app_data/huggingface
export XDG_CACHE_HOME=/data/calfaro/app_data/cache
export TORCH_HOME=/data/calfaro/app_data/cache/torch
export HF_HUB_DISABLE_XET=1
mkdir -p "$HF_HOME" "$XDG_CACHE_HOME" "$TORCH_HOME"

source /data/calfaro/.env   # provides HF_TOKEN

echo "Running on: $(hostname)"
nvidia-smi

cd "$PROJECT_DIR/src/scripts"
"$PYTHON" llm_judge.py "$@"
