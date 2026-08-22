#!/bin/bash
#SBATCH --job-name=fullctx_benchmark
#SBATCH --output=/data/calfaro/LLMs-profiling/results/logs/fullctx_%j.out
#SBATCH --error=/data/calfaro/LLMs-profiling/results/logs/fullctx_%j.err
#SBATCH --partition=nukwa
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --exclude=nukwa-00.cnca,nukwa-01.cnca,nukwa-02.cnca,nukwa-03.cnca
#SBATCH --propagate=NONE

set -euo pipefail

# NOTE: --time=06:00:00, longer than run_benchmark.slurm's 4h. Every
# question here re-processes the entire ~20-25K token corpus instead of
# ~2-3K tokens of retrieved chunks (~8-10x more prefill per question),
# so full-context runs take meaningfully longer than the RAG runs for
# the same model. Adjust upward again if a model's timing (visible after
# the --limit validation run) suggests 6h isn't enough headroom.

PROJECT_DIR=/data/calfaro/LLMs-profiling
PYTHON=/data/calfaro/miniconda/envs/llm-profiling/bin/python

cd "$PROJECT_DIR"
mkdir -p results/logs results/raw_fullcontext

export HF_HOME=/data/calfaro/app_data/huggingface
export XDG_CACHE_HOME=/data/calfaro/app_data/cache
export TORCH_HOME=/data/calfaro/app_data/cache/torch
export HF_HUB_DISABLE_XET=1
mkdir -p "$HF_HOME" "$XDG_CACHE_HOME" "$TORCH_HOME"

source /data/calfaro/.env   # provides HF_TOKEN

echo "Running on: $(hostname)"
nvidia-smi

MODEL_KEY="${1:?Usage: sbatch run_fullcontext_benchmark.slurm <model_key> [extra full_context_benchmark.py args]}"
shift

cd "$PROJECT_DIR/src/scripts"
"$PYTHON" full_context_benchmark.py --model "$MODEL_KEY" "$@"
