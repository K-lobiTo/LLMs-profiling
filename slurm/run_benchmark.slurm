#!/bin/bash
#SBATCH --job-name=benchmark
#SBATCH --output=/data/calfaro/LLMs-profiling/results/logs/benchmark_%j.out
#SBATCH --error=/data/calfaro/LLMs-profiling/results/logs/benchmark_%j.err
#SBATCH --partition=nukwa
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --exclude=nukwa-00.cnca,nukwa-01.cnca,nukwa-02.cnca,nukwa-03.cnca
#SBATCH --propagate=NONE

set -euo pipefail

PROJECT_DIR=/data/calfaro/LLMs-profiling
PYTHON=/data/calfaro/miniconda/envs/llm-profiling/bin/python

cd "$PROJECT_DIR"
mkdir -p results/logs results/raw

export HF_HOME=/data/calfaro/app_data/huggingface
export XDG_CACHE_HOME=/data/calfaro/app_data/cache
export TORCH_HOME=/data/calfaro/app_data/cache/torch
export HF_HUB_DISABLE_XET=1
mkdir -p "$HF_HOME" "$XDG_CACHE_HOME" "$TORCH_HOME"

source /data/calfaro/.env   # provides HF_TOKEN

echo "Running on: $(hostname)"
nvidia-smi

MODEL_KEY="${1:?Usage: sbatch run_benchmark.slurm <model_key> [extra run_benchmark.py args]}"
shift

cd "$PROJECT_DIR/src/scripts"
"$PYTHON" run_benchmark.py --model "$MODEL_KEY" "$@"
