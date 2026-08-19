#!/bin/bash
#SBATCH --job-name=smoke_test
#SBATCH --output=/data/calfaro/LLMs-profiling/results/logs/smoke_%j.out
#SBATCH --error=/data/calfaro/LLMs-profiling/results/logs/smoke_%j.err
#SBATCH --partition=nukwa-debug
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00
# nukwa-00..03 tienen V100 (sm_70) — verificar si la build de torch aun
# soporta Volta antes de quitar este exclude. Si torch SI soporta
# sm_70, se puede quitar esta linea y usar V100 tambien.
#SBATCH --exclude=nukwa-00.cnca,nukwa-01.cnca,nukwa-02.cnca,nukwa-03.cnca
#SBATCH --propagate=NONE

set -euo pipefail

# NOTE: verify whether --gres=gpu:1 is needed explicitly on Kabré, or if
# GPU allocation is handled purely via --partition selection. Submit once
# and check `nvidia-smi` in the output log to confirm a GPU was actually
# allocated; add --gres=gpu:1 above if not.

PROJECT_DIR=/data/calfaro/LLMs-profiling
PYTHON=/data/calfaro/miniconda/envs/llm-profiling/bin/python

cd "$PROJECT_DIR"
mkdir -p results/logs

# Redirect caches to /data instead of $HOME — home directories on HPC
# clusters usually have small quotas, and model downloads are multi-GB.
export HF_HOME=/data/calfaro/app_data/huggingface
export XDG_CACHE_HOME=/data/calfaro/app_data/cache
export TORCH_HOME=/data/calfaro/app_data/cache/torch
export HF_HUB_DISABLE_XET=1
mkdir -p "$HF_HOME" "$XDG_CACHE_HOME" "$TORCH_HOME"


echo "Running on: $(hostname)"
nvidia-smi

cd "$PROJECT_DIR/src/scripts"
"$PYTHON" smoke_test_model.py llama3.1-8b-4bit
