#!/bin/bash
#SBATCH --job-name=qwen_parser
#SBATCH --partition=paula         
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128GB                
#SBATCH --gres=gpu:4             
#SBATCH --time=0-02:00:00                   # Maximale Laufzeit anpassen
#SBATCH --output=logs/qwen_parser%j.log     # logs

source .venv/bin/activate

#pip3 install vllm

# Optional: Stelle sicher, dass du im richtigen Verzeichnis bist
cd $SLURM_SUBMIT_DIR

# NCCL Fixes & Debugging??
export NCCL_DEBUG=INFO
# export NCCL_IB_DISABLE=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn

python src/llm_parser/llm_parser.py \
    > logs/output_quwen.log 2>&1

deactivate
