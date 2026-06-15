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

module purge
module load GCCcore/15.2.0  # Passend zu deiner Python-Installation
module load Python/3.14.2
module load CUDA/12.4.0

# Optional: Stelle sicher, dass du im richtigen Verzeichnis bist
cd $SLURM_SUBMIT_DIR

source .venv/bin/activate

# 4. Umgebungsvariablen für vLLM / PyTorch Distributed (Bugfixes)
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export NCCL_DEBUG=WARN # INFO spammt dein Log mit über 100 Zeilen pro GPU voll, WARN reicht.

# Verhindert den IPv6 / localhost Error (errno: 97), den du im Log hast:
export VLLM_HOST_IP=127.0.0.1
export NCCL_SOCKET_IFNAME=eth0 # Oder den entsprechenden Infiniband/Ethernet-Namen des Paula-Knotens, z.B. ibs5


python src/llm_parser/llm_parser.py \
    > logs/output_quwen.log 2>&1

deactivate
