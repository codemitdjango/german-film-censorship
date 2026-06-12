#!/bin/bash
#SBATCH --job-name=llm_job
#SBATCH --partition=paula            # GPU-Partition zwingend erforderlich
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=24G                    
#SBATCH --gres=gpu:1                 
#SBATCH --time=0-02:00:00              # Maximale Laufzeit anpassen
#SBATCH --output=logs/llm_job_%j.log      # logs

source .venv/bin/activate

pip3 install vllm

python src/llm_parser/llm_parser.py \
    > logs/output_quwen.log 2>&1

deactivate
