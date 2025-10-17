#!/bin/bash
#SBATCH --job-name=debug_minigpt4
#SBATCH --output=/local/scratch/hlee959/2025_vlm_mia/logs/debug_minigpt4.out
#SBATCH --error=/local/scratch/hlee959/2025_vlm_mia/logs/debug_minigpt4.err
#SBATCH --gres=gpu:1
#SBATCH --mem=256G

# Activate conda environment if needed
# source activate your_env_name
source /local/scratch/hlee959/anaconda/bin/activate
conda activate vlm_mia

cd /home/hlee959/projects/2025_MIA_VLM/vlm_mia/MiniGPT-4/

python run_with_img.py

