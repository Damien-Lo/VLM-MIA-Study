#!/bin/bash
#SBATCH --job-name=generate
#SBATCH --output=/local/scratch/hlee959/2025_vlm_mia/logs/generate.out
#SBATCH --error=/local/scratch/hlee959/2025_vlm_mia/logs/generate.err
#SBATCH --gres=gpu:1
#SBATCH --mem=256G

# Set environment variables
export CONDA_PREFIX=/local/scratch/hlee959/miniconda3
export WANDB_MODE="online"
export TRITON_CACHE_DIR="/local/scratch/hlee959/triton_cache"

mkdir -p $TRITON_CACHE_DIR

CONDA_BASE="/local/scratch/hlee959/miniconda3"
# !! Contents within this block are managed by 'conda init' !!
__conda_setup="$('${CONDA_BASE}/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "${CONDA_BASE}/etc/profile.d/conda.sh" ]; then
        . "${CONDA_BASE}/etc/profile.d/conda.sh"
    else
        export PATH="${CONDA_BASE}/bin:$PATH"
    fi
fi
unset __conda_setup
# <<< conda initialize <<<
conda activate vlm_mia

cd /home/hlee959/projects/2025_MIA_VLM/vlm_mia

python data_generation.py \
    data.subset="img_dalle" \
    path.cache_dir="/local/scratch/hlee959/.cache" \
    path.output_dir="/local/scratch/hlee959/2025_vlm_mia/output" 
