#!/bin/bash
#SBATCH --job-name=debug
#SBATCH --output=/local/scratch/hlee959/2025_vlm_mia/logs/debug.out
#SBATCH --error=/local/scratch/hlee959/2025_vlm_mia/logs/debug.err
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

python mia.py \
    data.subset="img_dalle" \
    img_metrics.metrics_to_use=["aug_kl","cross_entropy_mink","cross_entropy_diff_mink"] \
    img_metrics.parts=["img"] \
    data.augmentations.RandomResize.use=False \
    data.augmentations.RandomRotation.use=False \
    data.augmentations.RandomAffine.use=False \
    data.augmentations.GaussianNoise.mean=[0.0,0.0,0.0,0.0,0.0] \
    data.augmentations.GaussianNoise.std=[0.1,0.2,0.3,0.4,0.5]




    # img_metrics.metrics_to_use=["aug_kld","mink","max_k_renyi_1_entro","max_k_renyi_2_entro","max_k_renyi_05_entro","max_prob_gap","min_k_renyi_1_entro","min_k_renyi_2_entro","min_k_renyi_05_entro","mod_renyi_1_entro","mod_renyi_2_entro","mod_renyi_05_entro"] \
