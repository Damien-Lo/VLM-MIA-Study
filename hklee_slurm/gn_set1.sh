#!/bin/bash
#SBATCH --job-name=gn_01
#SBATCH --output=/local/scratch/hlee959/2025_vlm_mia/logs/gn_01
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=140G

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
    path.cache_dir=/local/scratch/hlee959/.cache \
    path.output_dir=/local/scratch/hlee959/2025_vlm_mia/output/gn_01 \
    data.subset="img_dalle" \
    img_metrics.parts=["img"] \
\
    img_metrics.metrics_to_use='["min_k_renyi_05_kl_div","min_k_renyi_1_kl_div","min_k_renyi_2_kl_div","min_k_renyi_inf_kl_div","min_k_renyi_divergence_025","min_k_renyi_divergence_05","min_k_renyi_divergence_2","min_k_renyi_divergence_4"]' \
\
    img_metrics.get_token_labels=1000 \
    img_metrics.get_raw_images=5 \
\
    img_metrics.get_raw_meta_examples=1000 \
    img_metrics.get_raw_meta_metrics='["losses"]'\
\
    img_metrics.get_proc_meta_examples=1000 \
    img_metrics.get_proc_meta_metrics='["min_k_renyi_05_kl_div_tkn_vals","min_k_renyi_1_kl_div_tkn_vals","min_k_renyi_2_kl_div_tkn_vals","min_k_renyi_inf_kl_div_tkn_vals","min_k_renyi_divergence_025_tkn_vals","min_k_renyi_divergence_05_tkn_vals","min_k_renyi_divergence_4_tkn_vals"]'\
\
    data.augmentations.RandomResize.use=false \
    data.augmentations.RandomResize.size='[[256,256],[256,256],[256,256],[256,256],[256,256],[256,256],[256,256],[256,256],[256,256],[256,256]]' \
    data.augmentations.RandomResize.scale='[[0.2,0.2],[0.4,0.4],[0.6,0.6],[0.8,0.8],[1.0,1.0],[1.0,1.0],[1.0,1.0],[1.0,1.0],[1.0,1.0],[1.0,1.0]]' \
    data.augmentations.RandomResize.ratio='[[1.0,1.0],[1.0,1.0],[1.0,1.0],[1.0,1.0],[1.0,1.0],[0.5,0.5],[0.75,0.75],[1.0,1.0],[1.25,1.25],[1.5,1.5]]' \
    data.augmentations.RandomRotation.use=false \
    data.augmentations.RandomRotation.degrees='[0.1,0.2,0.3,0.4,0.5,5,30,45,60,90]' \
    data.augmentations.GaussianNoise.use=true \
    data.augmentations.GaussianNoise.mean='[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]' \
    data.augmentations.GaussianNoise.std='[0.0001,0.00025,0.0005,0.00075,0.001,0.0025,0.005,0.0075]' \
    data.augmentations.RandomAffine.use=false \
    data.augmentations.ColorJitter.use=false\


