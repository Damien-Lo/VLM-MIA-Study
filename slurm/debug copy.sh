#!/bin/bash
#SBATCH --job-name=debug
#SBATCH --output=/local/scratch/hlee959/2025_vlm_mia/logs/debug.out
#SBATCH --error=/local/scratch/hlee959/2025_vlm_mia/logs/debug.err
#SBATCH --gres=gpu:1
#SBATCH --mem=256G

# Activate conda environment if needed
# source activate your_env_name
source /local/scratch/hlee959/anaconda/bin/activate
conda activate vlm_mia

cd /home/hlee959/projects/2025_MIA_VLM/vlm_mia

python mia.py \
    img_metrics.metrics_to_use=["aug_kl","cross_entropy_mink","cross_entropy_diff_mink"] \
    img_metrics.parts=["img"] \
    data.augmentations.RandomResize.use=False \
    data.augmentations.RandomRotation.use=False \
    data.augmentations.RandomAffine.use=False \
    data.augmentations.GaussianNoise.mean=[0.0,0.0,0.0,0.0,0.0] \
    data.augmentations.GaussianNoise.std=[0.1,0.2,0.3,0.4,0.5]




    # img_metrics.metrics_to_use=["aug_kld","mink","max_k_renyi_1_entro","max_k_renyi_2_entro","max_k_renyi_05_entro","max_prob_gap","min_k_renyi_1_entro","min_k_renyi_2_entro","min_k_renyi_05_entro","mod_renyi_1_entro","mod_renyi_2_entro","mod_renyi_05_entro"] \
