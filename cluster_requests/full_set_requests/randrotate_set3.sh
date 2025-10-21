#!/bin/bash
#SBATCH --job-name=randrotate_set3
#SBATCH --output=out_randrotate_set3.log
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=140G


# Load environment
source ~/.bashrc
conda activate vlm_mia_llava_minigpt


export PYTHONPATH=$PYTHONPATH:/local/scratch/clo37/vlm_large_mia/

python /home/clo37/priv/VLM-MIA-Study/mia.py \
    path.output_dir=/local/scratch/clo37/VLM_MIA_STUDY_Archive_Data/LatestResults/full_results/randrotate_set3 \
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
    data.augmentations.RandomRotation.use=true \
    data.augmentations.RandomRotation.degrees='[60.0,80.0,100.0,120.0,135.0,145.0,150.0,160.0]' \
    data.augmentations.GaussianNoise.use=false \
    data.augmentations.GaussianNoise.mean='[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]' \
    data.augmentations.GaussianNoise.std='[75.0,100.0,250.0,500.0,750.0,1000.0,2500.0,5000.0]' \
    data.augmentations.RandomAffine.use=false \
    data.augmentations.ColorJitter.use=false\


