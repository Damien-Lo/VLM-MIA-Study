#!/bin/bash
#SBATCH --job-name=run_mia_v2
#SBATCH --output=out_run_mia_v2.log
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=300G


# Load environment
source ~/.bashrc
conda activate vlm_large_mia_llava_venv


export PYTHONPATH=$PYTHONPATH:/local/scratch/clo37/vlm_large_mia/

python /home/clo37/priv/VLM-MIA-Study/mia.py \
    path.output_dir=/local/scratch/clo37/VLM_MIA_STUDY_Archive_Data/LatestResults/2025_09_09_14-45 \
    img_metrics.parts=["img"] \
\
    img_metrics.metrics_to_use='[
        "min_k_renyi_05_kl_div",
        "min_k_renyi_1_kl_div",
        "min_k_renyi_2_kl_div",
        "min_k_renyi_inf_kl_div",
        "min_k_renyi_divergence_025",
        "min_k_renyi_divergence_05",
        "min_k_renyi_divergence_2",
        "min_k_renyi_divergence_4"
        ]' \
\
    img_metrics.get_token_labels=1000 \
    img_metrics.get_raw_images=5 \
\
    img_metrics.get_raw_meta_examples=1000 \
    img_metrics.get_raw_meta_metrics='[
        "ppl",
        "losses",
        ]'\
\
    img_metrics.get_proc_meta_examples=1000 \
    img_metrics.get_proc_meta_metrics='[
        "min_k_renyi_05_kl_div_tkn_vals",
        "min_k_renyi_1_kl_div_tkn_vals",
        "min_k_renyi_2_kl_div_tkn_vals",
        "min_k_renyi_inf_kl_div_tkn_vals",
        "min_k_renyi_divergence_025_tkn_vals",
        "min_k_renyi_divergence_05_tkn_vals",
        "min_k_renyi_divergence_4_tkn_vals",
        ]'\
\
    data.augmentations.RandomResize.use=false \
    data.augmentations.RandomResize.size='[[256,256],[256,256],[256,256],[256,256],[256,256],[256,256],[256,256],[256,256],[256,256],[256,256]]' \
    data.augmentations.RandomResize.scale='[[0.2,0.2],[0.4,0.4],[0.6,0.6],[0.8,0.8],[1.0,1.0],[1.0,1.0],[1.0,1.0],[1.0,1.0],[1.0,1.0],[1.0,1.0]]' \
    data.augmentations.RandomResize.ratio='[[1.0,1.0],[1.0,1.0],[1.0,1.0],[1.0,1.0],[1.0,1.0],[0.5,0.5],[0.75,0.75],[1.0,1.0],[1.25,1.25],[1.5,1.5]]' \
    data.augmentations.RandomRotation.use=false \
    data.augmentations.RandomRotation.degrees='[0.1,0.2,0.3,0.4,0.5,5,30,45,60,90]' \
    data.augmentations.GaussianNoise.use=true \
    data.augmentations.GaussianNoise.mean='[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]' \
    data.augmentations.GaussianNoise.std='[0.01,0.025,0.05,0.075,0.1,0.25,0.5,0.75]' \
    data.augmentations.RandomAffine.use=false \
    data.augmentations.ColorJitter.use=false\






    # img_metrics.get_raw_meta_metrics='[
    #     "ppl",
    #     "entropies",
    #     "all_prob",
    #     "probabilities",
    #     "log_probabilities",
    #     "modified_entropies",
    #     "max_probs",
    #     "gap_probs",
    #     "renyi_05_entro",
    #     "renyi_2_entro",
    #     "losses",
    #     "modified_entropies_alpha_05",
    #     "modified_entropies_alpha_2",
    #     "renyi_05_probs",
    #     "renyi_1_probs",
    #     "renyi_2_probs",
    #     "renyi_inf_probs",
    #     "per_token_CE_loss"
    #     ]'\