#!/bin/bash
#SBATCH --job-name=flickr_v1_baselines
#SBATCH --output=out_v1_baselines.log
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=140G
#SBATCH --account=csai


# Load environment
source ~/.bashrc
conda activate vlm_mia_latest_venv


export PYTHONPATH=$PYTHONPATH:/local/scratch/clo37/vlm_large_mia/

python /home/clo37/priv/VLM-MIA-Study/mia.py \
    job_meta_params.test_run=false \
    job_meta_params.description="'Baselines of original paper on Model: IFT Llava, dataset: 0.5 flickr members and 0.5 flicker nonmembers from flickr_pretrain_member_ratio_0.5__exact_flickr_KN'" \
    job_meta_params.job_type=evaluation \
    \
    path.output_dir=/local/scratch/clo37/VLM_MIA_STUDY_Archive_Data/FINAL_DATA/FINAL_RESULTS/llava/flickr/baselines/flickr_pretrain_member_ratio_0.5__exact_flickr_KN_baselines \
    \
    target_model="llava-v1.5-7b" \
    \
    data.save_datasets=true \
    data.dataset=/local/scratch/clo37/VLM_MIA_STUDY_Archive_Data/FINAL_DATA/FINAL_RESULTS/llava/flickr/hyperparam_tuning/flickr_pretrain_member_ratio_0.5__exact_flickr_KN/gn_set0/datasets/target_dataset.parquet \
    \
    img_metrics.parts=["img"] \
    img_metrics.metrics_to_use=["aug_kl","min_k_renyi_1_entro","min_k_renyi_05_entro","mink"] \
    img_metrics.get_raw_meta_metrics=[] \
    img_metrics.get_proc_meta_metrics=[] \
    \
    img_metrics.get_proc_meta_examples=0 \
    img_metrics.get_token_labels=0 \
    img_metrics.get_raw_images=0 \
    img_metrics.get_raw_meta_examples=0
