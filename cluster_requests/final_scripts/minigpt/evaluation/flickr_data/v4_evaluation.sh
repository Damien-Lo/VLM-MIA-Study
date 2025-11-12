#!/bin/bash
#SBATCH --job-name=flickr_v4_evaluation_extra_two
#SBATCH --output=out_v4_evaluation_extra_two.log
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=140G
#SBATCH --account=csai


# Load environment
source ~/.bashrc
conda activate vlm_mia_latest_venv

STD_SETS=(
  "[0.005,0.0078,0.012,0.019,0.03,0.046,0.072,0.11]"
  "[0.18,0.28,0.43,0.67,1.1,1.6,2.6,4.0]"
  "[6.2,9.8,15,24,37,58,91,140]"
  "[220,340,540,840,1300,2100,3200,5000]"
)


export PYTHONPATH=$PYTHONPATH:/local/scratch/clo37/vlm_large_mia/

for ((set=0; set<4; set++)); do
    printf "\n>>>===================\n\nUsing STD set $set: ${STD_SETS[$set]}\n\n=================== \n\n"
    python /home/clo37/priv/VLM-MIA-Study/mia.py \
        job_meta_params.test_run=false \
        job_meta_params.description="'Evaluation on Model: IFT Llava, dataset: 0.5 flicker members and 0.5 flicker nonmembers from flickr_pretrain_member_ratio_0.5__sharegpt_KN'" \
        job_meta_params.job_type=evaluation \
        \
        path.output_dir=/local/scratch/clo37/VLM_MIA_STUDY_Archive_Data/FINAL_DATA/FINAL_RESULTS/llava/flickr/evaluation/flickr_pretrain_member_ratio_0.5__sharegpt_KN_eval/run_three/gn_set${set} \
        \
        target_model="minigpt-4" \
        \
        data.dataset=/local/scratch/clo37/VLM_MIA_STUDY_Archive_Data/FINAL_DATA/FINAL_RESULTS/llava/flickr/hyperparam_tuning/aditional_flickr_pretrain_member_ratio_0.5__sharegpt_KN/run_two/gn_set0/datasets/target_dataset.parquet \
        \
        img_metrics.parts=["img"] \
        img_metrics.metrics_to_use=["max_k_no_norn_kl_div","max_k_renyi_05_kl_div","max_k_renyi_1_kl_div","max_k_renyi_2_kl_div","max_k_renyi_inf_kl_div","max_k_renyi_divergence_025","max_k_renyi_divergence_05","max_k_renyi_divergence_2","max_k_renyi_divergence_4"] \
        img_metrics.get_raw_meta_metrics=['losses'] \
        img_metrics.get_proc_meta_metrics=['max_k_no_norn_kl_div_tkn_vals','max_k_renyi_05_kl_div_tkn_vals','max_k_renyi_1_kl_div_tkn_vals','max_k_renyi_2_kl_div_tkn_vals','max_k_renyi_inf_kl_div_tkn_vals','max_k_renyi_divergence_025_tkn_vals','max_k_renyi_divergence_05_tkn_vals','max_k_renyi_divergence_2_tkn_vals','max_k_renyi_divergence_4_tkn_vals'] \
        \
        img_metrics.get_proc_meta_examples=1000 \
        img_metrics.get_token_labels=1000 \
        img_metrics.get_raw_images=1 \
        img_metrics.get_raw_meta_examples=1000 \
        \
        data.augmentations.GaussianNoise.use=true \
        data.augmentations.GaussianNoise.mean='[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]' \
        data.augmentations.GaussianNoise.std="${STD_SETS[$set]}" \
        data.augmentations.RandomResize.use=false \
        data.augmentations.RandomRotation.use=false \
        data.augmentations.RandomAffine.use=false \
        data.augmentations.ColorJitter.use=false
done
