#!/bin/bash
#SBATCH --job-name=run_mia_v1
#SBATCH --output=out_run_mia_v1.log
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=140G


# Load environment
source ~/.bashrc
conda activate vlm_mia_latest_venv

STD_SETS=(
  "[0.005,0.013,0.036,0.097,0.26]"
  "[0.69,2.5,5.0,13,36]"
  "[97,260,690,1900,5000]"
)


export PYTHONPATH=$PYTHONPATH:/local/scratch/clo37/vlm_large_mia/

for ((set=0; set<3; set++)); do
    python /home/clo37/priv/VLM-MIA-Study/mia.py \
        job_meta_params.test_run=false \
        job_meta_params.description="'LLava Fully Fine tune model hyperparam find tuning of flickr_0.5 original made set with member ratio 0.5 at To compare to best std'" \
        \
        path.output_dir=/local/scratch/clo37/VLM_MIA_STUDY_Archive_Data/mixed_set_results/llava/eval/flickr/flickr_member_ratio_0.5_eval/gn_set${set} \
        \
        target_model="llava-v1.5-7b" \
        \
        data.dataset='JaineLi/VL-MIA-image' \
        data.subset='img_Flickr' \
        data.single_desc_path='/home/clo37/priv/VLM-MIA-Study/gen_descriptions/llava/mixed_sets/flickr_mixed/flickr_pretrain_member_ratio_0.5.json' \
        \
        img_metrics.parts=["img"] \
        img_metrics.metrics_to_use=["max_k_no_norn_kl_div","max_k_renyi_05_kl_div","max_k_renyi_1_kl_div","max_k_renyi_2_kl_div","max_k_renyi_inf_kl_div","max_k_renyi_divergence_025","max_k_renyi_divergence_05","max_k_renyi_divergence_2","max_k_renyi_divergence_4"] \
        img_metrics.get_raw_meta_metrics=['losses'] \
        img_metrics.get_proc_meta_metrics=['max_k_no_norn_kl_div_tkn_vals','max_k_renyi_05_kl_div_tkn_vals','max_k_renyi_1_kl_div_tkn_vals','max_k_renyi_2_kl_div_tkn_vals','max_k_renyi_inf_kl_div_tkn_vals','max_k_renyi_divergence_025_tkn_vals','max_k_renyi_divergence_05_tkn_vals','max_k_renyi_divergence_2_tkn_vals','max_k_renyi_divergence_4_tkn_vals'] \
        \
        img_metrics.get_proc_meta_examples=1000 \
        img_metrics.get_token_labels=1000 \
        img_metrics.get_raw_images=5 \
        img_metrics.get_raw_meta_examples=1000 \
        \
        data.augmentations.GaussianNoise.use=true \
        data.augmentations.GaussianNoise.mean='[0.0,0.0,0.0,0.0,0.0]' \
        data.augmentations.GaussianNoise.std="${STD_SETS[$set]}" \
        data.augmentations.RandomResize.use=false \
        data.augmentations.RandomRotation.use=false \
        data.augmentations.RandomAffine.use=false \
        data.augmentations.ColorJitter.use=false
done
