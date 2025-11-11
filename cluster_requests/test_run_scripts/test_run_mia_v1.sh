#!/bin/bash
#SBATCH --job-name=test_run_v1
#SBATCH --output=out_test_run_v1.log
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=300G


# Load environment
source ~/.bashrc
conda activate vlm_mia_latest_venv

member_dataset="/local/scratch/clo37/VLM_MIA_STUDY_Archive_Data/FINAL_DATA/FINAL_DATASETS/coco_2017/coco_2017_members.json"
nonmember_dataset="/local/scratch/clo37/VLM_MIA_STUDY_Archive_Data/FINAL_DATA/FINAL_DATASETS/flickr/flickr_nonmember_subset.json"

export PYTHONPATH=$PYTHONPATH:/local/scratch/clo37/vlm_large_mia/

python /home/clo37/priv/VLM-MIA-Study/mia.py \
    job_meta_params.test_run=true \
    job_meta_params.description="'Test Run 1 with new dataset builder with target set of coco members, flickr nonmembers and sharegpt and flickr'" \
    job_meta_params.job_type=hyperparam_tuning \
    \
    path.output_dir=/local/scratch/clo37/VLM_MIA_STUDY_Archive_Data/test_results/TEST_MIA1 \
    \
    target_model="llava-v1.5-7b" \
    \
    data.save_datasets=true \
    data.target_set_size=300 \
    data.n_nm_ratio=0.5 \
    data.member_dataset=${member_dataset} \
    data.nonmember_dataset=${nonmember_dataset} \
    data.reference_datasets_list=["/local/scratch/clo37/VLM_MIA_STUDY_Archive_Data/FINAL_DATA/FINAL_DATASETS/share_gpt/global_nonmember_fullset.json","/local/scratch/clo37/VLM_MIA_STUDY_Archive_Data/FINAL_DATA/FINAL_DATASETS/flickr/flickr_nonmember_subset.json"] \
    data.reference_set_sample_distribution=[0.5,0.5] \
    \
    img_metrics.parts=["img"] \
    img_metrics.metrics_to_use=["min_k_renyi_05_kl_div","min_k_renyi_1_kl_div","min_k_renyi_2_kl_div","min_k_renyi_inf_kl_div","min_k_renyi_divergence_025","min_k_renyi_divergence_05","min_k_renyi_divergence_2","min_k_renyi_divergence_4"] \
    img_metrics.get_raw_meta_metrics=['losses'] \
    img_metrics.get_proc_meta_metrics=['min_k_renyi_05_kl_div_tkn_vals','min_k_renyi_1_kl_div_tkn_vals','min_k_renyi_2_kl_div_tkn_vals','min_k_renyi_inf_kl_div_tkn_vals','min_k_renyi_divergence_025_tkn_vals','min_k_renyi_divergence_05_tkn_vals','min_k_renyi_divergence_2_tkn_vals','min_k_renyi_divergence_4_tkn_vals'] \
    \
    img_metrics.get_proc_meta_examples=1000 \
    img_metrics.get_token_labels=1000 \
    img_metrics.get_raw_images=5 \
    img_metrics.get_raw_meta_examples=1000 \
    \
    data.augmentations.RandomResize.use=false \
    data.augmentations.RandomResize.size='[[256,256],[256,256],[256,256],[256,256],[256,256],[256,256],[256,256],[256,256],[256,256],[256,256]]' \
    data.augmentations.RandomResize.scale='[[0.2,0.2],[0.4,0.4],[0.6,0.6],[0.8,0.8],[1.0,1.0],[1.0,1.0],[1.0,1.0],[1.0,1.0],[1.0,1.0],[1.0,1.0]]' \
    data.augmentations.RandomResize.ratio='[[1.0,1.0],[1.0,1.0],[1.0,1.0],[1.0,1.0],[1.0,1.0],[0.5,0.5],[0.75,0.75],[1.0,1.0],[1.25,1.25],[1.5,1.5]]' \
    data.augmentations.RandomRotation.use=false \
    data.augmentations.RandomRotation.degrees='[0.1,0.2,0.3,0.4,0.5,5,30,45,60,90]' \
    data.augmentations.GaussianNoise.use=true \
    data.augmentations.GaussianNoise.mean='[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]' \
    data.augmentations.GaussianNoise.std='[1.0,2.5,5.0,7.5,10.0,25.0,50.0,75.0]' \
    data.augmentations.RandomAffine.use=false \
    data.augmentations.ColorJitter.use=false\