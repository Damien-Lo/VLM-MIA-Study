#!/bin/bash
#SBATCH --job-name=pretrain_llava_flickr_gn_set4
#SBATCH --output=out_pretrain_llava_flickr_gn_set4.log
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=300G


# Load environment
source ~/.bashrc
conda activate vlm_mia_latest_venv

export PYTHONPATH=$PYTHONPATH:/local/scratch/clo37/vlm_large_mia/

python /home/clo37/priv/VLM-MIA-Study/mia.py \
    job_meta_params.test_run=false \
    job_meta_params.description="'Pretrain LLaVA Model with MEMBERS=flickr, NONMEMBERS=flickr at std set 4: [75.0,100.0,250.0,500.0,750.0,1000.0,2500.0,5000.0]'" \
    \
    path.output_dir=/local/scratch/clo37/VLM_MIA_STUDY_Archive_Data/results/2025_10_26/pretrain_llava/flickr/gn_set4 \
    \
    target_model="llava-v1.5-mlp2x-336px-pretrain-vicuna-7b-v1.5" \
    \
    data.member_dataset='JaineLi/VL-MIA-image' \
    data.member_subset='img_Flickr' \
    data.member_desc_path='/home/clo37/priv/VLM-MIA-Study/gen_descriptions/llava_pretrain/img_Flickr/member_sentences.json' \
    data.nonmember_dataset='JaineLi/VL-MIA-image' \
    data.nonmember_subset='img_Flickr' \
    data.nonmember_desc_path='/home/clo37/priv/VLM-MIA-Study/gen_descriptions/llava_pretrain/img_Flickr/nonmember_sentences.json' \
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
    data.augmentations.GaussianNoise.std='[75.0,100.0,250.0,500.0,750.0,1000.0,2500.0,5000.0]' \
    data.augmentations.RandomAffine.use=false \
    data.augmentations.ColorJitter.use=false\