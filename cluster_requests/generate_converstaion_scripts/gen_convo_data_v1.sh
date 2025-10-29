#!/bin/bash
#SBATCH --job-name=gen_convo_data
#SBATCH --output=out_gen_convo_data_v1.log
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=300G


# Load environment
source ~/.bashrc
conda activate vlm_mia_latest_venv

export PYTHONPATH=$PYTHONPATH:/local/scratch/clo37/vlm_large_mia/


for RATIO in 0.7 0.8 0.9 1.0; do

    python /home/clo37/priv/VLM-MIA-Study/data_generation.py \
        job_meta_params.test_run=false \
        job_meta_params.description="Generate Conversation for img_coco mixed for LLaVA model" \
        \
        path.output_dir=/home/clo37/priv/VLM-MIA-Study/gen_descriptions/llava/mixed_sets/coco_2017_mixed/coco_2017_IFT_member_ratio_${RATIO}.json \
        \
        target_model="llava-v1.5-7b" \
        \
        data.member_dataset=mixed_coco_${ratio} \
        data.subset=/local/scratch/clo37/datasets/mixed_sets/coco_2017_ift/coco_2017_IFT_member_ratio_${RATIO}/mixed_subset_memrat_${RATIO}_target.arrow
done