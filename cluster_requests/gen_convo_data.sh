#!/bin/bash
#SBATCH --job-name=gen_convo_data
#SBATCH --output=out_gen_convo_data.log
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=300G


# Load environment
source ~/.bashrc
conda activate vlm_large_mia_llava_venv

export PYTHONPATH=$PYTHONPATH:/local/scratch/clo37/vlm_large_mia/

python /home/clo37/priv/VLM-MIA-Study/data_generation.py \
    job_meta_params.test_run=false \
    job_meta_params.description="Generate Coverstaion for COCO members data" \
    \
    path.output_dir=/home/clo37/priv/VLM-MIA-Study/gen_descriptions/llava/coco_2024 \
    \
    target_model="llava-v1.5-7b" \
    \
    data.dataset='coco_2014' \
    data.subset='/local/scratch/clo37/datasets/LLaVA-Instruct-150K/img_coco/member_dataset/member_dataset.arrow'