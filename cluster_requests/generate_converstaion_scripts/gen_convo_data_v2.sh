#!/bin/bash
#SBATCH --job-name=gen_convo_data
#SBATCH --output=out_gen_convo_data_v2.log
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=300G


# Load environment
source ~/.bashrc
conda activate vlm_mia_latest_venv

export PYTHONPATH=$PYTHONPATH:/local/scratch/clo37/vlm_large_mia/

for RATIO in 0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0; do
    echo ">>> RATIO=${RATIO}"

    python /home/clo37/priv/VLM-MIA-Study/data_generation.py \
        job_meta_params.test_run=false \
        job_meta_params.description="Generate Conversation for flickr mixed for LLaVA model" \
        \
        path.output_dir=/home/clo37/priv/VLM-MIA-Study/gen_descriptions/llava/mixed_sets/flickr_mixed/flickr_pretrain_member_ratio_${RATIO}.json \
        \
        target_model="llava-v1.5-7b" \
        \
        data.member_dataset=mixed_flickr_${ratio} \
        data.subset=/local/scratch/clo37/datasets/mixed_sets/flickr/flickr_pretrain_member_ratio_${RATIO}/mixed_subset_memrat_${RATIO}_target.arrow
done