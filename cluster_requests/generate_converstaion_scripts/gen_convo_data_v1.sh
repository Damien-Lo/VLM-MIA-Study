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

RATIOS=(0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0)

# for ratio in "${RATIOS[@]}"; do
python /home/clo37/priv/VLM-MIA-Study/data_generation.py \
    job_meta_params.test_run=false \
    job_meta_params.description="Generate converstaion for flickr original members" \
    \
    path.output_dir=/home/clo37/priv/VLM-MIA-Study/gen_descriptions/llava/img_Flickr/member_sentences.json \
    \
    target_model="llava-v1.5-7b" \
    \
    data.dataset=flickr_original_members \
    data.subset=/local/scratch/clo37/datasets/JaineLi_VL-MIA/flickr/flickr_member_subset.arrow
# done