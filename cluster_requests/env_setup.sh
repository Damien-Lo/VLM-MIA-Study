#!/bin/bash
#SBATCH --job-name=env_setup
#SBATCH --output=out_env_setup.log
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=140G


# Load environment
source ~/.bashrc
conda activate vlm_mia_llava_minigpt

python -m pip install --upgrade pip wheel setuptools
export PIP_EXTRA_INDEX_URL="https://download.pytorch.org/whl/cu126"

python -m pip install accelerate==1.10.1
python -m pip install aiohappyeyeballs==2.6.1
python -m pip install aiohttp==3.12.15
python -m pip install aiosignal==1.4.0
python -m pip install annotated-types==0.7.0
python -m pip install antlr4-python3-runtime==4.9.3
python -m pip install attrs==25.3.0
python -m pip install braceexpand==0.1.7
python -m pip install certifi==2025.7.14
python -m pip install charset-normalizer==3.4.2
python -m pip install click==8.3.0
python -m pip install contourpy==1.3.3
python -m pip install cycler==0.12.1
python -m pip install datasets==4.1.1
python -m pip install decord==0.6.0
python -m pip install dill==0.4.0
python -m pip install filelock==3.18.0
python -m pip install fonttools==4.60.1
python -m pip install frozenlist==1.7.0
python -m pip install fsspec==2025.7.0
python -m pip install gitdb==4.0.12
python -m pip install GitPython==3.1.45
python -m pip install hf-xet==1.1.5
python -m pip install huggingface-hub==0.35.3
python -m pip install hydra-core==1.3.2
python -m pip install idna==3.10
python -m pip install imageio==2.37.0
python -m pip install iopath==0.1.10
python -m pip install Jinja2==3.1.6
python -m pip install joblib==1.5.2
python -m pip install kiwisolver==1.4.9
python -m pip install lazy_loader==0.4
python -m pip install MarkupSafe==3.0.2
python -m pip install matplotlib==3.10.6
python -m pip install mpmath==1.3.0
python -m pip install multidict==6.6.4
python -m pip install multiprocess==0.70.16
python -m pip install networkx==3.5
python -m pip install numpy==2.2.6
python -m pip install nvidia-cublas-cu12==12.6.4.1
python -m pip install nvidia-cuda-cupti-cu12==12.6.80
python -m pip install nvidia-cuda-nvrtc-cu12==12.6.77
python -m pip install nvidia-cuda-runtime-cu12==12.6.77
python -m pip install nvidia-cudnn-cu12==9.5.1.17
python -m pip install nvidia-cufft-cu12==11.3.0.4
python -m pip install nvidia-cufile-cu12==1.11.1.6
python -m pip install nvidia-curand-cu12==10.3.7.77
python -m pip install nvidia-cusolver-cu12==11.7.1.2
python -m pip install nvidia-cusparse-cu12==12.5.4.2
python -m pip install nvidia-cusparselt-cu12==0.6.3
python -m pip install nvidia-nccl-cu12==2.26.2
python -m pip install nvidia-nvjitlink-cu12==12.6.85
python -m pip install nvidia-nvtx-cu12==12.6.77
python -m pip install omegaconf==2.3.0
python -m pip install opencv-python==4.12.0.88
python -m pip install packaging==25.0
python -m pip install pandas==2.3.3
python -m pip install peft==0.17.1
python -m pip install pillow==11.3.0
python -m pip install platformdirs==4.4.0
python -m pip install portalocker==3.2.0
python -m pip install progressbar2==4.5.0
python -m pip install propcache==0.3.2
python -m pip install protobuf==6.32.1
python -m pip install psutil==7.1.0
python -m pip install pyarrow==21.0.0
python -m pip install pydantic==2.11.10
python -m pip install pydantic_core==2.33.2
python -m pip install pyparsing==3.2.5
python -m pip install python-dateutil==2.9.0.post0
python -m pip install python-utils==3.9.1
python -m pip install pytz==2025.2
python -m pip install PyYAML==6.0.2
python -m pip install regex==2024.11.6
python -m pip install requests==2.32.4
python -m pip install safetensors==0.5.3
python -m pip install scikit-image==0.25.2
python -m pip install scikit-learn==1.7.2
python -m pip install scipy==1.16.2
python -m pip install sentencepiece==0.2.1
python -m pip install sentry-sdk==2.39.0
python -m pip install setuptools==78.1.1
python -m pip install six==1.17.0
python -m pip install smmap==5.0.2
python -m pip install sympy==1.14.0
python -m pip install threadpoolctl==3.6.0
python -m pip install tifffile==2025.9.30
python -m pip install timm==1.0.20
python -m pip install tokenizers==0.22.1
python -m pip install torch==2.7.1
python -m pip install torchaudio==2.7.1
python -m pip install torchvision==0.22.1
python -m pip install tqdm==4.67.1
python -m pip install transformers==4.57.0
python -m pip install triton==3.3.1
python -m pip install typing-inspection==0.4.2
python -m pip install typing_extensions==4.14.1
python -m pip install tzdata==2025.2
python -m pip install urllib3==2.5.0
python -m pip install visual-genome==1.1.1
python -m pip install wandb==0.22.1
python -m pip install webdataset==1.0.2
python -m pip install wheel==0.45.1
python -m pip install xxhash==3.6.0
python -m pip install yarl==1.20.1
