import numpy as np 
import pandas as pd
import json
import random
from pathlib import Path
from datasets import load_dataset, Dataset
from datasets import Features, Value  # (optional)


def get_random_subset():
    data_path = '/local/scratch/clo37/datasets/LLaVA-Instruct-150K/llava_instruct_150k.json'
    with open(data_path, "r") as f:
        data = json.load(f)
        
    print(f"Length: {len(data)}")
    print(f"Type: {type(data)}")

    def random_int_list(m: int, n: int):
        return [random.randint(0, n) for _ in range(m)]

    desired_idxs = random_int_list(300, len(data))
    result = list()
    for idx, sample in enumerate(data):
        if idx in desired_idxs:
            result.append(
                {
                    'image': f"/local/scratch/clo37/datasets/COCO_2014_train/train2014/COCO_train2014_{sample['image']}",
                    'label': 1
                }
                )
            
    out_path = Path("/local/scratch/clo37/datasets/LLaVA-Instruct-150K/img_coco/train.jsonl")
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2, sort_keys=True)
        

def build_local_dataset():

    json_path = "/local/scratch/clo37/datasets/LLaVA-Instruct-150K/img_coco/train.jsonl"

    # Load as a single 'train' split
    ds = load_dataset("json", data_files=json_path)["train"]

    # (Optional) enforce simple types; we want 'image' to stay as a string path
    # so your convert_to_augmentation_mod_infer keeps working.
    features = Features({
        "image": Value("string"),
        "label": Value("int64"),
    })
    ds = ds.cast(features)
    
    save_dir = "/local/scratch/clo37/datasets/LLaVA-Instruct-150K/img_coco/member_dataset"
    ds.save_to_disk(save_dir)




def main():
    print("Running")
    
    
    build_local_dataset()
    
    
    print("Done")


if __name__ == "__main__":
    main()

