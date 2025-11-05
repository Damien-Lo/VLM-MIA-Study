import numpy as np 
import pandas as pd
import json
import random
from pathlib import Path
from datasets import load_dataset, Dataset
from datasets import Features, Value  # (optional)
import os


# Helper Functions
def random_int_list(m: int, n: int):
    if m > n + 1:
        raise ValueError("m cannot exceed the size of the range (n+1).")
    return random.sample(range(n), m)



# Main Functions
def build_target_set(set_size, m_nm_ratio, member_data_path, non_member_data_path):
    with open(member_data_path, "r") as f:
        member_data = json.load(f)
    with open(non_member_data_path, "r") as f:
        nonmember_data = json.load(f)
    
    
    num_of_mem = int(set_size * m_nm_ratio)
    num_of_nonmem = set_size - num_of_mem
    
    mem_idxs = random_int_list(num_of_mem, len(member_data))
    nonmem_idxs = random_int_list(num_of_nonmem, len(nonmember_data))
    
    target_set = list()
    
    for idx in mem_idxs:
        sample = member_data[idx]
        sample['tune_label'] = 1
        target_set.append(sample)
    for idx in nonmem_idxs:
        sample = nonmember_data[idx]
        sample['tune_label'] = 1
        target_set.append(sample)
    
    return target_set


def json_to_dataset(json_path, out_path=None):

    ds = load_dataset("json", data_files=json_path)["train"]

    features = Features({
        "image": Value("string"),
        "label": Value("int64"),
    })
    ds = ds.cast(features)
    
    if out_path != None:
        ds.save_to_disk(out_path)
        
        data_info_path = os.path.join(out_path, "dataset_info.json")
        state_path = os.path.join(out_path, "state.json")
        
        os.remove(data_info_path)
        os.remove(state_path)
        
    return ds
