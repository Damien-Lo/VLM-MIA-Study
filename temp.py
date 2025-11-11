import numpy as np 
import pandas as pd
import json
import random
from pathlib import Path
from datasets import load_dataset, Dataset, concatenate_datasets
from datasets import Features, Value  # (optional)
import os
from datasets.features import Image as HFImage
import math
import sys


def json_to_dataset(json_path_one, json_path_two , out_path=None):

    ds_one = load_dataset("json", data_files=json_path_one)['train']
    
    ds_two = load_dataset("json", data_files=json_path_two)['train']
    
    ds = concatenate_datasets([ds_one,ds_two])

    ds = ds.cast_column("image", HFImage(decode=True))
    
    if out_path != None:
        ds.to_parquet(out_path)
        
    return ds


json_to_dataset("/local/scratch/clo37/VLM_MIA_STUDY_Archive_Data/FINAL_DATA/FINAL_DATASETS/flickr/flickr_member_subset.json",
                "/local/scratch/clo37/VLM_MIA_STUDY_Archive_Data/FINAL_DATA/FINAL_DATASETS/flickr/flickr_nonmember_subset.json",
                "/local/scratch/clo37/VLM_MIA_STUDY_Archive_Data/FINAL_DATA/FINAL_DATASETS/flickr/full_dataset.parquet")

