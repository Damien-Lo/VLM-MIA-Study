import numpy as np 
import pandas as pd
import json
import random
from pathlib import Path
from datasets import load_dataset, Dataset
from datasets import Features, Value  # (optional)

def build_mix_set(out_path):
    return
    



def main():
    print("Running")
    data_path = "/home/clo37/priv/VLM-MIA-Study/gen_descriptions/llava_pretrain/img_Flickr/sentences.json"
    out_path = "/home/clo37/priv/VLM-MIA-Study/gen_descriptions/llava_pretrain/img_Flickr/nonmember_sentences.json"
    
    
    build_mix_set(out_path)
    
    
    print("Done")


if __name__ == "__main__":
    main()

