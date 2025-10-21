import os
import json
import torch

def save_to_json(dict_obj, filename, cfg):
    output_dir = cfg.path.output_dir
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, f"{filename}.json")
    with open(save_path, "w") as f:
        json.dump(dict_obj, f, indent=4)
    print(f"Saved {filename} to {save_path}")
    
def save_to_pt(tensor, filename, cfg):
    output_dir = cfg.path.output_dir
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, f"{filename}.pt")
    torch.save(tensor, save_path)
    print(f"Saved {filename} to {save_path}")


def load_conversation_template(model_name):
    if "llama-2" in model_name.lower():
        conv_mode = "llava_llama_2"
    elif "mistral" in model_name.lower():
        conv_mode = "mistral_instruct"
    elif "v1.6-34b" in model_name.lower():
        conv_mode = "chatml_direct"
    elif "v1" in model_name.lower():
        conv_mode = "llava_v1"
    elif "mpt" in model_name.lower():
        conv_mode = "mpt"
    else:
        conv_mode = "llava_v0"
    return conv_mode

from pathlib import Path

def save_run_meta(cfg):
    output_dir = cfg.path.output_dir
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, f"run_parameters.txt")
    
    txt = f"""\
        Description: {cfg.job_meta_params.description}
        TEST CASE: {cfg.job_meta_params.test_run}, Only Running on first {cfg.inference.test_number_of_batches} batches
        Target Model: {cfg.target_model.type}
        Target Dataset: {cfg.data.subset}
        Augmentations Used: {cfg.data.augmentations}
        Parts Tested: {cfg.img_metrics.parts}
        Metrics Tested: {cfg.img_metrics.metrics_to_use}

        Requested token labels of first {cfg.img_metrics.get_token_labels} of each class
        Requested Raw Augmented Images of first {cfg.img_metrics.get_raw_images} of each class
        Requested metrics: {cfg.img_metrics.get_raw_meta_metrics} of first {cfg.img_metrics.get_raw_meta_examples} of each class
        Requested metrics: {cfg.img_metrics.get_proc_meta_metrics} of first {cfg.img_metrics.get_proc_meta_examples} of each class
        """
    Path(save_path).write_text(txt, encoding="utf-8")
    return save_path

def build_descriptions_dataset(cfg):
    member_desc_path = os.path.join(str(cfg.data.desc_data_root_path), str(cfg.target_model.type), str(cfg.data.member_subset), "member_sentences.json")
    nonmember_desc_path = os.path.join(str(cfg.data.desc_data_root_path), str(cfg.target_model.type), str(cfg.data.nonmember_subset), "nonmember_sentences.json")
    with open(member_desc_path, 'r') as f:
        mem_data = json.load(f)
    with open(nonmember_desc_path, 'r') as f:
        nonmem_data = json.load(f)
    
    member_idxs = mem_data['idxs']
    nonmember_idxs = nonmem_data['idxs']
        
    descriptions = list()
    descriptions.extend(mem_data["sentences"])
    descriptions.extend(nonmem_data['sentences'])
    return member_idxs, nonmember_idxs, descriptions