import os
import json
import torch
import numpy as np

def convert_tensor_to_serializable(obj):
    """
    Convert PyTorch tensors and other non-serializable objects to JSON-serializable formats.
    """
    if isinstance(obj, torch.Tensor):
        # Convert tensor to numpy array, then to list
        return obj.detach().cpu().numpy().tolist()
    elif isinstance(obj, np.ndarray):
        # Convert numpy array to list
        return obj.tolist()
    elif isinstance(obj, dict):
        # Recursively convert dictionary values
        return {key: convert_tensor_to_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, (list, tuple)):
        # Recursively convert list/tuple elements
        return [convert_tensor_to_serializable(item) for item in obj]
    elif isinstance(obj, (int, float, str, bool, type(None))):
        # These are already JSON-serializable
        return obj
    else:
        # For other types, try to convert to string representation
        return str(obj)

def save_to_json(dict_obj, filename, cfg):
    """
    Save the dictionary object to the file
    """
    output_dir = cfg.path.output_dir
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, f"{filename}.json")
    
    # Convert any tensors in the dictionary to serializable format
    serializable_dict = convert_tensor_to_serializable(dict_obj)
    
    with open(save_path, "w") as f:
        json.dump(serializable_dict, f, indent=4)
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
