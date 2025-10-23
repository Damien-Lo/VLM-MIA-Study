import numpy as np
from collections import defaultdict
import copy
import sys
from torchvision import transforms

"""
Main MIA entry point
"""
import os
import json
import hydra
import torch
from src.eval import evaluate
from src.inference import inference
from src.data import get_mod_infer_data
from src.data import get_generation_data
from src.model import generate, mod_infer_batch
from llava.mm_utils import get_model_name_from_path
from llava.model.builder import load_pretrained_model
from src.misc import save_to_json, save_to_pt, save_run_meta, load_conversation_template, build_descriptions_dataset
from textwrap import dedent

@hydra.main(version_base=None, config_path="./config", config_name="run_img")
def main(cfg):
    
    print('''
          \n \n
          ==================================================
                            STARTING RUN 
          ==================================================
          \n \n
          '''
          )
    
    if cfg.job_meta_params.test_run:
        print('''
          \n \n
          ==================================================
                        !!THIS IS A TEST RUN!!
          ==================================================
          \n \n
          '''
          )
        print(f"Only Running on first {cfg.inference.test_number_of_batches} batches")
    
    print("Full Config Paramters:")
    print(cfg)
    print("\n")
    print("AUGMENTATIONS USED:")
    print(cfg.data.augmentations)
    print("\nREQUESTED DATA")
    
    if cfg.img_metrics.get_token_labels > 0 :
        print(f"Requested token labels of first {cfg.img_metrics.get_token_labels} of each class")
        
    if cfg.img_metrics.get_raw_images > 0:
        print(f"Requested Raw Augmented Images of first {cfg.img_metrics.get_raw_images} of each class")
    
    if cfg.img_metrics.get_raw_meta_examples > 0:
        print(f"Requested metrics: {cfg.img_metrics.get_raw_meta_metrics} of first {cfg.img_metrics.get_raw_meta_examples} of each class")
        
    if cfg.img_metrics.get_proc_meta_examples > 0:
        print(f"Requested metrics: {cfg.img_metrics.get_proc_meta_metrics} of first {cfg.img_metrics.get_proc_meta_examples} of each class")
        
    save_run_meta(cfg)
        
        
    print('''
          \n \n
          ==================================================
                            LOADING MODEL
          ==================================================
          '''
          )

    # Load the target model
    model_name = get_model_name_from_path(cfg.target_model.model_path)

    tokenizer, model, image_processor, context_len = load_pretrained_model(
        cfg.target_model.model_path, 
        cfg.target_model.model_base, 
        model_name, 
        gpu_id=cfg.target_model.gpu_id,
        cache_dir=cfg.path.cache_dir
    )
    conv_mode = load_conversation_template(model_name)


    # Generation data
    text = cfg.prompt.text

    # gen_data = get_generation_data(cfg, tokenizer, image_processor, text, model.config, conv_mode)
    # indices, descriptions = generate(model, tokenizer, gen_data, cfg)
    
    from gen_text import flickr_sentences, dalle_sentences
    
    # If we want to get meta values and labels for some samples (first x members and nonmembers) find the indecies these samples live
    print('''
          \n \n
          ==================================================
                            PREPARING DATA
          ==================================================
          \n \n
          '''
          )
    
    member_idxs, nonmember_idxs, descriptions = build_descriptions_dataset(cfg)

    print("Generating Inference and Augmentations.....")
    mod_infer_data, image_sampled_indicies = get_mod_infer_data(cfg, member_idxs, nonmember_idxs, descriptions, tokenizer, image_processor, text, model.config, conv_mode)
    proc_meta_vaues_sampled_indices = list()
    raw_meta_vaues_sampled_indices = list()
    class_labels = mod_infer_data["label"]
    
    if cfg.job_meta_params.test_run:
        class_labels = class_labels[: (cfg.inference.batch_size * cfg.inference.test_number_of_batches)]
    

    if cfg.img_metrics.get_raw_meta_examples > 0:
        raw_meta_vaues_sampled_indices = np.sort(np.concatenate([
        np.where(np.array(class_labels) == 1)[0][:cfg.img_metrics.get_raw_meta_examples],
        np.where(np.array(class_labels) == 0)[0][:cfg.img_metrics.get_raw_meta_examples]
        ]))
        
        
    if cfg.img_metrics.get_proc_meta_examples > 0:
        proc_meta_vaues_sampled_indices = np.sort(np.concatenate([
        np.where(np.array(class_labels) == 1)[0][:cfg.img_metrics.get_proc_meta_examples],
        np.where(np.array(class_labels) == 0)[0][:cfg.img_metrics.get_proc_meta_examples]
        ]))
    print("Completed. Tokens Acquired")
    
    print(f"Raw Meta values sampled Indecies: {raw_meta_vaues_sampled_indices}")
    print(f"Processed Meta values sampled Indecies: {proc_meta_vaues_sampled_indices}")
    
    

    # Get the Raw Original Image and Augment Tensor Image
    if len(mod_infer_data['orig_raw_images']) > 0:
        print('''
          \n \n
          ==================================================
                    SAVING RAW IMAGES TENSORS.....
          ==================================================
          \n \n
          '''
        )
        
        
        save_stack = list()
        for img in mod_infer_data['orig_raw_images']:
            if img != None:
                save_stack.append(img)
        save_to_pt(save_stack, "orig_image_tensors", cfg)
        
        save_stack = list()
        for img in mod_infer_data['aug_raw_images']:
            if img !=None:
                save_stack.append(img)
        save_to_pt(save_stack, "aug_image_tensors", cfg)
    
        print("RAW IMAGE SAVE COMPLETE")   
    if cfg.img_metrics.get_raw_images > 0:
        save_to_json(image_sampled_indicies, "image_sampled_indicies", cfg)
        
        
    print('''
          \n \n
          ==================================================
                        BEGINNING INFERENCE
          ==================================================
          \n \n
          '''
          )
    preds, sampled_raw_meta, proc_meta, global_token_labels = inference(model, tokenizer, mod_infer_data, raw_meta_vaues_sampled_indices, proc_meta_vaues_sampled_indices, cfg)
    
    print('''
          \n \n
          ==================================================
                        INFERENCE COMPLETE
          ==================================================
          \n \n
          '''
          )
    
    print("Saving preds to json....")
    save_to_json(preds, "preds", cfg)
    
    if cfg.img_metrics.get_token_labels > 0:
        print("Saving token labels to json...")
        save_to_json(proc_meta_vaues_sampled_indices.tolist(), "all_proc_meta_sampled_examples",cfg)
        save_to_json(class_labels, "class_labels", cfg)
        save_to_json(global_token_labels, "token_labels", cfg)
        
    if cfg.img_metrics.get_raw_meta_examples > 0:
        print("Saving raw meta values to pt......")
        save_to_json(raw_meta_vaues_sampled_indices.tolist(), "all_raw_meta_sampled_examples",cfg)
        save_to_json(class_labels, "class_labels", cfg)
        save_to_pt(sampled_raw_meta, "raw_meta_values", cfg)
    

    if cfg.img_metrics.get_proc_meta_examples > 0:
        print("Saving processed meta values to json....")
        save_to_json(proc_meta_vaues_sampled_indices.tolist(), "all_proc_meta_sampled_examples",cfg)
        save_to_json(class_labels, "class_labels", cfg)
        save_to_json(proc_meta, "processed_meta_values", cfg)
        
    print('''
          \n \n
          ==================================================
                        BEGINNING EVALUATION
          ==================================================
          \n \n
          '''
          )    
    # Evaluation
    auc, acc, auc_low = evaluate(preds, mod_infer_data["label"], "img", cfg)        

    # Save
    print("Saving evaluation results.....")
    save_to_json(auc, "auc", cfg)
    save_to_json(acc, "acc", cfg)
    save_to_json(auc_low, "auc_low", cfg)
    
    
    print('''
          \n \n
          ==================================================
                            RUN COMPLETED
          ==================================================
          \n \n
          '''
          ) 

if __name__ == "__main__":
    main()