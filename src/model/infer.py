import torch
from PIL import Image
from io import BytesIO
from src.model.utils import get_parts_slices
from minigpt.minigpt4.conversation.interact import Interact

class BatchProcessor:
    def __init__(self, dataset, batch_size, eos_token_id, use_augmentation):
        self.dataset = dataset
        self.use_augmentation = use_augmentation
        self.batch_size = batch_size
        self.current_batch = 0
        self.num_batch = int(len(dataset)/self.batch_size) if len(dataset) % self.batch_size == 0 else int(len(dataset)/self.batch_size) + 1
        self.eos_token_id = eos_token_id

    def __len__(self):
        return self.num_batch

    def __iter__(self):
        return self
    
    def __next__(self):
        if self.use_augmentation:
            return self._get_augmented_batch()    
        else:
            return self._get_normal_batch()

    def _get_normal_batch(self):
        """
        Do the manual padding & Attention masks
        Set the padding to the max-size of the input_ids (of current batch)
        """

        if self.current_batch == self.num_batch:
            raise StopIteration
 
        indices = list()
        input_ids = list()
        padded_input_ids = list()
        attention_masks = list()
        image_tensors = list()
        image_sizes = list()
        prompt_0 = list()
        prompt_1 = list()
        desc_shape = list()
        
        batch_begin = self.current_batch * self.batch_size
        if self.current_batch == self.num_batch-1:
            batch_end = len(self.dataset)
        else:
            batch_end = batch_begin + self.batch_size  
        
        input_ids_len = list()
        for _idx in range(batch_begin, batch_end):
            indices.append(self.dataset[_idx]["indices"])
            input_ids.append(self.dataset[_idx]["input_ids"])
            input_ids_len.append(len(self.dataset[_idx]["input_ids"]))
            image_tensors.append(self.dataset[_idx]["image_tensors"])
            image_sizes.append(self.dataset[_idx]["image_sizes"])
            prompt_0.append(self.dataset[_idx]["prompt_0"])
            prompt_1.append(self.dataset[_idx]["prompt_1"])
            desc_shape.append(self.dataset[_idx]["desc_shape"])

        max_length = max(input_ids_len)
        del input_ids_len

        for _idx in range(len(input_ids)):
            _input_ids = input_ids[_idx]
            padding_size = max_length - len(_input_ids)
            padding = self.eos_token_id * torch.ones(padding_size, dtype=torch.long)
            padded_input_id = torch.cat([torch.tensor(_input_ids, dtype=torch.long), padding])
            attention_mask = torch.zeros(max_length, dtype=torch.long)
            attention_mask[:len(_input_ids)] = 1
            padded_input_ids.append(padded_input_id)
            attention_masks.append(attention_mask)
        self.current_batch+=1

        return {
            "indices" : indices,
            "input_ids" : torch.stack(padded_input_ids, dim=0),
            "attention_masks" : torch.stack(attention_masks, dim=0),
            "image_sizes" :  torch.tensor(image_sizes),
            "image_tensors": torch.tensor(image_tensors, dtype=torch.float16),
            "prompt_0": prompt_0,
            "prompt_1": prompt_1,
            "desc_shape": desc_shape
        }

    def _get_augmented_batch(self):
        """
        Same: Do the manual padding & Attention masks
        Set the padding to the max-size of the current input_ids
        """

        if self.current_batch == self.num_batch:
            raise StopIteration
 
        indices = list()
        input_ids = list()
        padded_input_ids = list()
        attention_masks = list()
        orig_image_tensors = list()
        image_sizes = list()
        prompt_0 = list()
        prompt_1 = list()
        desc_shape = list()
        
        batch_begin = self.current_batch * self.batch_size
        if self.current_batch == self.num_batch-1:
            batch_end = len(self.dataset)
        else:
            batch_end = batch_begin + self.batch_size  
        
        input_ids_len = list()
        aug_image_tensors = dict()
        for _idx in range(batch_begin, batch_end):
            indices.append(self.dataset[_idx]["indices"])
            input_ids.append(self.dataset[_idx]["input_ids"])
            input_ids_len.append(len(self.dataset[_idx]["input_ids"]))
            orig_image_tensors.append(self.dataset[_idx]["orig_image_tensors"])
            image_sizes.append(self.dataset[_idx]["image_sizes"])
            prompt_0.append(self.dataset[_idx]["prompt_0"])
            prompt_1.append(self.dataset[_idx]["prompt_1"])
            desc_shape.append(self.dataset[_idx]["desc_shape"])

            for k, aug_imgs in self.dataset[_idx]["aug_image_tensors"].items():
                if k not in aug_image_tensors :
                    aug_image_tensors[k] = [[] for _ in range(len(aug_imgs))]
                for _aug_idx, _aug_img in enumerate(aug_imgs):
                    aug_image_tensors[k][_aug_idx].append(_aug_img)

        for k, aug_imgs in aug_image_tensors.items():
            for _aug_idx in range(len(aug_imgs)):
                aug_image_tensors[k][_aug_idx] = torch.tensor(aug_image_tensors[k][_aug_idx], dtype=torch.float16)

        max_length = max(input_ids_len)
        del input_ids_len

        for _idx in range(len(input_ids)):
            _input_ids = input_ids[_idx]
            padding_size = max_length - len(_input_ids)
            padding = self.eos_token_id * torch.ones(padding_size, dtype=torch.long)
            padded_input_id = torch.cat([torch.tensor(_input_ids, dtype=torch.long), padding])
            attention_mask = torch.zeros(max_length, dtype=torch.long)
            attention_mask[:len(_input_ids)] = 1
            padded_input_ids.append(padded_input_id)
            attention_masks.append(attention_mask)

        self.current_batch+=1

        return {
            "indices" : indices,
            "input_ids" : torch.stack(padded_input_ids, dim=0),
            "attention_masks" : torch.stack(attention_masks, dim=0),
            "image_sizes" :  torch.tensor(image_sizes),
            "orig_image_tensors": torch.tensor(orig_image_tensors, dtype=torch.float16),
            "aug_image_tensors": aug_image_tensors,
            "prompt_0": prompt_0,
            "prompt_1": prompt_1,
            "desc_shape": desc_shape
        }

class BatchProcessor_minigpt:
    def __init__(self, dataset, batch_size, use_augmentation):
        self.dataset = dataset
        self.use_augmentation = use_augmentation
        self.batch_size = batch_size
        self.current_batch = 0
        self.num_batch = int(len(dataset)/self.batch_size) if len(dataset) % self.batch_size == 0 else int(len(dataset)/self.batch_size) + 1

    def __len__(self):
        return self.num_batch

    def __iter__(self):
        return self

    def __next__(self):
        if self.use_augmentation:
            return self._get_augmented_batch()
        else:
            return self._get_normal_batch()

    def _get_normal_batch(self):
        if self.current_batch == self.num_batch:
            raise StopIteration
        
        images = list()
        inst = list()
        desc = list()

        batch_begin = self.current_batch * self.batch_size
        if self.current_batch == self.num_batch-1:
            batch_end = len(self.dataset)
        else:
            batch_end = batch_begin + self.batch_size

        for _idx in range(batch_begin, batch_end):
            images.append(self.dataset[_idx]["raw_images"])
            inst.append(self.dataset[_idx]["inst"])
            desc.append(self.dataset[_idx]["desc"])
        self.current_batch+=1

        return {
            "images": images,
            "inst": inst,
            "desc": desc
        }

    def _get_augmented_batch(self):
        if self.current_batch == self.num_batch:
            raise StopIteration
        
        images = list()
        aug_images = list()
        inst = list()
        desc = list()

        batch_begin = self.current_batch * self.batch_size
        if self.current_batch == self.num_batch-1:
            batch_end = len(self.dataset)
        else:
            batch_end = batch_begin + self.batch_size

        for _idx in range(batch_begin, batch_end):
            _orig_img = self.dataset[_idx]["orig_images"]
            if isinstance(_orig_img, dict):
                _orig_img = Image.open(BytesIO(_orig_img["bytes"])).convert("RGB")
            images.append(_orig_img)
            _aug_img = self.dataset[_idx]["aug_images"]
            for k, v in _aug_img.items():
                for _vi in range(len(v)):
                    if isinstance(v[_vi], dict):
                        v[_vi] = Image.open(BytesIO(v[_vi]["bytes"])).convert("RGB")
            aug_images.append(_aug_img)
            inst.append(self.dataset[_idx]["inst"])
            desc.append(self.dataset[_idx]["desc"])             
        self.current_batch+=1

        return {
            "images": images,
            "aug_images": aug_images,
            "inst": inst,
            "desc": desc
        }

def mod_infer(model, dataset, cfg, tokenizer=None, vis_processor=None, chat_state=None, gpu_id=None):
    """
    Mod infer function
    Run the inference
    """
    if cfg.target_model.type == "llava":
        assert tokenizer == None
        batch_processor = BatchProcessor(dataset=dataset,
                                        batch_size=cfg.inference.batch_size,
                                        eos_token_id=tokenizer.eos_token_id,
                                        use_augmentation=cfg.inference.use_augmentation)
        
        all_results = []
        for b_idx, batch in enumerate(batch_processor):
            mix_input_ids, mix_attention_masks, target_parts = mod_infer_batch(
                model, batch, tokenizer, cfg.image_metrics.parts, cfg.inference.use_augmentation)
            all_results.append((mix_input_ids, mix_attention_masks, target_parts))

    elif cfg.target_model.type == "minigpt":
        assert vis_processor == None
        assert chat_state == None
        assert gpu_id == None
        batch_processor = BatchProcessor_minigpt(dataset=dataset,
                                                batch_size=cfg.inference.batch_size,
                                                use_augmentation=cfg.inference.use_augmentation)
        all_results = []
        for b_idx, batch in enumerate(batch_processor):
            mix_input_ids, mix_attention_masks, target_parts = mod_infer_batch_minigpt(
                model, vis_processor, batch, cfg.image_metrics.parts, chat_state, gpu_id, cfg.inference.use_augmentation)
            all_results.append((mix_input_ids, mix_attention_masks, target_parts))

    return all_results

def mod_infer_batch(model, batch, tokenizer, parts, use_augmentation):
    """
    mod_infer function.
    With the instruction, image and the descriptions,
    Do an inference using them.
    Return the batch meta metics (for all parts)

    model: target model
    batch: a batch of the dataloader from get_mod_infer_data
    tokenizer: tokenizer,
    use_augmentation : True if we use augmented
    """

    def _get_parts(input_ids, logits, attention_masks, prompt_0, prompt_1, desc_shape):
        target_parts = dict()
        labels_per_sample = list()
        for _input_ids, _logits, _attention_mask, _prompt_0, _prompt_1, _desc_shape \
            in zip(input_ids, logits, attention_masks, prompt_0, prompt_1, desc_shape):

            _img_loss_slice, _img_slice, _inst_desc, _inst, _desc = get_parts_slices(_prompt_0, _prompt_1, _desc_shape)
            _img_loss_slices = _logits[_img_loss_slice, :]
            _img_target = torch.nn.functional.softmax(_img_loss_slices, dim=-1)
            _max_indices = torch.argmax(_img_target, dim=-1)

            # tensor a: Whatever that comes before the image
            # tensor b: From the second token after image to the end
            tensor_a = torch.tensor(_prompt_0).cuda() if not isinstance(_prompt_0, torch.Tensor) else _prompt_0
            tensor_b = torch.tensor(_prompt_1[1:]).cuda() if not isinstance(_prompt_1[1:], torch.Tensor) else _prompt_1[1:]

            _mix_input_ids = torch.cat([tensor_a, _max_indices, tensor_b], dim=0)

            for p in parts:
                if p not in target_parts:
                    target_parts[p] = {"input_ids": list(), "probabilities": list(), "log_probabilities": list()}
                if p == "img":
                    _slice = _img_slice
                elif p == "inst_desp":
                    _slice = _inst_desc
                elif p == "inst":
                    _slice = _inst
                elif p == "desp":
                    _slice = _desc
                else:
                    raise ValueError(f"Not supported goal {p}")

                target_parts[p]["input_ids"].append(_mix_input_ids[_slice])                
                _slice_logits = _logits[_slice, :]
                target_parts[p]["probabilities"].append(torch.nn.functional.softmax(_slice_logits, dim=-1))
                target_parts[p]["log_probabilities"].append(torch.nn.functional.log_softmax(_slice_logits, dim=-1))
                
            # Building Total Label
            labels = [''] * _mix_input_ids.shape[0]
            labels[_img_slice] = ['img']  * len(_mix_input_ids[_img_slice])
            labels[_inst]      = ['inst'] * len(_mix_input_ids[_inst])
            labels[_desc]      = ['desc'] * len(_mix_input_ids[_desc])
            labels_per_sample.append(labels)

        return target_parts, labels_per_sample
        

    if use_augmentation:
        input_ids = batch["input_ids"].cuda()
        attention_masks = batch["attention_masks"].cuda()
        orig_image_tensors = batch["orig_image_tensors"].cuda()
        image_sizes = batch["image_sizes"].cuda()

        prompt_0 = batch["prompt_0"]
        prompt_1 = batch["prompt_1"]
        desc_shape = batch["desc_shape"]
        
        total_parts = dict()
        total_token_labels = list()

        # 1. Conduct inference using the original images
        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_masks,
                images=orig_image_tensors,
                image_sizes=image_sizes
            )

        logits = outputs.logits
        target_parts, labels_per_sample = _get_parts(input_ids, logits, attention_masks, prompt_0, prompt_1, desc_shape)
        total_parts["orig"] = [target_parts]
        total_token_labels.extend(labels_per_sample)
        
        # 2. Conduct inference using the augmented images
        for k, aug_images in batch["aug_image_tensors"].items():
            total_parts[k] = list()
            for _aug_img in aug_images:
                with torch.no_grad():
                    outputs = model(
                        input_ids=input_ids,
                        attention_mask=attention_masks,
                        images=_aug_img.cuda(),
                        image_sizes=image_sizes
                    )
                logits = outputs.logits
                target_parts, labels_per_sample = _get_parts(input_ids, logits, attention_masks, prompt_0, prompt_1, desc_shape)
                total_parts[k].append(target_parts)

        return total_parts, total_token_labels

    else:
        total_parts = dict()
        total_token_labels = list()
        

        input_ids = batch["input_ids"].cuda()
        image_tensors = batch["image_tensors"].cuda()
        attention_masks = batch["attention_masks"].cuda()
        image_sizes = batch["image_sizes"].cuda()

        prompt_0 = batch["prompt_0"]
        prompt_1 = batch["prompt_1"]
        desc_shape = batch["desc_shape"]

        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_masks,
                images=image_tensors,
                image_sizes=image_sizes
            )

        logits = outputs.logits
        target_parts, labels_per_sample = _get_parts(input_ids, logits, attention_masks, prompt_0, prompt_1, desc_shape)
        total_parts["orig"] = [target_parts]
        total_token_labels.extend(labels_per_sample)

        return total_parts, total_token_labels

def mod_infer_batch_minigpt(model, vis_processor, batch, parts, chat_state, gpu_id, use_augmentation):

    def _get_parts_from_one_sample(input_ids, logits, seg_tokens, descp_encoding):
        """
        Slice the logit of oneinto different parts
        Input should only contain 1 sample
            - Consider implementing batched part processor in future (for the better efficiency)
        """
        target_parts = dict()
        labels_per_sample = list()

        _img_slice = slice(seg_tokens[0].shape[1],-seg_tokens[1].shape[1])
        _inst_desp = slice(-seg_tokens[-1].shape[1],None)
        _inst = slice(-seg_tokens[-1].shape[1],-descp_encoding.shape[1])
        _desp = slice(-descp_encoding.shape[1],None)

        img_loss_slice = logits[0, _img_slice.start-1:_img_slice.stop-1, :]
        img_target = torch.nn.functional.softmax(img_loss_slice, dim=-1)
        max_indices = torch.argmax(img_target, axis=-1)
        _mix_input_ids = torch.cat([seg_tokens[0][0], max_indices, seg_tokens[1][0]], dim=0)

        for p in parts:
            if p not in target_parts:
                target_parts[p] = { "input_ids": list(), "probabilities": list(), "log_probabilities": list()}
            if p == "img":
                _slice = _img_slice
            elif p == "inst_desp":
                _slice = _inst_desp
            elif p == "inst":
                _slice = _inst
            elif p == "desp":
                _slice = _desp
            else:
                raise ValueError(f"Not supported goal split {p}")

            target_parts[p]["input_ids"].append(_mix_input_ids[_slice])
            logits_slice = logits[0, _slice, :]
            
            target_parts[p]["probabilities"].append(
                torch.nn.functional.softmax(logits_slice, dim=-1)
            )
            target_parts[p]["log_probabilities"].append(
                torch.nn.functional.log_softmax(logits_slice, dim=-1)
            )

        # building total label
        labels = [''] * input_ids.shape[0]
        labels[_img_slice] = ['img'] * len(_mix_input_ids[_img_slice])
        labels[_inst] = ['inst'] * len(_mix_input_ids[_inst])
        labels[_desp] = ['desc'] * len(_mix_input_ids[_desp])
        labels_per_sample.append(labels)
        
        return target_parts, labels_per_sample

    total_parts = {
        "orig":  list()   
    }

    total_token_labels = list()

    if use_augmentation:
        images = batch["images"]
        aug_images = batch["aug_images"]
        inst = batch["inst"]
        desc = batch["desc"]

        # Serialized computation
        for _image, _aug_image, _inst, _desc in zip(images, aug_images, inst, desc):
            chat = Interact(model, vis_processor, device="cuda:{}".format(gpu_id))
            _img_list = []
            _chat_state = chat_state.copy()

            # Make an inference on the augmented image            
            llm_message = chat.upload_img(_image, _chat_state, _img_list)
            chat.encode_img(_img_list)

            chat.ask(_inst, _chat_state)
            _chat_state.append_message(_chat_state.roles[1], None)
            _chat_state.append_message(_desc, None)

            outputs, input_ids, seg_tokens = chat.get_output_by_emb(
                conv=_chat_state,
                img_list = _img_list
            )

            desc_encoding = chat.model.llama_tokenizer(_desc, return_tensors="pt", add_special_tokens=False).to(chat.device).input_ids
            logits = outputs.logits
            target_parts, labels_per_sample = _get_parts_from_one_sample(input_ids, logits, seg_tokens, desc_encoding)

            if not len(total_parts["orig"]):
                total_parts["orig"].append(target_parts)
            else:
                for _part in parts:
                    for _key in total_parts["orig"][0][_part].keys():
                        total_parts["orig"][0][_part][_key].extend(target_parts[_part][_key])
            total_token_labels.extend(labels_per_sample)

            for k, aug_images in _aug_image.items():
                # For each augmentation type

                if k not in total_parts:
                    # Initialize the list for saving each setting
                    total_parts[k] = [None for _ in range(len(aug_images))]

                for _setting_idx, _aug_img in enumerate(aug_images):
                    # For each settings from the k-th augmentation
                    chat = Interact(model, vis_processor, device="cuda:{}".format(gpu_id))
                    _img_list = []
                    _chat_state = chat_state.copy()    
                    print(k, _setting_idx, type(_aug_img))
                    llm_message = chat.upload_img(_aug_img, _chat_state, _img_list)
                    chat.encode_img(_img_list)

                    chat.ask(_inst, _chat_state)
                    _chat_state.append_message(_chat_state.roles[1], None)
                    _chat_state.append_message(_desc, None)

                    outputs, input_ids, seg_tokens = chat.get_output_by_emb(
                        conv=_chat_state,
                        img_list = _img_list
                    )
                    desc_encoding = chat.model.llama_tokenizer(_desc, return_tensors="pt", add_special_tokens=False).to(chat.device).input_ids
                    logits = outputs.logits
                    target_parts, labels_per_sample = _get_parts_from_one_sample(input_ids, logits, seg_tokens, desc_encoding)
                    if total_parts[k][_setting_idx] == None:
                        total_parts[k][_setting_idx] = target_parts
                    else:
                        for _part in parts:
                            # Insert (input_ids, probabilities, log_probabilities) from each part to the corresponding setting
                            for _key in total_parts[k][_setting_idx][_part].keys():
                                total_parts[k][_setting_idx][_part][_key].extend(target_parts[_part][_key]) 
        
        return total_parts, total_token_labels

    else:
        # No augmnentation
        
        total_parts = {
            "orig": []
        }
        total_token_labels = list()

        images = batch["images"]
        inst = batch["inst"]
        desc = batch["desc"]

        # Serialized computation
        # chat = Interact(model, vis_processor, device="cuda:{}".format(gpu_id))
        for _image, _inst, _desc in zip(images, inst, desc):
            chat = Interact(model, vis_processor, device="cuda:{}".format(gpu_id))

            _img_list = []
            _chat_state = chat_state.copy()

            # Make an inference on the original image
            llm_message = chat.upload_img(_image, _chat_state, _img_list)
            chat.encode_img(_img_list)

            chat.ask(_inst, _chat_state)
            _chat_state.append_message(_chat_state.roles[1], None)
            _chat_state.append_message(_desc, None)

            outputs, input_ids, seg_tokens = chat.get_output_by_emb(
                conv=_chat_state,
                img_list = _img_list
            )

            desc_encoding = chat.model.llama_tokenizer(_desc, return_tensors="pt", add_special_tokens=False).to(chat.device).input_ids
            logits = output.logits
            target_parts, labels_per_sample = _get_parts_from_one_sample(input_ids, logits, seg_tokens, desc_encoding)

            if not len(total_parts["orig"]):
                total_parts["orig"].append(target_parts)
            else:
                for _key in total_parts["orig"][0].keys():
                    total_parts["orig"][0][_key].extend(target_parts[_key])
            total_token_labels.extend(labels_per_sample)

        return total_parts, total_token_labels
