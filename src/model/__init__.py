from src.model.generate import generate, generate_a_batch
from src.model.infer import mod_infer_batch
from llava.mm_utils import get_model_name_from_path
from llava.model.builder import load_pretrained_model

from minigpt.minigpt4.common.registry import registry
from minigpt4.conversation.interact import Interact, CONV_VISION_Vicuna0, CONV_VISION_LLama2


def load_target_model(cfg):
    """
    cfg: target model config
    """
    if cfg.target_model.type == "llava":
        model_name = get_model_name_from_path(cfg.model_path)

        tokenizer, model, image_processor, context_len = load_pretrained_model(
            cfg.target_model.model_path, 
            cfg.target_model.model_base, 
            model_name, 
            gpu_id=cfg.target_model.gpu_id,
            cache_dir=cfg.path.cache_dir
        )
        conv_mode = load_conversation_template(model_name)

        return model, tokenizer,

    elif cfg.target_model.type == "minigpt":
        model_config = cfg.target_model.model_cfg
        model_config.device_8bit = args.gpu_id
        model_cls = registry.get_model_class(model_config.arch)
        model = model_cls.from_config(model_config).to('cuda:{}'.format(args.gpu_id))

        CONV_VISION = conv_dict[model_config.model_type]

        vis_processor_cfg = cfg.datasets_cfg.cc_sbu_align.vis_processor.train
        vis_processor = registry.get_processor_class(
            vis_processor_cfg.name).from_config(vis_processor_cfg)

        chat = Interact(model, vis_processor)


        
    else:
        raise ValueError(f"Unable to recognize the model {cfg.type}")        