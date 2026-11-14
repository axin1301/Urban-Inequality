import torch


def family(name: str) -> str:
    low = str(name).lower()
    if "internvl" in low:
        return "internvl"
    if "gemma" in low:
        return "gemma"
    if "qwen2.5-vl" in low or "qwen2_5_vl" in low:
        return "qwen25"
    return "qwen2"


def load_processor(name, cache_dir, min_pixels, max_pixels, padding_side="right"):
    from transformers import AutoProcessor
    fam = family(name)
    if fam == "gemma":
        proc = AutoProcessor.from_pretrained(name, cache_dir=cache_dir)
    elif fam == "internvl":
        proc = AutoProcessor.from_pretrained(name, cache_dir=cache_dir)
        ip = getattr(proc, "image_processor", None)
        if ip is not None and hasattr(ip, "crop_to_patches"):
            ip.crop_to_patches = False
    else:
        proc = AutoProcessor.from_pretrained(name, cache_dir=cache_dir,
                                             min_pixels=min_pixels, max_pixels=max_pixels)
    if padding_side is not None:
        proc.tokenizer.padding_side = padding_side
    return proc


def load_model(name, cache_dir, **kw):
    fam = family(name)
    if fam in ("internvl", "gemma"):
        from transformers import AutoModelForImageTextToText as M
    elif fam == "qwen25":
        from transformers import Qwen2_5_VLForConditionalGeneration as M
    else:
        from transformers import Qwen2VLForConditionalGeneration as M
    kw.setdefault("torch_dtype", torch.bfloat16)
    kw.setdefault("attn_implementation", "sdpa")
    kw.setdefault("low_cpu_mem_usage", True)
    return M.from_pretrained(name, cache_dir=cache_dir, **kw)


_LM = r"(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)"


def lora_targets(name, vision_lora=False):
    fam = family(name)
    if fam == "gemma":
        lm = rf".*language_model.*\.{_LM}$"
        if not vision_lora:
            return lm
        return lm + r"|.*vision_tower.*\.(q_proj|k_proj|v_proj|out_proj|fc1|fc2)$"
    if fam == "internvl":
        lm = rf".*language_model.*\.{_LM}$"
        if not vision_lora:
            return lm
        return (lm + r"|.*vision_tower.*\.(q_proj|k_proj|v_proj|projection_layer|fc1|fc2)$"
                + r"|.*multi_modal_projector\.(linear_1|linear_2)$")
    if not vision_lora:
        return ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    return (rf".*\.{_LM}$"
            r"|.*visual\.blocks\.\d+\.(attn\.qkv|attn\.proj|mlp\.fc1|mlp\.fc2)$"
            r"|.*visual\.merger\.mlp\.(0|2)$")


def is_vision_param(param_name: str) -> bool:
    return any(k in param_name for k in ("visual", "vision_tower", "multi_modal_projector"))


def assistant_marker(name) -> str:
    return "<start_of_turn>model" if family(name) == "gemma" else "<|im_start|>assistant"
