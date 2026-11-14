#!/usr/bin/env python3

import os

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("PYTHONFAULTHANDLER", "1")

import argparse
import faulthandler
import gc
import sys
import traceback

faulthandler.enable(all_threads=True)

import torch
from peft import LoraConfig
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

from config import HF_CACHE, MODEL_NAME, OUTPUT_DIR, TRAIN_CFG
from dataset import MultiTaskDataset
from rewards import REWARD_VERSIONS

REWARD_FN = REWARD_VERSIONS["v2"]

os.environ["HF_HOME"] = HF_CACHE

from transformers import TrainerCallback


class KeepAdapterCallback(TrainerCallback):

    def __init__(self, keep_every: int):
        self.keep_every = keep_every

    def on_save(self, args, state, control, **kwargs):
        step = state.global_step
        if not self.keep_every or step % self.keep_every or not state.is_world_process_zero:
            return
        import shutil
        src = os.path.join(args.output_dir, f"checkpoint-{step}")
        dst = os.path.join(args.output_dir, f"keep-{step}")
        if not os.path.isdir(src):
            return
        os.makedirs(dst, exist_ok=True)
        for f in os.listdir(src):
            if f.startswith("adapter_") or f in ("preprocessor_config.json", "tokenizer_config.json",
                                                 "tokenizer.json", "chat_template.json", "chat_template.jinja",
                                                 "vocab.json", "merges.txt", "special_tokens_map.json"):
                shutil.copy2(os.path.join(src, f), dst)
        print(f"[keep] adapter of step {step} -> {dst}", flush=True)


class GPUMemoryCallback(TrainerCallback):
    def on_log(self, args, state, control, **kwargs):
        if not torch.cuda.is_available():
            return

        _, rank, local_rank = get_dist_info()

        alloc = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        peak_alloc = torch.cuda.max_memory_allocated() / 1024**3
        peak_reserved = torch.cuda.max_memory_reserved() / 1024**3
        free_b, total_b = torch.cuda.mem_get_info()

        print(
            f"[rank={rank} gpu={local_rank}] "
            f"step={state.global_step} | "
            f"current={alloc:.2f} GiB | "
            f"reserved={reserved:.2f} GiB | "
            f"PEAK={peak_alloc:.2f}/{peak_reserved:.2f} GiB | "
            f"free={free_b / 1024**3:.2f} GiB",
            flush=True,
        )

def get_dist_info():
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    return world_size, rank, local_rank


def is_main_process():
    _, rank, _ = get_dist_info()
    return rank == 0


def print0(*args, **kwargs):
    if is_main_process():
        kwargs.setdefault("flush", True)
        print(*args, **kwargs)


def mark(name):
    if is_main_process():
        print(f"\n>>> CHECKPOINT {name}", flush=True)


def bind_local_gpu():
    world_size, rank, local_rank = get_dist_info()

    if not torch.cuda.is_available():
        raise RuntimeError("ROCm GPU 不可用：torch.cuda.is_available() == False")

    visible = torch.cuda.device_count()
    if local_rank >= visible:
        raise RuntimeError(
            f"LOCAL_RANK={local_rank}，但当前进程只看到 {visible} 张 GPU"
        )

    torch.cuda.set_device(local_rank)

    props = torch.cuda.get_device_properties(local_rank)
    arch = getattr(props, "gcnArchName", "N/A")

    print(
        f"[DIST] rank={rank}/{world_size} "
        f"local_rank={local_rank} "
        f"visible_gpus={visible} "
        f"device={torch.cuda.current_device()} "
        f"GPU={props.name} "
        f"arch={arch}",
        flush=True,
    )

    return world_size, rank, local_rank


def disable_trl_vllm_import():
    import trl.import_utils as trl_import_utils
    trl_import_utils.is_vllm_available = lambda *args, **kwargs: False


def gpu_mem(tag):
    torch.cuda.synchronize()

    allocated = torch.cuda.memory_allocated() / 1024**3
    reserved = torch.cuda.memory_reserved() / 1024**3
    free_b, total_b = torch.cuda.mem_get_info()

    free = free_b / 1024**3
    total = total_b / 1024**3

    _, rank, local_rank = get_dist_info()

    print(
        f"[rank={rank} gpu={local_rank}] "
        f"[GPU MEM {tag}] "
        f"allocated={allocated:.2f} GiB | "
        f"reserved={reserved:.2f} GiB | "
        f"free={free:.2f}/{total:.2f} GiB",
        flush=True,
    )


_REWARD_DEBUG = int(os.environ.get("REWARD_DEBUG", "0"))
_reward_debug_count = 0


def multitask_reward_fn(completions, label, task, aux=None, view_ids=None, **kwargs):
    global _reward_debug_count
    scores = []
    if aux is None:
        aux = [None] * len(completions)
    if view_ids is None:
        view_ids = [None] * len(completions)

    for completion, lbl, tsk, ax, vids in zip(completions, label, task, aux, view_ids):
        if tsk not in REWARD_FN:
            raise KeyError(
                f"未知 task={tsk!r}，可用 task={list(REWARD_FN.keys())}"
            )

        score = REWARD_FN[tsk](
            [completion],
            labels=[lbl],
            aux=[ax],
            view_ids=[vids],
        )[0]

        if _REWARD_DEBUG and _reward_debug_count < _REWARD_DEBUG and is_main_process():
            _reward_debug_count += 1
            from rewards import _completion_text, all_components
            comps = all_components(tsk, completion, lbl, ax, vids)
            print(
                f"\n[REWARD_DEBUG {_reward_debug_count}] task={tsk} label={lbl} "
                f"aux_keys={list((ax or {}).keys())} n_views={len(vids or [])}\n"
                f"  score={score:.3f} comps={ {k: (round(v, 3) if isinstance(v, float) else v) for k, v in comps.items()} }\n"
                f"  completion={_completion_text(completion)[:800]!r}",
                flush=True,
            )

        scores.append(float(score))

    return scores


def _optional_grpo_kwargs(**kw):
    import dataclasses
    from trl import GRPOConfig
    names = {f.name for f in dataclasses.fields(GRPOConfig)}
    out = {}
    for k, v in kw.items():
        if k in names:
            out[k] = v
        else:
            print0(f"[train] WARNING: 当前 TRL 的 GRPOConfig 不支持 {k}，已忽略")
    return out


def _scale_rewards_kwarg(mode):
    import dataclasses
    from trl import GRPOConfig
    f = {x.name: x for x in dataclasses.fields(GRPOConfig)}.get("scale_rewards")
    if f is None:
        print0("[train] WARNING: 当前 TRL 不支持 scale_rewards，已忽略")
        return {}
    if isinstance(f.default, bool):
        if mode == "batch":
            print0("[train] WARNING: 旧版 TRL 不支持 batch，按 group 处理")
        return {"scale_rewards": mode != "none"}
    return {"scale_rewards": mode}


def cfg_get(cfg, key, default=None):
    if key in cfg:
        return cfg[key]
    if default is not None:
        return default
    raise KeyError(f"TRAIN_CFG 缺少必须字段: {key}")


def validate_grpo_batch(
    per_device_bs,
    num_generations,
    world_size,
    gradient_accumulation_steps,
):
    if per_device_bs < 1:
        raise ValueError(
            "per_device_train_batch_size 必须 >= 1"
        )

    if gradient_accumulation_steps < 1:
        raise ValueError(
            "gradient_accumulation_steps 必须 >= 1"
        )

    if num_generations < 2:
        raise ValueError(
            "GRPO 的 num_generations 必须 >= 2"
        )

    effective_batch_size = (
        per_device_bs
        * world_size
        * gradient_accumulation_steps
    )

    if effective_batch_size % num_generations != 0:
        valid_values = [
            n
            for n in range(2, effective_batch_size + 1)
            if effective_batch_size % n == 0
        ]

        raise ValueError(
            "GRPO batch 配置不合法："
            f"world_size={world_size}, "
            f"per_device_train_batch_size={per_device_bs}, "
            f"gradient_accumulation_steps={gradient_accumulation_steps}, "
            f"effective_batch_size={effective_batch_size}, "
            f"num_generations={num_generations}。"
            f"合法 num_generations={valid_values}"
        )

    print0(
        "[train] batch check OK: "
        f"world_size={world_size}, "
        f"per_device_batch={per_device_bs}, "
        f"gradient_accumulation_steps={gradient_accumulation_steps}, "
        f"effective_batch_size={effective_batch_size}, "
        f"num_generations={num_generations}"
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--max_steps",
        type=int,
        default=None,
        help="覆盖 TRAIN_CFG['max_steps']；例如 --max_steps 10 做快速测试",
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=None,
        help="覆盖 TRAIN_CFG 中的 per-device batch size",
    )
    parser.add_argument(
        "--num_generations",
        type=int,
        default=None,
        help="覆盖 TRAIN_CFG 中的 num_generations",
    )
    parser.add_argument(
        "--max_completion_length",
        type=int,
        default=None,
        help="覆盖 TRAIN_CFG['max_new_tokens']",
    )
    parser.add_argument(
        "--keep_every",
        type=int,
        default=0,
        help="每多少步把 adapter 另存到 keep-<step>/ 供之后在验证集上选 checkpoint（0 = 不保存）",
    )
    parser.add_argument(
        "--save_steps",
        type=int,
        default=None,
        help="每多少步保存 checkpoint（覆盖 config.TRAIN_CFG，默认 20）",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="GRPO 训练随机种子（采样 / 数据顺序等）；默认 42 = TRL 默认值，与之前的运行一致。"
             "训练子集（--data_fraction）固定用 seed 42 抽取，不受此参数影响",
    )
    parser.add_argument(
        "--report_to",
        choices=["none", "wandb"],
        default="none",
        help="默认关闭 WandB；确认训练稳定后可设 wandb",
    )
    parser.add_argument(
        "--run_name",
        type=str,
        default="cityriui-lora-multitask",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="例如 /scratch/.../multitask/checkpoint-1000；写 auto 则自动找输出目录里最新的 checkpoint",
    )
    parser.add_argument(
        "--output_name",
        type=str,
        default="multitask",
        help="输出子目录名（OUTPUT_DIR/<output_name>），不同实验用不同名字",
    )
    parser.add_argument(
        "--min_pixels",
        type=int,
        default=128 * 28 * 28,
        help="Qwen2-VL processor 最小视觉像素数",
    )
    parser.add_argument(
        "--max_pixels",
        type=int,
        default=768 * 28 * 28,
        help="Qwen2-VL processor 最大视觉像素数；默认沿用已验证配置",
    )
    parser.add_argument(
        "--reward_version",
        choices=["v1", "v2", "v3"],
        default="v2",
        help="v1 = 原版二值奖励（消融对照）；v2 = 连续 accuracy + 中间量对齐 + 一致性",
    )
    parser.add_argument("--gradient_accumulation_steps", type=int, default=None,
                        help="覆盖 TRAIN_CFG；显存不够时配合 --per_device_train_batch_size 1 使用")
    parser.add_argument("--epsilon_high", type=float, default=0.28,
                        help="DAPO clip-higher：PPO 上侧裁剪阈值（下侧仍为 0.2），缓解熵坍缩；设 0.2 即恢复对称裁剪")
    parser.add_argument("--temperature", type=float, default=1.0, help="rollout 采样温度")
    parser.add_argument("--scale_rewards", choices=["group", "batch", "none"], default="group",
                        help="advantage 是否除以组内 std；none = Dr. GRPO（v3 的 DR 类别权重需要 none 才生效）")
    parser.add_argument("--attr_weight", type=float, default=0.0,
                        help="辅助任务 Attributes 一致性奖励的权重（三个任务相同；默认 0 = 不启用）")
    parser.add_argument("--level_weight", type=float, default=0.0,
                        help="辅助任务 Levels 正确性奖励的权重（默认 0 = 不启用）")
    parser.add_argument("--aux_tasks", default="GC,DR,RIR",
                        help="对哪些任务启用 attr/lvl 奖励（逗号分隔）；其余任务这两项权重为 0。prompt/SFT 格式不变")
    parser.add_argument("--evid_max_lines", type=int, default=5,
                        help="证据行数上限（超过按比例扣 evid 分）；默认 5 = v4；全视图证据用 12")
    parser.add_argument("--outcome_mode", choices=["hit", "band"], default="hit",
                        help="v3 GC/RIR outcome：hit = 0.5·命中 + 0.5·连续（原 v3）；band = 容差内满分、容差外连续")
    parser.add_argument("--halluc_pen", type=float, default=0.0,
                        help="引用不存在的 View 时总 reward 减去 λ×无效引用比例（默认 0 = 原 v3）")
    parser.add_argument("--learning_rate", type=float, default=None,
                        help="覆盖 TRAIN_CFG 的 learning_rate")
    parser.add_argument("--zero_weights", type=str, default=None,
                        help="消融用：把这些奖励分项的权重设为 0，如 'dist,prop,income,cons'（去掉过程奖励与一致性）")
    parser.add_argument("--aux_scale", type=float, default=None,
                        help="format/evid/ground 三个辅助奖励项的权重倍数，如 0.3")
    parser.add_argument("--dr_fn_pen", type=str, default=None,
                        help="DR 漏报惩罚，如 'Black=0.5'（真值是 Black 却没答 Black 时扣分）")
    parser.add_argument("--dr_fp_pen", type=str, default=None,
                        help="DR 误报惩罚，如 'Hispanic=0.5,Asian=0.5'（答了但答错时扣分）")
    parser.add_argument("--gc_cons_weight", type=float, default=None,
                        help="GC 一致性项（答案 ≈ 由自己的分布算出的 Gini）的权重，默认 0.1")
    parser.add_argument("--gc_dist_weight", type=float, default=None,
                        help="GC 奖励里分布项(KL)的权重，默认 0.3；提高可改善 GC KL")
    parser.add_argument("--dr_class_cap", type=float, default=6.0,
                        help="DR 类别权重上限（Asian 样本极少，权重过大会导致过度预测 Asian）")
    parser.add_argument("--dr_class_power", type=float, default=None,
                        help="v3 的 DR 类别权重 ∝ 先验^(-p)：0.5=默认 inverse-sqrt，1=完全 inverse-frequency")
    parser.add_argument("--train_vision", action="store_true",
                        help="GRPO 时继续训练 adapter 里的视觉端 LoRA（默认冻结）")
    parser.add_argument("--init_adapter", type=str, default=None,
                        help="从已有 LoRA（如 sft_warmup.py 的输出）继续做 GRPO，而不是新建 LoRA")
    parser.add_argument("--gradient_checkpointing", action="store_true",
                        help="开启梯度检查点（use_reentrant=False，兼容 DDP+LoRA），显存大幅下降、速度约慢 30%%")
    parser.add_argument("--grounding", action="store_true",
                        help="启用视觉 grounding reward（需 --clip_path 与 --clip_cache）")
    parser.add_argument("--clip_path", type=str, default=None,
                        help="CLIP snapshot 路径（与 view_select.py 相同）")
    parser.add_argument("--clip_cache", type=str, default=None,
                        help="view_select.py 的 score_cache 目录")
    parser.add_argument("--data_fraction", type=float, default=0.4,
                        help="每个 task 随机取的训练数据比例（原代码写死 0.4）")
    parser.add_argument("--lora_r", type=int, default=32)
    parser.add_argument("--lora_alpha", type=int, default=64)

    args = parser.parse_args()

    global REWARD_FN
    REWARD_FN = REWARD_VERSIONS[args.reward_version]
    import rewards as _R
    if args.aux_scale is not None:
        _R.scale_aux_weights(args.aux_scale)
        print0(f"[train] aux weights x{args.aux_scale}: {_R.WEIGHTS_V2}")
    for _arg, _tbl in ((args.dr_fn_pen, _R.DR_FN_PEN), (args.dr_fp_pen, _R.DR_FP_PEN)):
        if _arg:
            for kv in _arg.split(","):
                k, v = kv.split("=")
                _tbl[k.strip().capitalize()] = float(v)
    if args.dr_fn_pen or args.dr_fp_pen:
        print0(f"[train] DR penalties  FN={_R.DR_FN_PEN}  FP={_R.DR_FP_PEN}")
    if args.gc_cons_weight is not None:
        _R.WEIGHTS_V2["GC"]["cons"] = args.gc_cons_weight
        print0(f"[train] GC cons weight -> {args.gc_cons_weight}")
    if args.gc_dist_weight is not None:
        _R.WEIGHTS_V2["GC"]["dist"] = args.gc_dist_weight
        print0(f"[train] GC weights: {_R.WEIGHTS_V2['GC']}")
    _R.EVID_MAX_LINES = args.evid_max_lines
    _aux_on = {x.strip().upper() for x in args.aux_tasks.split(",") if x.strip()}
    for _t in _R.WEIGHTS_V2:
        _on = _t.upper() in _aux_on
        _R.WEIGHTS_V2[_t]["attr"] = args.attr_weight if _on else 0.0
        _R.WEIGHTS_V2[_t]["lvl"] = args.level_weight if _on else 0.0
    if args.attr_weight or args.level_weight:
        print0(f"[train] aux task weights: attr={args.attr_weight} lvl={args.level_weight} on {sorted(_aux_on)}")
        print0(f"[train] weights: {_R.WEIGHTS_V2}")
    _R.OUTCOME_MODE = args.outcome_mode
    _R.HALLUC_PEN = args.halluc_pen
    print0(f"[train] outcome_mode={args.outcome_mode}  halluc_pen={args.halluc_pen}")
    if args.zero_weights:
        for k in [x.strip() for x in args.zero_weights.split(",") if x.strip()]:
            for tsk in _R.WEIGHTS_V2:
                if k in _R.WEIGHTS_V2[tsk]:
                    _R.WEIGHTS_V2[tsk][k] = 0.0
        print0(f"[train] ablation zero_weights={args.zero_weights}: {_R.WEIGHTS_V2}")
    if args.dr_class_power is not None:
        import rewards as _R
        print0(f"[train] DR class weights (power={args.dr_class_power}): {_R.set_dr_class_power(args.dr_class_power, cap=args.dr_class_cap)}")

    if args.grounding:
        if args.reward_version not in ("v2", "v3"):
            raise ValueError("--grounding 只能与 --reward_version v2/v3 一起用")
        if not (args.clip_path and args.clip_cache):
            raise ValueError("--grounding 需要同时提供 --clip_path 和 --clip_cache")
        import grounding
        grounding.enable(args.clip_path, args.clip_cache)

    world_size, rank, local_rank = bind_local_gpu()

    cfg = dict(TRAIN_CFG)

    max_steps = (
        args.max_steps
        if args.max_steps is not None
        else int(cfg_get(cfg, "max_steps"))
    )
    per_device_bs = (
        args.per_device_train_batch_size
        if args.per_device_train_batch_size is not None
        else int(cfg_get(cfg, "per_device_train_batch_size", 1))
    )
    num_generations = (
        args.num_generations
        if args.num_generations is not None
        else int(cfg_get(cfg, "num_generations", 2))
    )
    max_completion_length = (
        args.max_completion_length
        if args.max_completion_length is not None
        else int(cfg_get(cfg, "max_new_tokens", 32))
    )
    gradient_accumulation_steps = (
        args.gradient_accumulation_steps
        if args.gradient_accumulation_steps is not None
        else int(cfg_get(cfg, "gradient_accumulation_steps", 1))
    )
    learning_rate = float(args.learning_rate if args.learning_rate is not None
                          else cfg_get(cfg, "learning_rate", 1e-5))
    warmup_steps = int(cfg_get(cfg, "warmup_steps", 0))
    save_steps = int(args.save_steps if args.save_steps is not None else cfg_get(cfg, "save_steps", 20))
    logging_steps = int(cfg_get(cfg, "logging_steps", 1))

    validate_grpo_batch(
        per_device_bs,
        num_generations,
        world_size,
        gradient_accumulation_steps,
    )

    output_dir = os.path.join(OUTPUT_DIR, args.output_name)
    os.makedirs(output_dir, exist_ok=True)
    print0(f"[train] output_dir={output_dir}")

    disable_trl_vllm_import()
    from trl import GRPOConfig, GRPOTrainer
    import vlm_base as _vb
    if _vb.family(MODEL_NAME) not in ("qwen2", "qwen25"):
        import trl_multiimage_patch
        trl_multiimage_patch.apply()

    gc.collect()
    torch.cuda.empty_cache()

    mark("PROCESSOR")

    import vlm_base
    processor = vlm_base.load_processor(MODEL_NAME, HF_CACHE, args.min_pixels, args.max_pixels,
                                        padding_side=None)
    print0(f"[train] base family={vlm_base.family(MODEL_NAME)}")

    print0(f"[train] MODEL_NAME={MODEL_NAME}")
    print0(f"[train] visual min_pixels={args.min_pixels}")
    print0(f"[train] visual max_pixels={args.max_pixels}")

    mark("MODEL_LOAD")

    print0(f"[train] 加载模型: {MODEL_NAME}")

    model = vlm_base.load_model(MODEL_NAME, HF_CACHE)

    model.config.use_cache = False

    print(
        f"[rank={rank}] model loaded: "
        f"device={next(model.parameters()).device}, "
        f"dtype={next(model.parameters()).dtype}",
        flush=True,
    )

    if args.init_adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.init_adapter, is_trainable=True)
        if args.gradient_checkpointing:
            model.enable_input_require_grads()
        print0(f"[train] 从 LoRA 继续训练: {args.init_adapter}")
        n_vis = 0
        for n_, p_ in model.named_parameters():
            if p_.requires_grad and vlm_base.is_vision_param(n_):
                n_vis += 1
                if not args.train_vision:
                    p_.requires_grad_(False)
        if n_vis:
            print0(f"[train] adapter 含 {n_vis} 个视觉端 LoRA 参数 → "
                   f"{'继续训练（find_unused_parameters=True）' if args.train_vision else '已冻结'}")

    gpu_mem("AFTER_BASE_MODEL")

    mark("DATASET")

    import random
    from torch.utils.data import Subset


    def sample_fraction_by_task(dataset, fraction=0.4, seed=42):
        rng = random.Random(seed)

        task_to_indices = {}

        if hasattr(dataset, "datasets") and hasattr(dataset, "cumulative_sizes"):
            start = 0
            for sub, end in zip(dataset.datasets, dataset.cumulative_sizes):
                task_to_indices.setdefault(sub.task, []).extend(range(start, end))
                start = end
        else:
            for idx in range(len(dataset)):
                task_to_indices.setdefault(dataset[idx]["task"], []).append(idx)

        selected_indices = []

        for task, indices in sorted(task_to_indices.items()):
            n_total = len(indices)
            n_keep = max(1, int(n_total * fraction))

            chosen = rng.sample(indices, n_keep)
            selected_indices.extend(chosen)

            print(
                f"[DATA SAMPLE] {task}: "
                f"{n_keep}/{n_total} "
                f"({n_keep / n_total:.1%})",
                flush=True,
            )

        rng.shuffle(selected_indices)

        print(
            f"[DATA SAMPLE] total: "
            f"{len(selected_indices)}/{len(dataset)} "
            f"({len(selected_indices) / len(dataset):.1%})",
            flush=True,
        )

        return Subset(dataset, selected_indices)


    train_dataset = MultiTaskDataset(split="train")

    train_dataset = sample_fraction_by_task(
        train_dataset,
        fraction=args.data_fraction,
        seed=42,
    )
    print0(f"[train] MultiTaskDataset(train) 共 {len(train_dataset)} 条")

    if len(train_dataset) == 0:
        raise RuntimeError("MultiTaskDataset(train) 为空")

    if is_main_process():
        sample = train_dataset[0]

        if not isinstance(sample, dict):
            raise TypeError(
                f"dataset item 必须为 dict，实际为 {type(sample)}"
            )

        keys = sorted(sample.keys())
        print(f"[train] 首条样本字段: {keys}", flush=True)

        for required in ("prompt", "label", "task"):
            if required not in sample:
                raise KeyError(f"dataset 首条样本缺少 {required!r}")

        if "image" not in sample and "images" not in sample:
            raise KeyError("dataset 首条样本缺少 image/images")

    mark("LORA_CONFIG")

    lora_targets = [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ]

    if vlm_base.family(MODEL_NAME) in ("internvl", "gemma"):
        lora_targets = vlm_base.lora_targets(MODEL_NAME, False)
    import re as _re
    found_targets = []
    for name, _module in model.named_modules():
        if (_re.fullmatch(lora_targets, name) if isinstance(lora_targets, str)
                else any(name.endswith(t) for t in lora_targets)):
            found_targets.append(name)

    if not found_targets:
        raise RuntimeError(
            "模型中未找到 LoRA target modules：" + ", ".join(lora_targets)
        )

    print0(f"[train] 找到 {len(found_targets)} 个 LoRA target modules")

    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=lora_targets,
    )

    print0(
        f"[train] LoRA r={args.lora_r}, alpha={args.lora_alpha}"
    )

    mark("GRPO_CONFIG")

    num_train_epochs = float(
        cfg_get(cfg, "num_train_epochs", 1.0)
    )

    grpo_cfg = GRPOConfig(
        output_dir=output_dir,
        seed=args.seed,

        num_generations=num_generations,
        per_device_train_batch_size=per_device_bs,
        gradient_accumulation_steps=gradient_accumulation_steps,

        max_steps=max_steps,
        num_train_epochs=num_train_epochs,

        max_completion_length=max_completion_length,
        max_prompt_length=None,

        learning_rate=learning_rate,
        warmup_steps=warmup_steps,

        save_strategy="steps",
        save_steps=save_steps,
        save_total_limit=3,

        logging_steps=logging_steps,

        bf16=True,
        fp16=False,

        use_vllm=False,
        beta=0.0,
        temperature=args.temperature,
        **_optional_grpo_kwargs(epsilon_high=args.epsilon_high),
        **_scale_rewards_kwarg(args.scale_rewards),

        gradient_checkpointing=args.gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False} if args.gradient_checkpointing else None,
        ddp_find_unused_parameters=bool(args.train_vision),

        optim="adamw_torch",
        torch_compile=False,
        use_liger_kernel=False,

        dataloader_num_workers=0,
        dataloader_pin_memory=False,
        dataloader_persistent_workers=False,

        remove_unused_columns=False,

        report_to=args.report_to,
        run_name=args.run_name,
    )

    print0(
        "[train] GRPO parameters:\n"
        f"  world_size={world_size}\n"
        f"  max_steps={max_steps}\n"
        f"  per_device_batch={per_device_bs}\n"
        f"  gradient_accumulation_steps={gradient_accumulation_steps}\n"
        f"  num_generations={num_generations}\n"
        f"  max_completion_length={max_completion_length}\n"
        f"  learning_rate={learning_rate}\n"
        f"  warmup_steps={warmup_steps}\n"
        f"  save_steps={save_steps}\n"
        f"  logging_steps={logging_steps}\n"
        f"  report_to={args.report_to}\n"
        f"  reward_version={args.reward_version}  scale_rewards={args.scale_rewards}\n"
        f"  grounding={args.grounding}\n"
        f"  data_fraction={args.data_fraction}\n"
        f"  epsilon_high={args.epsilon_high}  temperature={args.temperature}"
    )

    gpu_mem("BEFORE_TRAINER")

    mark("GRPO_TRAINER_INIT")

    trainer = GRPOTrainer(
        model=model,
        processing_class=processor,
        reward_funcs=[multitask_reward_fn],
        args=grpo_cfg,
        train_dataset=train_dataset,
        peft_config=None if args.init_adapter else peft_config,
    )
    trainer.add_callback(GPUMemoryCallback())
    if args.keep_every:
        trainer.add_callback(KeepAdapterCallback(args.keep_every))

    if is_main_process():
        if hasattr(trainer.model, "print_trainable_parameters"):
            trainer.model.print_trainable_parameters()

        trainable = sum(
            p.numel() for p in trainer.model.parameters() if p.requires_grad
        )
        total = sum(p.numel() for p in trainer.model.parameters())

        print(
            f"[train] trainable={trainable:,} / total={total:,} "
            f"({100.0 * trainable / total:.4f}%)",
            flush=True,
        )

    gpu_mem("AFTER_TRAINER")

    mark("TRAIN_START")

    if args.resume_from_checkpoint == "auto":
        import glob as _glob
        ckpts = sorted(_glob.glob(os.path.join(output_dir, "checkpoint-*")),
                       key=lambda p: int(p.rsplit("-", 1)[-1]) if p.rsplit("-", 1)[-1].isdigit() else -1)
        args.resume_from_checkpoint = ckpts[-1] if ckpts else None

    if args.resume_from_checkpoint:
        print0(
            f"[train] resume_from_checkpoint={args.resume_from_checkpoint}"
        )
    else:
        print0("[train] 从头开始训练")

    torch.cuda.reset_peak_memory_stats()
    trainer.train(
        resume_from_checkpoint=args.resume_from_checkpoint
    )

    mark("TRAIN_DONE")
    peak_alloc = torch.cuda.max_memory_allocated() / 1024**3
    peak_reserved = torch.cuda.max_memory_reserved() / 1024**3

    print(
        f"[rank={rank}] PEAK GPU MEM: "
        f"allocated={peak_alloc:.2f} GiB, "
        f"reserved={peak_reserved:.2f} GiB",
        flush=True,
    )

    torch.cuda.synchronize()
    gpu_mem("END")

    trainer.save_model(output_dir)

    if is_main_process():
        processor.save_pretrained(output_dir)
        print(
            f"[train] 训练完成，LoRA 模型与 processor 已保存到：{output_dir}",
            flush=True,
        )

    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        raise
    except Exception:
        traceback.print_exc()
        sys.exit(1)
