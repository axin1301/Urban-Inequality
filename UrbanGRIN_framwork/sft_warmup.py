#!/usr/bin/env python3

import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import torch
from torch.utils.data import Dataset


class SFTDataset(Dataset):

    def __init__(self, sft_path: str, class_weighted: bool = False, max_per_task: int = 0):
        from collections import Counter
        from dataset import UICoTDataset
        self.class_weighted = class_weighted
        targets, cnt = {}, Counter()
        with open(sft_path, encoding="utf-8") as f:
            for l in f:
                if l.strip():
                    r = json.loads(l)
                    if max_per_task and cnt[r["task"]] >= max_per_task:
                        continue
                    cnt[r["task"]] += 1
                    targets[(r["task"], str(r["sample_id"]))] = r["target"]
        print(f"[SFT] targets per task: {dict(cnt)}", flush=True)
        self.items = []
        self.sources = {}
        for task in sorted({t for t, _ in targets}):
            ds = UICoTDataset(task=task, split="train")
            self.sources[task] = ds
            for i, rec in enumerate(ds.records):
                key = (task, str(rec.get("sample_id", i)))
                if key in targets:
                    self.items.append((task, i, targets[key]))
        print(f"[SFT] matched {len(self.items)}/{len(targets)} targets to training samples", flush=True)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, k):
        task, i, target = self.items[k]
        s = self.sources[task][i]
        w = 1.0
        if self.class_weighted and task == "DR":
            from rewards import DR_CLASS_W
            w = DR_CLASS_W.get(str(s.get("label")).strip().capitalize(), 1.0)
        return {"prompt": s["prompt"], "images": s["images"], "target": target, "weight": w}


class Collator:
    def __init__(self, processor, max_len: int):
        self.p = processor
        self.max_len = max_len
        self._asst = None

    def _messages(self, ex, with_answer: bool):
        text = ex["prompt"][0]["content"]
        msgs = [{"role": "user",
                 "content": [{"type": "image"} for _ in ex["images"]] + [{"type": "text", "text": text}]}]
        if with_answer:
            msgs.append({"role": "assistant", "content": [{"type": "text", "text": ex["target"]}]})
        return msgs

    def __call__(self, batch):
        fulls, imgs = [], []
        for ex in batch:
            fulls.append(self.p.apply_chat_template(self._messages(ex, True), tokenize=False))
            imgs.extend(ex["images"])
        enc = self.p(text=fulls, images=imgs, return_tensors="pt", padding=True)
        labels = enc["input_ids"].clone()
        labels[enc["attention_mask"] == 0] = -100
        if self._asst is None:
            import vlm_base
            from config import MODEL_NAME
            self._asst = self.p.tokenizer(vlm_base.assistant_marker(MODEL_NAME), add_special_tokens=False)["input_ids"]
        a = torch.tensor(self._asst)
        L = len(self._asst)
        for b in range(labels.shape[0]):
            ids = enc["input_ids"][b]
            win = ids.unfold(0, L, 1)
            hit = (win == a).all(1).nonzero()
            if not len(hit):
                raise RuntimeError("SFT collator: 找不到回答起点标记（对话模板与 vlm_base.assistant_marker 不一致）")
            cut = int(hit[-1]) + L + 1
            labels[b, :cut] = -100
        enc["labels"] = labels
        enc["sample_weight"] = torch.tensor([float(ex.get("weight", 1.0)) for ex in batch])
        return enc


def _make_trainer_cls():
    from transformers import Trainer

    class WeightedTrainer(Trainer):

        def compute_loss(self, model, inputs, return_outputs=False, **kw):
            w = inputs.pop("sample_weight", None)
            labels = inputs.pop("labels")
            out = model(**inputs)
            tgt = labels[:, 1:]
            mask = tgt != -100
            sel = out.logits[:, :-1][mask].float()
            ce = torch.nn.functional.cross_entropy(sel, tgt[mask], reduction="none")
            row = mask.nonzero()[:, 0]
            nb = tgt.shape[0]
            per = torch.zeros(nb, device=ce.device).index_add_(0, row, ce) / \
                mask.sum(1).clamp(min=1).to(ce.device)
            if w is not None:
                w = w.to(per.device, per.dtype)
                loss = (per * w).sum() / w.sum().clamp(min=1e-6)
            else:
                loss = per.mean()
            return (loss, out) if return_outputs else loss

    return WeightedTrainer


def main():
    from peft import LoraConfig, get_peft_model
    from transformers import AutoProcessor, Qwen2VLForConditionalGeneration, TrainingArguments
    from config import HF_CACHE, MODEL_NAME, OUTPUT_DIR

    ap = argparse.ArgumentParser()
    ap.add_argument("--sft-data", required=True)
    ap.add_argument("--output_name", default="sft_visual_cot")
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--max_steps", type=int, default=-1)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--per_device_train_batch_size", type=int, default=1)
    ap.add_argument("--gradient_accumulation_steps", type=int, default=8)
    ap.add_argument("--min_pixels", type=int, default=128 * 28 * 28)
    ap.add_argument("--max_pixels", type=int, default=512 * 28 * 28)
    ap.add_argument("--lora_r", type=int, default=32)
    ap.add_argument("--lora_alpha", type=int, default=64)
    ap.add_argument("--gradient_checkpointing", action="store_true")
    ap.add_argument("--max_per_task", type=int, default=0,
                    help="每个任务最多用多少条 SFT 样本（各任务预算一致，便于论文说明）；0 = 全部")
    ap.add_argument("--num_workers", type=int, default=4, help="DataLoader 进程数（读图/预处理与 GPU 并行）")
    ap.add_argument("--resume", action="store_true", help="从 output 目录里最新的 checkpoint 继续")
    ap.add_argument("--vision_lora", action="store_true",
                    help="同时给视觉编码器(qkv/proj/fc1/fc2)和 merger 加 LoRA，让视觉端适配街景")
    ap.add_argument("--class_weighted", action="store_true",
                    help="DR 样本 loss 乘类别权重（White .62 / Black .97 / Hispanic 2.05 / Asian 3.95）")
    args = ap.parse_args()

    out_dir = os.path.join(OUTPUT_DIR, args.output_name)
    os.makedirs(out_dir, exist_ok=True)

    import vlm_base
    print(f"[SFT] base={MODEL_NAME}  family={vlm_base.family(MODEL_NAME)}", flush=True)
    processor = vlm_base.load_processor(MODEL_NAME, HF_CACHE, args.min_pixels, args.max_pixels, "right")
    model = vlm_base.load_model(MODEL_NAME, HF_CACHE)
    model.config.use_cache = False
    if args.gradient_checkpointing:
        model.enable_input_require_grads()

    targets = vlm_base.lora_targets(MODEL_NAME, args.vision_lora)
    model = get_peft_model(model, LoraConfig(
        r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=0.05, bias="none",
        task_type="CAUSAL_LM", target_modules=targets))
    model.print_trainable_parameters()

    train_ds = SFTDataset(args.sft_data, class_weighted=args.class_weighted, max_per_task=args.max_per_task)
    targs = TrainingArguments(
        output_dir=out_dir, num_train_epochs=args.epochs, max_steps=args.max_steps,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.lr, lr_scheduler_type="cosine", warmup_ratio=0.05,
        bf16=True, logging_steps=5, save_strategy="steps", save_steps=100, save_total_limit=2,
        gradient_checkpointing=args.gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False} if args.gradient_checkpointing else None,
        ddp_find_unused_parameters=False, remove_unused_columns=False,
        dataloader_num_workers=args.num_workers,
        dataloader_persistent_workers=args.num_workers > 0, report_to="none")
    trainer = _make_trainer_cls()(model=model, args=targs, train_dataset=train_ds,
                      data_collator=Collator(processor, max_len=8192))
    ckpt = None
    if args.resume:
        import glob as _g
        cks = sorted(_g.glob(os.path.join(out_dir, "checkpoint-*")), key=lambda p: int(p.rsplit("-", 1)[-1]))
        ckpt = cks[-1] if cks else None
        print(f"[SFT] resume from {ckpt}", flush=True)
    trainer.train(resume_from_checkpoint=ckpt)
    trainer.save_model(out_dir)
    if trainer.is_world_process_zero():
        processor.save_pretrained(out_dir)
        print(f"[SFT] LoRA adapter saved to {out_dir}  →  train_grpo.py --init_adapter {out_dir}", flush=True)


if __name__ == "__main__":
    main()
