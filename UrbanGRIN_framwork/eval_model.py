#!/usr/bin/env python3
import argparse
import copy
import json
import os
import time
from collections import defaultdict

import torch
from peft import PeftModel
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

from config import HF_CACHE, MODEL_NAME
from dataset import MultiTaskDataset
from rewards import (
    REWARD_FN,
    format_reward,
    gc_distribution_reward,
    gc_accuracy_reward,
    dr_proportion_reward,
    dr_accuracy_reward,
    rir_accuracy_reward,
)


def normalize_images(sample):
    if sample.get("images") is not None:
        images = sample["images"]
    elif sample.get("image") is not None:
        images = [sample["image"]]
    else:
        return []

    if not isinstance(images, (list, tuple)):
        images = [images]
    return [x for x in images if x is not None]


def prepare_messages(prompt, images):
    if isinstance(prompt, str):
        messages = [{"role": "user", "content": prompt}]
    else:
        messages = copy.deepcopy(prompt)

    if not images:
        return messages

    image_blocks = []
    for msg in messages:
        content = msg.get("content")
        if isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and block.get("type") == "image":
                    image_blocks.append(block)

    used = 0
    for block in image_blocks:
        if used >= len(images):
            break
        if block.get("image") is None:
            block["image"] = images[used]
            used += 1

    if image_blocks and used == 0:
        return messages

    remaining = images[used:]
    if not remaining:
        return messages

    user_idx = None
    for i in range(len(messages) - 1, -1, -1):
        if messages[i].get("role") == "user":
            user_idx = i
            break
    if user_idx is None:
        raise ValueError("Prompt contains images but no user message.")

    content = messages[user_idx].get("content", "")
    extra = [{"type": "image", "image": img} for img in remaining]

    if isinstance(content, str):
        messages[user_idx]["content"] = extra + [{"type": "text", "text": content}]
    elif isinstance(content, list):
        messages[user_idx]["content"] = extra + content
    else:
        raise TypeError(f"Unsupported content type: {type(content)}")

    return messages


def generate_one(model, processor, sample, device, max_new_tokens):
    messages = prepare_messages(sample["prompt"], normalize_images(sample))

    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    image_inputs, video_inputs = process_vision_info(messages)

    inputs = processor(
        text=[text],
        images=image_inputs if image_inputs else None,
        videos=video_inputs if video_inputs else None,
        padding=True,
        return_tensors="pt",
    )

    inputs = {
        k: v.to(device) if hasattr(v, "to") else v
        for k, v in inputs.items()
    }

    with torch.inference_mode():
        generated = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=True,
        )

    prompt_len = inputs["input_ids"].shape[1]
    completion_ids = generated[:, prompt_len:]

    return processor.batch_decode(
        completion_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]


def generate_batch(model, processor, samples, device, max_new_tokens, num_samples=1, temperature=0.7):
    texts, images = [], []
    for s in samples:
        msgs = prepare_messages(s["prompt"], normalize_images(s))
        texts.append(processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True))
        im, _ = process_vision_info(msgs)
        images.extend(im or [])
    inputs = processor(text=texts, images=images or None, padding=True, return_tensors="pt")
    inputs = {k: v.to(device) if hasattr(v, "to") else v for k, v in inputs.items()}
    gkw = dict(do_sample=False) if num_samples <= 1 else dict(
        do_sample=True, temperature=temperature, top_p=0.95, num_return_sequences=num_samples)
    with torch.inference_mode():
        gen = model.generate(**inputs, max_new_tokens=max_new_tokens, use_cache=True,
                             pad_token_id=processor.tokenizer.pad_token_id, **gkw)
    outs = processor.batch_decode(gen[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True,
                                  clean_up_tokenization_spaces=False)
    k = max(1, num_samples)
    groups = [outs[i * k:(i + 1) * k] for i in range(len(samples))]
    reps = [sc_aggregate(str(s["task"]).upper(), g) for s, g in zip(samples, groups)]
    return reps, groups


DR_TRAIN_PRIOR = {"White": 0.658, "Black": 0.267, "Hispanic": 0.059, "Asian": 0.016}
DR_PRIOR_TAU = 0.0


def sc_aggregate(task, outs):
    from rewards import _extract_first_float, _extract_tags, _race_from_answer
    if len(outs) == 1:
        return outs[0]
    if task == "DR":
        votes = [(_race_from_answer(_extract_tags(o)[1]), o) for o in outs]
        valid = [v for v in votes if v[0]]
        if not valid:
            return outs[0]
        from collections import Counter
        cnt = Counter(v[0] for v in valid)
        top = max(cnt, key=lambda r: cnt[r] / (len(valid) * DR_TRAIN_PRIOR.get(r, 0.25) ** DR_PRIOR_TAU))
        return next(o for r, o in valid if r == top)
    lo, hi = (0.0, 1.0) if task == "GC" else (0.0, 2.0)
    vals = []
    for o in outs:
        p = _extract_first_float(_extract_tags(o)[1])
        if p is not None and lo <= p <= hi:
            vals.append((p, o))
    if not vals:
        return outs[0]
    med = sorted(p for p, _ in vals)[len(vals) // 2]
    return min(vals, key=lambda x: abs(x[0] - med))[1]


def score_one(task, pred, label):
    task = str(task).upper()
    reward = float(REWARD_FN[task]([pred], labels=[label])[0])
    fmt = float(format_reward([pred])[0])

    if task == "GC":
        aux = float(gc_distribution_reward([pred])[0])
        acc = float(gc_accuracy_reward([pred], labels=[label])[0])
    elif task == "DR":
        aux = float(dr_proportion_reward([pred])[0])
        acc = float(dr_accuracy_reward([pred], labels=[label])[0])
    elif task == "RIR":
        aux = None
        acc = float(rir_accuracy_reward([pred], labels=[label])[0])
    else:
        raise ValueError(f"Unknown task: {task}")

    return reward, fmt, acc, aux


def summarize(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[row["task"]].append(row)

    out = {}
    for task in ["GC", "DR", "RIR"]:
        rs = groups.get(task, [])
        if not rs:
            continue

        item = {
            "n": len(rs),
            "mean_reward": sum(x["reward"] for x in rs) / len(rs),
            "format_rate": sum(x["format"] for x in rs) / len(rs),
            "accuracy": sum(x["accuracy"] for x in rs) / len(rs),
        }

        aux = [x["aux"] for x in rs if x["aux"] is not None]
        if aux:
            key = "distribution_rate" if task == "GC" else "mean_proportion_reward"
            item[key] = sum(aux) / len(aux)

        out[task] = item

    if rows:
        out["ALL"] = {
            "n": len(rows),
            "mean_reward": sum(x["reward"] for x in rows) / len(rows),
            "format_rate": sum(x["format"] for x in rows) / len(rows),
            "accuracy": sum(x["accuracy"] for x in rows) / len(rows),
        }

    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--base_only", action="store_true", help="Evaluate the original base model without LoRA")
    ap.add_argument("--max_new_tokens", type=int, default=640)
    ap.add_argument("--max_samples", type=int, default=None)
    ap.add_argument("--min_pixels", type=int, default=128 * 28 * 28)
    ap.add_argument("--max_pixels", type=int, default=768 * 28 * 28)
    ap.add_argument("--output_dir", default=None)
    ap.add_argument("--tasks", default="GC,DR,RIR", help="只评估这些任务，如 GC 或 GC,DR（便于多卡并行）")
    ap.add_argument("--per_task", type=int, default=None, help="每个任务只评估前 N 条（快速看效果）")
    ap.add_argument("--tag", default="", help="输出文件名后缀，并行评估时避免互相覆盖")
    ap.add_argument("--skip_per_task", type=int, default=0, help="每个任务跳过前 N 条（已评估过的）")
    ap.add_argument("--resume", action="store_true", help="读取已有 predictions 文件，跳过已完成的样本，接着写")
    ap.add_argument("--num_samples", type=int, default=1,
                    help=">1 时每题采样 K 个回答做 self-consistency（GC/RIR 中位数，DR 投票）")
    ap.add_argument("--sc_temperature", type=float, default=0.7)
    ap.add_argument("--dr_prior_tau", type=float, default=0.0,
                    help="DR 投票时除以训练集类别先验^τ（logit adjustment），0 = 普通投票，常用 0.5~1")
    ap.add_argument("--split", choices=["test", "val"], default="test",
                    help="val = 从训练集留出的验证集（调参用），test = 最终报告")
    ap.add_argument("--batch_size", type=int, default=1, help="一次生成的样本数（64G 卡可试 8）")
    ap.add_argument("--shard", default=None, help="i/n：只跑第 i 份（共 n 份），多卡并行用，如 0/2 与 1/2")
    ap.add_argument("--skip_glob", default=None,
                    help="其他预测文件的 glob：启动时跳过它们已完成的样本（中途换分片数、多节点分担时用）")
    args = ap.parse_args()
    global DR_PRIOR_TAU
    DR_PRIOR_TAU = args.dr_prior_tau

    if not torch.cuda.is_available():
        raise RuntimeError("GPU not available")

    if args.base_only and args.checkpoint is not None:
        raise ValueError("Use either --base_only or --checkpoint, not both.")
    if not args.base_only and args.checkpoint is None:
        raise ValueError("Specify --base_only or --checkpoint PATH.")

    checkpoint = None
    if args.checkpoint is not None:
        checkpoint = os.path.abspath(args.checkpoint)
        if not os.path.isfile(os.path.join(checkpoint, "adapter_config.json")):
            raise FileNotFoundError(
                f"{checkpoint}/adapter_config.json not found; "
                "expected a LoRA/PEFT checkpoint."
            )

    torch.cuda.set_device(0)
    device = torch.device("cuda:0")

    if args.base_only:
        run_name = "base"
        default_output_dir = os.path.join(os.path.dirname(__file__), "outputs", "multitask")
    else:
        run_name = os.path.basename(checkpoint.rstrip("/"))
        default_output_dir = os.path.dirname(checkpoint)

    output_dir = args.output_dir or default_output_dir
    os.makedirs(output_dir, exist_ok=True)

    suffix = f"_{args.tag}" if args.tag else ""
    pred_path = os.path.join(output_dir, f"eval_{run_name}{suffix}_predictions.jsonl")
    summary_path = os.path.join(output_dir, f"eval_{run_name}{suffix}_summary.json")

    print(f"mode       = {'BASELINE' if args.base_only else 'LORA CHECKPOINT'}", flush=True)
    print(f"checkpoint = {checkpoint}", flush=True)

    processor = AutoProcessor.from_pretrained(
        MODEL_NAME,
        cache_dir=HF_CACHE,
        min_pixels=args.min_pixels,
        max_pixels=args.max_pixels,
    )
    processor.tokenizer.padding_side = "left"

    print("Loading base model...", flush=True)
    base = Qwen2VLForConditionalGeneration.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
        cache_dir=HF_CACHE,
        low_cpu_mem_usage=True,
    )
    base.config.use_cache = True

    if args.base_only:
        print("Using original base model (no LoRA adapter).", flush=True)
        model = base
    else:
        print("Loading LoRA adapter...", flush=True)
        model = PeftModel.from_pretrained(base, checkpoint, is_trainable=False)

    model = model.to(device)
    model.eval()

    print("Loading test split...", flush=True)
    ds = MultiTaskDataset(split=args.split)
    want = {t.strip().upper() for t in args.tasks.split(",") if t.strip()}
    indices, start_i = [], 0
    for sub, end in zip(ds.datasets, ds.cumulative_sizes):
        if sub.task in want:
            idx = list(range(start_i, end))
            idx = idx[args.skip_per_task:]
            if args.per_task is not None:
                idx = idx[:args.per_task]
            indices.extend(idx)
        start_i = end
    if args.max_samples is not None:
        indices = indices[:args.max_samples]
    if args.shard:
        si, sn = (int(x) for x in args.shard.split("/"))
        indices = indices[si::sn]
    n = len(indices)

    print(f"test samples = {n}/{len(ds)}  tasks={sorted(want)}", flush=True)
    if n:
        print(f"sample keys = {sorted(ds[indices[0]].keys())}", flush=True)

    rows = []
    start = time.time()

    mode = "w"
    if args.resume and os.path.isfile(pred_path):
        good = []
        with open(pred_path, encoding="utf-8") as fr:
            for line in fr:
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if r.get("error") is None:
                    good.append(r)
        with open(pred_path, "w", encoding="utf-8") as fw:
            for r in good:
                fw.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")
        finished = {r["index"] for r in good}
        rows = good
        indices = [i for i in indices if i not in finished]
        n = len(indices)
        mode = "a"
        print(f"[resume] {len(finished)} done, {n} remaining", flush=True)

    if args.skip_glob:
        import glob as _glob
        other = set()
        for fp in _glob.glob(args.skip_glob):
            if os.path.abspath(fp) == os.path.abspath(pred_path):
                continue
            for line in open(fp, encoding="utf-8"):
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if r.get("error") is None:
                    other.add(r["index"])
        before = len(indices)
        indices = [i for i in indices if i not in other]
        n = len(indices)
        print(f"[skip_glob] {before - n} done elsewhere, {n} remaining", flush=True)

    bs = max(1, args.batch_size)
    done = 0
    with open(pred_path, mode, encoding="utf-8") as f:
        for b0 in range(0, n, bs):
            chunk = indices[b0:b0 + bs]
            samples = [ds[j] for j in chunk]
            preds, errs = None, [None] * len(chunk)
            try:
                preds, allouts = generate_batch(model, processor, samples, device, args.max_new_tokens,
                                                args.num_samples, args.sc_temperature)
            except Exception as e:
                print(f"[batch fallback] {type(e).__name__}: {str(e)[:200]}", flush=True)
                torch.cuda.empty_cache()
                preds, allouts = [], [None] * len(samples)
                for k, s_ in enumerate(samples):
                    try:
                        preds.append(generate_one(model, processor, s_, device, args.max_new_tokens))
                    except Exception as e2:
                        preds.append("")
                        errs[k] = f"{type(e2).__name__}: {e2}"
            for ds_idx, sample, pred, error, allo in zip(chunk, samples, preds, errs, allouts):
                task = str(sample["task"]).upper()
                label = sample["label"]
                try:
                    reward, fmt, acc, aux = score_one(task, pred, label)
                except Exception as e:
                    reward = fmt = acc = 0.0
                    aux = None
                    error = error or f"{type(e).__name__}: {e}"
                row = {
                    "index": ds_idx,
                    "tract_id": sample.get("tract_id"),
                    "task": task,
                    "label": label,
                    "prediction": pred,
                    "reward": reward,
                    "format": fmt,
                    "accuracy": acc,
                    "aux": aux,
                    "error": error,
                }
                if args.num_samples > 1 and allo:
                    row["samples"] = allo
                rows.append(row)
                f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
            f.flush()
            done += len(chunk)
            elapsed = time.time() - start
            print(f"[{done}/{n}] task={task} {elapsed/done:.2f}s/sample  "
                  f"ETA {(n-done)*elapsed/done/60:.1f} min", flush=True)

    summary = summarize(rows)
    summary["mode"] = "base" if args.base_only else "lora"
    summary["model_name"] = MODEL_NAME
    summary["checkpoint"] = checkpoint
    summary["elapsed_sec"] = time.time() - start
    summary["max_new_tokens"] = args.max_new_tokens
    summary["min_pixels"] = args.min_pixels
    summary["max_pixels"] = args.max_pixels

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\n========== TEST SUMMARY ==========", flush=True)
    for task in ["GC", "DR", "RIR", "ALL"]:
        if task not in summary:
            continue
        print(f"\n[{task}]", flush=True)
        for k, v in summary[task].items():
            if isinstance(v, float):
                print(f"  {k:24s}: {v:.4f}", flush=True)
            else:
                print(f"  {k:24s}: {v}", flush=True)

    print(f"\npredictions: {pred_path}", flush=True)
    print(f"summary    : {summary_path}", flush=True)


if __name__ == "__main__":
    main()
