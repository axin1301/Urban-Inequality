#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple


def _lazy_imports():
    global torch, dist, Image, tqdm
    import torch
    import torch.distributed as dist
    from PIL import Image
    try:
        from tqdm import tqdm
    except ImportError:
        def tqdm(it, **kw): return it
    return torch, dist, Image, tqdm


@dataclass
class VerifyConfig:
    gc_tol: float = 0.0
    bw_tol: float = 0.0
    max_sv_images: int = 20
    max_new_tokens: int = 64
    batch_size: int = 4
    hf_cache: Optional[str] = None
    debug: int = 0
    resume: bool = True
    save_every: int = 10


ZS_PROMPTS = {
    "GC": (
        "The images show a US census tract (first image: satellite view; "
        "remaining images: street-view photos).\n"
        "Estimate the Gini coefficient of income inequality for this tract. "
        "The value must be between 0 (perfect equality) and 1 (maximum inequality).\n\n"
        "Reply ONLY in this exact format:\n"
        "<answer>X.XX</answer>"
    ),
    "DR": (
        "The images show a US census tract (first image: satellite view; "
        "remaining images: street-view photos).\n"
        "Which racial group is DOMINANT (largest population share) in this tract?\n"
        "Choose exactly one: White, Black, Hispanic, Asian.\n\n"
        "Reply ONLY in this exact format:\n"
        "<answer>White</answer>  or  <answer>Black</answer>  "
        "or  <answer>Hispanic</answer>  or  <answer>Asian</answer>"
    ),
    "BW": (
        "The images show a US census tract (first image: satellite view; "
        "remaining images: street-view photos).\n"
        "Estimate the Black-to-White median household income ratio for this tract. "
        "The value must be between 0 and 2 (values > 1 mean Black households earn more).\n\n"
        "Reply ONLY in this exact format:\n"
        "<answer>X.XX</answer>"
    ),
}


_IMAGE_INDEX_CACHE: Dict[str, Dict[str, str]] = {}

_VALID_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def build_image_index(image_root: str) -> Dict[str, str]:
    root = str(Path(image_root).resolve())
    if root in _IMAGE_INDEX_CACHE:
        return _IMAGE_INDEX_CACHE[root]

    root_path = Path(root)
    if not root_path.is_dir():
        raise FileNotFoundError(f"--image-root does not exist: {root}")

    index: Dict[str, str] = {}
    dup = 0
    for p in root_path.rglob("*"):
        if not p.is_file() or p.suffix.lower() not in _VALID_EXTS:
            continue
        if p.name in index and index[p.name] != str(p):
            dup += 1
            continue
        index[p.name] = str(p)

    print(f"[ImageIndex] root={root}  images={len(index)}  dup_basenames={dup}", flush=True)
    if index:
        sample_keys = list(index.keys())[:3]
        print(f"[ImageIndex] sample basenames: {sample_keys}", flush=True)
    else:
        print("[ImageIndex] WARNING: no images found! Check --image-root path.", flush=True)
    _IMAGE_INDEX_CACHE[root] = index
    return index


def _clean_path(path) -> str:
    if path is None:
        return ""
    return str(path).strip().strip("'\"").replace("\\", "/")


def _basename(path: str) -> str:
    return path.replace("\\", "/").rstrip("/").rsplit("/", 1)[-1]


def resolve_image(old_path: str, image_root: str, index: Dict[str, str]):
    old_path = _clean_path(old_path)
    if not old_path:
        return None

    if os.path.isfile(old_path):
        return old_path

    name = _basename(old_path)
    if not name:
        return None

    direct = os.path.join(image_root, name)
    if os.path.isfile(direct):
        return direct

    return index.get(name)


def load_pil(path: str):
    from PIL import Image
    try:
        with Image.open(path) as img:
            return img.convert("RGB").copy()
    except Exception as e:
        print(f"[WARN] load failed: {path!r}: {e}", flush=True)
        return None


def get_images_for_record(
    rec: dict,
    image_root: str,
    index: Dict[str, str],
    max_sv: int,
) -> Tuple[Optional[object], List[object]]:
    paths = rec.get("image", [])
    if paths is None:
        paths = []
    if isinstance(paths, str):
        paths = [paths]
    paths = [_clean_path(p) for p in paths if str(p).strip()]

    sat_path = paths[0] if paths else None
    sv_paths = paths[1:] if len(paths) > 1 else []

    if not sat_path:
        return None, []

    abs_sat = resolve_image(sat_path, image_root, index)
    if abs_sat is None:
        return None, []

    sat_img = load_pil(abs_sat)
    if sat_img is None:
        return None, []

    sv_imgs = []
    for p in sv_paths:
        if len(sv_imgs) >= max_sv:
            break
        abs_p = resolve_image(p, image_root, index)
        if abs_p is None:
            continue
        img = load_pil(abs_p)
        if img is not None:
            sv_imgs.append(img)

    return sat_img, sv_imgs


_NUM_RE = re.compile(r"[+-]?(?:\d+(?:\.\d+)?|\.\d+)")

_RACE_CANONICAL = {
    "white": "White", "white alone": "White",
    "black": "Black", "african american": "Black",
    "hispanic": "Hispanic", "hispanic or latino": "Hispanic",
    "asian": "Asian",
}


def _first_float(text) -> Optional[float]:
    m = _NUM_RE.search(str(text))
    if m is None:
        return None
    try:
        v = float(m.group())
        return v if math.isfinite(v) else None
    except ValueError:
        return None


def _canonical_race(text: str) -> Optional[str]:
    low = str(text).lower()
    for key in ("white", "black", "hispanic", "asian"):
        if re.search(rf"\b{key}\b", low):
            return _RACE_CANONICAL[key]
    return None


def _gpt_text(rec: dict) -> str:
    for msg in rec.get("conversations", []):
        if str(msg.get("from", "")).lower() in {"gpt", "assistant"}:
            return str(msg.get("value", ""))
    return ""


def _parse_choice_mapping(rec: dict) -> Dict[str, str]:
    import ast as _ast
    value = rec.get("choice_list")
    if value is None:
        return {}

    if isinstance(value, dict):
        out = {}
        for k, v in value.items():
            key = str(k).strip().upper()
            race = _canonical_race(str(v))
            if key in {"A", "B", "C", "D"} and race:
                out[key] = race
        return out

    if isinstance(value, (list, tuple)):
        out = {}
        for i, v in enumerate(list(value)[:4]):
            race = _canonical_race(str(v))
            if race:
                out[chr(ord("A") + i)] = race
        return out

    text_val = str(value).strip()
    if not text_val or text_val.lower() in {"null", "none"}:
        return {}

    for parser in (json.loads, _ast.literal_eval):
        try:
            parsed = parser(text_val)
            if isinstance(parsed, (dict, list, tuple)):
                return _parse_choice_mapping({"choice_list": parsed})
        except Exception:
            pass

    out = {}
    for letter, race_text in re.findall(
        r"\b([ABCD])[.):\-]\s*(White|Black|Hispanic|Asian)\b",
        text_val, re.IGNORECASE,
    ):
        race = _canonical_race(race_text)
        if race:
            out[letter.upper()] = race
    return out


def get_gt_label(rec: dict, task: str):
    text = _gpt_text(rec)

    if task == "GC":
        for pat in [
            r"Gini\s+coefficient\s*:\s*([0-9]*\.?[0-9]+)",
            r"Gini\s*:\s*([0-9]*\.?[0-9]+)",
            r"<answer>\s*([0-9]*\.?[0-9]+)\s*</answer>",
        ]:
            m = re.search(pat, text, re.IGNORECASE)
            if m:
                v = _first_float(m.group(1))
                if v is not None and 0.0 <= v <= 1.0:
                    return v
        return None

    if task == "BW":
        for pat in [
            r"Income\s+Ratio\s*:\s*([0-9]*\.?[0-9]+)",
            r"Black[-\s]*to[-\s]*White\s+(?:Income\s+)?Ratio\s*:\s*([0-9]*\.?[0-9]+)",
            r"<answer>\s*([0-9]*\.?[0-9]+)\s*</answer>",
        ]:
            m = re.search(pat, text, re.IGNORECASE)
            if m:
                v = _first_float(m.group(1))
                if v is not None and 0.0 <= v <= 2.0:
                    return v
        return None

    if task == "DR":
        direct = _canonical_race(text.strip())
        if direct:
            return direct
        m = re.search(r"\b([ABCD])\b", text, re.IGNORECASE)
        if m:
            letter = m.group(1).upper()
            mapping = _parse_choice_mapping(rec)
            if letter in mapping:
                return mapping[letter]
            for msg in rec.get("conversations", []):
                if str(msg.get("from", "")).lower() in {"human", "user"}:
                    human = str(msg.get("value", ""))
                    for L, rt in re.findall(
                        r"\b([ABCD])[.):\-]\s*(White|Black|Hispanic|Asian)\b",
                        human, re.IGNORECASE,
                    ):
                        if L.upper() == letter:
                            return _canonical_race(rt)
        return None

    raise ValueError(f"Unknown task: {task}")


def parse_model_answer(output_text: str, task: str):
    m = re.search(r"<answer>\s*(.*?)\s*</answer>", output_text, re.DOTALL | re.IGNORECASE)
    raw = m.group(1).strip() if m else output_text.strip()

    if task in ("GC", "BW"):
        return _first_float(raw)
    if task == "DR":
        return _canonical_race(raw)
    raise ValueError(f"Unknown task: {task}")


def _gc_bucket(v: float) -> int:
    return 0 if v < 0.35 else (1 if v <= 0.50 else 2)


_GC_BUCKET_NAMES = {0: "Low(<0.35)", 1: "Medium(0.35-0.50)", 2: "High(>0.50)"}


def is_consistent(pred, gt, task: str, cfg: VerifyConfig) -> Tuple[bool, str]:
    if pred is None:
        return False, "model_parse_failed"
    if gt is None:
        return False, "gt_parse_failed"

    if task == "GC":
        p, g = float(pred), float(gt)
        if cfg.gc_tol > 0:
            diff = abs(p - g)
            if diff <= cfg.gc_tol:
                return True, f"GC ok |pred={p:.3f} gt={g:.3f} Δ={diff:.3f}|"
            return False, f"GC mismatch pred={p:.3f} gt={g:.3f} Δ={diff:.3f}>{cfg.gc_tol}"
        else:
            bp, bg = _gc_bucket(p), _gc_bucket(g)
            if bp == bg:
                return True, (
                    f"GC ok bucket={_GC_BUCKET_NAMES[bg]} "
                    f"pred={p:.3f} gt={g:.3f}"
                )
            return False, (
                f"GC bucket mismatch "
                f"pred={p:.3f}({_GC_BUCKET_NAMES[bp]}) "
                f"gt={g:.3f}({_GC_BUCKET_NAMES[bg]})"
            )

    if task == "BW":
        p, g = float(pred), float(gt)
        if cfg.bw_tol > 0:
            diff = abs(p - g)
            if diff <= cfg.bw_tol:
                return True, f"BW ok |pred={p:.3f} gt={g:.3f} Δ={diff:.3f}|"
            return False, f"BW mismatch pred={p:.3f} gt={g:.3f} Δ={diff:.3f}>{cfg.bw_tol}"
        else:
            pred_high = p > 1.0
            gt_high   = g > 1.0
            direction = "high(B>W)" if gt_high else "low(B≤W)"
            if pred_high == gt_high:
                return True, f"BW ok dir={direction} pred={p:.3f} gt={g:.3f}"
            return False, (
                f"BW dir mismatch "
                f"pred={p:.3f}({'high' if pred_high else 'low'}) "
                f"gt={g:.3f}({direction})"
            )

    if task == "DR":
        if str(pred).lower() == str(gt).lower():
            return True, f"DR ok |{pred}|"
        return False, f"DR mismatch pred={pred} gt={gt}"

    raise ValueError(task)


def load_model_and_processor(model_name: str, device, hf_cache: Optional[str]):
    import torch
    from transformers import AutoProcessor

    if hf_cache:
        os.environ.setdefault("HF_HOME", hf_cache)
        os.environ.setdefault("TRANSFORMERS_CACHE", hf_cache)

    print(f"[Stage1] Loading processor: {model_name}", flush=True)
    processor = AutoProcessor.from_pretrained(model_name)
    processor.tokenizer.padding_side = "left"

    model_lower = model_name.lower()

    if "qwen2-vl" in model_lower or "qwen2_vl" in model_lower:
        from transformers import Qwen2VLForConditionalGeneration

        dtype = torch.bfloat16
        if device == "auto":
            model = Qwen2VLForConditionalGeneration.from_pretrained(
                model_name,
                torch_dtype=dtype,
                device_map="auto",
            )
        else:
            model = Qwen2VLForConditionalGeneration.from_pretrained(
                model_name,
                torch_dtype=dtype,
            ).to(device)

    elif "qwen2.5-vl" in model_lower or "qwen2_5_vl" in model_lower:
        from transformers import Qwen2_5_VLForConditionalGeneration

        dtype = torch.bfloat16
        if device == "auto":
            model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_name,
                torch_dtype=dtype,
                device_map="auto",
            )
        else:
            model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_name,
                torch_dtype=dtype,
            ).to(device)

    else:
        from transformers import AutoModelForVision2Seq

        dtype = torch.bfloat16
        if device == "auto":
            model = AutoModelForVision2Seq.from_pretrained(
                model_name,
                torch_dtype=dtype,
                device_map="auto",
            )
        else:
            model = AutoModelForVision2Seq.from_pretrained(
                model_name,
                torch_dtype=dtype,
            ).to(device)

    model.eval()
    print(f"[Stage1] Model loaded on {device}", flush=True)
    return model, processor


def run_batch(
    messages_list: List[List[dict]],
    images_list: List[List],
    processor,
    model,
    device,
    max_new_tokens: int,
) -> List[str]:
    import torch

    texts = [
        processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        for msgs in messages_list
    ]

    image_inputs, video_inputs = [], []
    try:
        from qwen_vl_utils import process_vision_info
        for imgs in images_list:
            dummy_msgs = [{"role": "user", "content": [
                {"type": "image", "image": img} for img in imgs
            ]}]
            ii, vi = process_vision_info(dummy_msgs)
            image_inputs.extend(ii or [])
            video_inputs.extend(vi or [])
    except ImportError:
        for imgs in images_list:
            image_inputs.extend(imgs)

    proc_kwargs = dict(
        text=texts,
        return_tensors="pt",
        padding=True,
    )
    if image_inputs:
        proc_kwargs["images"] = image_inputs
    if video_inputs:
        proc_kwargs["videos"] = video_inputs

    inputs = processor(**proc_kwargs)

    if device != "auto" and not isinstance(device, str):
        inputs = {k: v.to(device) if hasattr(v, "to") else v for k, v in inputs.items()}

    generated = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=processor.tokenizer.eos_token_id,
    )

    input_len = inputs["input_ids"].shape[1]
    outputs = processor.batch_decode(
        generated[:, input_len:],
        skip_special_tokens=True,
    )
    return outputs


def verify_file(
    path: str,
    task: str,
    model,
    processor,
    image_root: str,
    image_index: Dict[str, str],
    cfg: VerifyConfig,
    out_dir: Path,
    report_lines: List[str],
    device,
    rank: int = 0,
    world_size: int = 1,
) -> dict:
    try:
        from tqdm import tqdm as _tqdm
    except ImportError:
        _tqdm = lambda it, **kw: it

    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        data = [data]

    total = len(data)

    if rank == 0:
        print(f"[Stage1/{task}] total={total}  showing first 3 image fields:", flush=True)
        for _rec in data[:3]:
            _imgs = _rec.get("image", "<missing>")
            _sid  = _rec.get("sample_id", "?")
            _imgs_preview = _imgs[:2] if isinstance(_imgs, list) else _imgs
            print(f"  id={_sid}  image={_imgs_preview!r}", flush=True)

    my_indices = list(range(rank, total, world_size))
    my_data    = [data[i] for i in my_indices]

    prompt_text = ZS_PROMPTS[task]
    stem = Path(path).stem

    suffix = f"_rank{rank}" if world_size > 1 else ""
    progress_path = out_dir / f"{stem}_progress{suffix}.jsonl"
    if not cfg.resume and progress_path.exists():
        progress_path.unlink()

    results: Dict[int, Tuple[str, bool, str, str]] = {}
    if progress_path.exists():
        with open(progress_path, encoding="utf-8") as f:
            for line in f:
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                results[int(r["idx"])] = (r["sid"], bool(r["ok"]), r["note"], r.get("raw", ""))
        print(f"[Stage1/{task}] rank={rank} resume: {len(results)} 条已完成，跳过 ({progress_path})",
              flush=True)
        with open(progress_path, "rb") as f:
            f.seek(0, os.SEEK_END)
            if f.tell() > 0:
                f.seek(-1, os.SEEK_END)
                needs_nl = f.read(1) != b"\n"
            else:
                needs_nl = False
        if needs_nl:
            with open(progress_path, "a", encoding="utf-8") as f:
                f.write("\n")

    progress_f = open(progress_path, "a", encoding="utf-8")
    pending: List[str] = []

    def save_progress() -> None:
        if pending:
            progress_f.write("".join(pending))
            progress_f.flush()
            pending.clear()

    def record(idx: int, sid: str, ok: bool, note: str, raw: str = "") -> None:
        results[idx] = (sid, ok, note, raw)
        pending.append(json.dumps(
            {"idx": idx, "sid": sid, "ok": ok, "note": note, "raw": raw},
            ensure_ascii=False) + "\n")
        if len(pending) >= max(1, cfg.save_every):
            save_progress()

    batch_items: list = []
    n_debug = [0]

    def flush_batch():
        if not batch_items:
            return
        msgs_list = [it[3] for it in batch_items]
        imgs_list = [it[4] for it in batch_items]
        try:
            outputs = run_batch(
                msgs_list, imgs_list, processor, model, device, cfg.max_new_tokens
            )
        except Exception as e:
            print(f"[WARN] batch inference failed: {e}", flush=True)
            outputs = [""] * len(batch_items)

        for (idx, sid, gt, _, _), out_text in zip(batch_items, outputs):
            pred = parse_model_answer(out_text, task)
            ok, note = is_consistent(pred, gt, task, cfg)
            record(idx, sid, ok, note, out_text[:200])

            if cfg.debug > 0 and n_debug[0] < cfg.debug:
                n_debug[0] += 1
                status = "✓ KEEP" if ok else "✗ DROP"
                print(
                    f"\n  [{status}] id={sid}\n"
                    f"    GT       : {gt}\n"
                    f"    pred     : {pred}\n"
                    f"    reason   : {note}\n"
                    f"    raw_out  : {out_text[:300]!r}",
                    flush=True,
                )

        batch_items.clear()
        del msgs_list, imgs_list, outputs

    todo = [i for i in my_indices if i not in results]
    try:
        for idx in _tqdm(todo, desc=f"[Stage1/{task}] rank={rank}", disable=(rank != 0)):
            rec = data[idx]
            sid = str(rec.get("sample_id", "?"))

            gt = get_gt_label(rec, task)
            if gt is None:
                record(idx, sid, False, "gt_parse_failed")
                continue

            sat_img, sv_imgs = get_images_for_record(
                rec, image_root, image_index, cfg.max_sv_images
            )
            if sat_img is None:
                record(idx, sid, False, "sat_image_missing")
                continue

            all_imgs = [sat_img] + sv_imgs

            content = []
            for i, img in enumerate(all_imgs):
                label = "satellite" if i == 0 else f"street view {i}"
                content.append({"type": "image", "image": img})
                content.append({"type": "text", "text": f"({label})"})
            content.append({"type": "text", "text": "\n" + prompt_text})

            msgs = [{"role": "user", "content": content}]
            batch_items.append((idx, sid, gt, msgs, all_imgs))
            del sat_img, sv_imgs, all_imgs, content, msgs

            if len(batch_items) >= cfg.batch_size:
                flush_batch()

        flush_batch()
    finally:
        save_progress()
        progress_f.close()

    details = [results[i] for i in my_indices if i in results]

    if world_size > 1:
        import torch.distributed as dist

        my_kept_pairs = [(i, results[i][1]) for i in my_indices if i in results]
        gathered = [None] * world_size
        dist.all_gather_object(gathered, my_kept_pairs)

        if rank == 0:
            idx_kept: Dict[int, bool] = {}
            for pairs in gathered:
                for idx, flag in pairs:
                    idx_kept[idx] = flag
            kept_mask = [idx_kept.get(i, False) for i in range(total)]
        else:
            kept_mask = None
    else:
        kept_mask = [results[i][1] if i in results else False for i in range(total)]

    if rank == 0:
        kept    = [rec for rec, flag in zip(data, kept_mask) if flag]
        n_kept  = len(kept)
        n_drop  = total - n_kept
        rate    = n_kept / total if total else 0.0

        out_path = out_dir / f"{stem}_stage1.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(kept, f, ensure_ascii=False, indent=2)

        n_img_fail   = sum(1 for _, ok, note, _ in details if note == "sat_image_missing")
        n_gt_fail    = sum(1 for _, ok, note, _ in details if note == "gt_parse_failed")
        n_model_fail = sum(1 for _, ok, note, _ in details if note == "model_parse_failed")
        n_mismatch   = n_drop - n_img_fail - n_gt_fail - n_model_fail

        header = (
            f"\n{'─'*64}\n"
            f"  Task       : {task}\n"
            f"  Input      : {path}\n"
            f"  Total      : {total}   Kept: {n_kept} ({rate*100:.1f}%)   Dropped: {n_drop}\n"
            f"  Output     : {out_path}\n"
            f"  Failures   : sat_missing={n_img_fail}  gt_parse={n_gt_fail}"
            f"  model_parse={n_model_fail}  mismatch={n_mismatch}"
        )
        report_lines.append(header)

        fails = [(sid, note, raw) for sid, ok, note, raw in details if not ok]
        if fails:
            report_lines.append(f"\n  FAIL details (max 200):")
            for sid, note, raw in fails[:200]:
                report_lines.append(f"    id={sid}  reason={note}  output={raw!r:.100}")
            if len(fails) > 200:
                report_lines.append(f"    ... ({len(fails)-200} more)")

        return {
            "task":      task,
            "total":     total,
            "kept":      n_kept,
            "dropped":   n_drop,
            "keep_rate": rate,
            "output":    str(out_path),
        }
    else:
        return {}


def main() -> None:
    _BASE   = os.environ.get("CITYRIUI_BASE", "/path/to/CityRiUI")
    _DATA   = _BASE + "/UI-COT-data"
    _IMGS   = _DATA + "/all_images/UI-COT-images_full"
    _MODEL  = _BASE + "/hf_cache/models--Qwen--Qwen2-VL-7B-Instruct/snapshots/latest"

    parser = argparse.ArgumentParser(
        description="Stage 1 Zero-Shot Model Verification — CityRiUI"
    )
    parser.add_argument(
        "--bw",
        default=f"{_DATA}/bw/CoT/bw_CoT_train_change_prompt.json",
        help="BW 原始训练 JSON",
    )
    parser.add_argument(
        "--dr",
        default=f"{_DATA}/dr/CoT/dr_CoT_train_change_prompt.json",
        help="DR 原始训练 JSON",
    )
    parser.add_argument(
        "--gc",
        default=f"{_DATA}/CoT_and_Original_Data/CoT_Data/gc_CoT_train.json",
        help="GC 原始训练 JSON",
    )
    parser.add_argument(
        "--image-root",
        default=_IMGS,
        help="UI-COT-images_full 目录（文件名与 JSON 内 basename 对应）",
    )
    parser.add_argument(
        "--out",
        default=f"{_BASE}/codes/outputs/stage1_filtered",
        help="输出目录（自动创建）",
    )

    parser.add_argument(
        "--model",
        default=_MODEL,
        help="HuggingFace 模型名或本地路径（默认指向 hf_cache 里缓存的快照）",
    )
    parser.add_argument(
        "--hf-cache",
        default=f"{_BASE}/hf_cache",
        help="HuggingFace 缓存目录",
    )

    parser.add_argument("--max-sv",         type=int,   default=20,
                        help="每个样本最多取几张街景图（越多越慢）")
    parser.add_argument("--max-new-tokens", type=int,   default=64)
    parser.add_argument("--batch-size",     type=int,   default=4,
                        help="每 GPU 批大小")

    parser.add_argument(
        "--gc-tol", type=float, default=0.0,
        help=(
            "GC 绝对误差阈值（默认 0 = 使用序数分桶模式：Low<0.35 / Medium≤0.50 / High>0.50）；"
            "设置 >0 时退回到 |pred−gt|<tol 判断"
        ),
    )
    parser.add_argument(
        "--bw-tol", type=float, default=0.0,
        help=(
            "BW 绝对误差阈值（默认 0 = 使用方向性匹配：pred>1 == gt>1）；"
            "设置 >0 时退回到 |pred−gt|<tol 判断"
        ),
    )

    parser.add_argument(
        "--debug", type=int, default=0, metavar="N",
        help="每个 task 打印前 N 条样本的推理细节（0=关闭）",
    )
    parser.add_argument(
        "--task", choices=["BW", "DR", "GC", "all"], default="all",
        help="只处理指定 task（默认 all）",
    )
    parser.add_argument(
        "--no-resume", action="store_true",
        help="忽略已有的 *_progress.jsonl，从头重跑（改了阈值/模型后要加）",
    )
    parser.add_argument(
        "--save-every", type=int, default=10, metavar="N",
        help="每 N 条样本把结果写入 progress jsonl 一次（被杀最多丢 N-1 条）",
    )

    args = parser.parse_args()

    _lazy_imports()
    import torch
    import torch.distributed as dist

    local_rank  = int(os.environ.get("LOCAL_RANK",  -1))
    rank        = int(os.environ.get("RANK",        0))
    world_size  = int(os.environ.get("WORLD_SIZE",  1))

    if local_rank >= 0:
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
        dist.init_process_group(backend="nccl")
        if rank == 0:
            print(f"[Stage1] torchrun mode: world_size={world_size}", flush=True)
    else:
        device = "auto"
        print("[Stage1] Single-process mode (device_map='auto')", flush=True)

    cfg = VerifyConfig(
        gc_tol          = args.gc_tol,
        bw_tol          = args.bw_tol,
        max_sv_images   = args.max_sv,
        max_new_tokens  = args.max_new_tokens,
        batch_size      = args.batch_size,
        hf_cache        = args.hf_cache,
        debug           = args.debug,
        resume          = not args.no_resume,
        save_every      = args.save_every,
    )

    model, processor = load_model_and_processor(args.model, device, cfg.hf_cache)

    image_index = build_image_index(args.image_root)

    out_dir = Path(args.out)
    if rank == 0:
        out_dir.mkdir(parents=True, exist_ok=True)
    if world_size > 1:
        dist.barrier()

    _all_tasks_and_paths = [
        (args.bw, "BW"),
        (args.dr, "DR"),
        (args.gc, "GC"),
    ]
    if args.task == "all":
        tasks_and_paths = _all_tasks_and_paths
    else:
        tasks_and_paths = [(p, t) for p, t in _all_tasks_and_paths if t == args.task]

    report_lines: List[str] = []
    if rank == 0:
        report_lines = [
            "=" * 64,
            "CityRiUI — Stage 1 Zero-Shot Model Verification Report",
            f"Model  : {args.model}",
            f"Config : {cfg}",
            "=" * 64,
        ]

    summaries = []
    for path, task in tasks_and_paths:
        if rank == 0:
            print(f"\n[Stage1] Processing {task}: {path}", flush=True)

        if world_size > 1:
            dist.barrier()

        s = verify_file(
            path        = path,
            task        = task,
            model       = model,
            processor   = processor,
            image_root  = args.image_root,
            image_index = image_index,
            cfg         = cfg,
            out_dir     = out_dir,
            report_lines= report_lines,
            device      = device,
            rank        = rank,
            world_size  = world_size,
        )

        if rank == 0 and s:
            summaries.append(s)
            print(f"         → kept {s['kept']}/{s['total']} ({s['keep_rate']*100:.1f}%)",
                  flush=True)

    if rank == 0:
        total_all = sum(s["total"] for s in summaries)
        kept_all  = sum(s["kept"]  for s in summaries)
        report_lines += [
            "\n" + "=" * 64,
            "OVERALL SUMMARY",
            f"  Model         : {args.model}",
            f"  Total samples : {total_all}",
            f"  Kept          : {kept_all}  ({kept_all/total_all*100:.1f}%)",
            f"  Dropped       : {total_all - kept_all}",
            "=" * 64,
        ]

        report_path = out_dir / "stage1_report.txt"
        with open(report_path, "w", encoding="utf-8") as f:
            f.write("\n".join(report_lines))

        print(f"\n[Stage1] Done. Report → {report_path}", flush=True)
        for line in report_lines[:6]:
            print(line)
        for s in summaries:
            print(f"  {s['task']}: kept {s['kept']}/{s['total']}  ({s['keep_rate']*100:.1f}%)")

    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
