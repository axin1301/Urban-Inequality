#!/usr/bin/env python3

from __future__ import annotations

import argparse
import copy
import glob
import json
import os
import random
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

BASE = os.environ.get("CITYRIUI_BASE", "/path/to/CityRiUI")
DATA = BASE + "/UI-COT-data"

TASK_FILES = [
    ("GC",  f"{DATA}/CoT_and_Original_Data/CoT_Data/gc_CoT_train.json"),
    ("DR",  f"{DATA}/dr/CoT/dr_CoT_train_change_prompt.json"),
    ("RIR", f"{DATA}/bw/CoT/bw_CoT_train_change_prompt.json"),
]

VALID_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

INFO_PROMPTS = [
    "a street view photo of residential houses",
    "a street view photo of apartment buildings",
    "a street view photo of single-family homes with front yards",
    "a street view photo of shops and storefronts",
    "a street view photo of a commercial street with business signs",
    "a street view photo of cars parked in front of houses",
    "a street view photo of a residential neighborhood with sidewalks",
    "a street view photo of run-down or boarded-up buildings",
    "a street view photo of a school, church or public building",
    "a street view photo of an industrial area with warehouses",
    "a street view photo of mobile homes in a trailer park",
    "a street view photo of a gas station or fast food restaurant",
    "a street view photo of a tall roadside business sign",
    "a street view photo of a parking lot in front of a store",
]
TASK_INFO_PROMPTS = {
    "DR": [
        "a street view photo of a storefront with Spanish-language signs",
        "a street view photo of a Mexican or Latin American grocery store or restaurant",
        "a street view photo of a shop with Chinese, Korean or Vietnamese signs",
        "a street view photo of an ethnic restaurant or market",
        "a street view photo of people walking on a sidewalk",
        "a street view photo of a church or place of worship",
        "a street view photo of a colorful mural on a wall",
        "a street view photo of a barbershop, hair salon or beauty supply store",
        "a street view photo of a laundromat or check-cashing store",
        "a street view photo of public housing apartment blocks",
    ],
    "GC": [
        "a street view photo of a large luxury house",
        "a street view photo of a run-down house next to a well-kept house",
        "a street view photo of a dollar store or pawn shop",
        "a street view photo of new upscale apartments",
    ],
    "RIR": [
        "a street view photo of a large luxury house",
        "a street view photo of public housing apartment blocks",
        "a street view photo of a dollar store or pawn shop",
        "a street view photo of a well-maintained suburban street",
    ],
}

NOINFO_PROMPTS = [
    "a street view photo of an empty road with nothing around it",
    "a street view photo of a highway",
    "a street view photo of dense trees and bushes",
    "a street view photo of mostly sky",
    "a blurry street view photo",
    "a street view photo taken inside a tunnel",
    "a street view photo of an empty field with no buildings",
    "a street view photo of a close-up wall or fence",
    "a very dark street view photo",
    "a street view photo of a road under a bridge",
]


def to_test_path(train_path: str) -> str:
    d, name = os.path.split(train_path)
    return os.path.join(d, name.replace("train", "test"))


def build_index(root: str) -> Dict[str, str]:
    index: Dict[str, str] = {}
    for dirpath, _, files in os.walk(root):
        for f in files:
            if os.path.splitext(f)[1].lower() in VALID_EXTS:
                index.setdefault(f, os.path.join(dirpath, f))
    print(f"[Index] {root}: {len(index)} images", flush=True)
    return index


def clean(p) -> str:
    return str(p).strip().strip("'\"") if p is not None else ""


def get_paths(rec) -> List[str]:
    paths = rec.get("image", []) or []
    if isinstance(paths, str):
        paths = [paths]
    return [clean(p) for p in paths if clean(p)]


def load_json(path: str) -> list:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return data if isinstance(data, list) else [data]


def iter_files(limit: int):
    for task, train_path in TASK_FILES:
        for split, p in (("train", train_path), ("test", to_test_path(train_path))):
            if not os.path.isfile(p):
                print(f"[WARN] 文件不存在，跳过: {p}", flush=True)
                continue
            recs = load_json(p)
            if limit > 0:
                recs = recs[:limit]
            yield task, split, p, recs


def _as_tensor(x):
    import torch
    if isinstance(x, torch.Tensor):
        return x
    for attr in ("image_embeds", "text_embeds", "pooler_output"):
        v = getattr(x, attr, None)
        if isinstance(v, torch.Tensor):
            return v
    return x[0]


def load_cache(cache_dir: Path) -> Tuple[Dict[str, int], np.ndarray, np.ndarray]:
    names, embs, ps = [], [], []
    for f in sorted(cache_dir.glob("shard_*.npz")):
        try:
            z = np.load(f, allow_pickle=False)
        except Exception as e:
            print(f"[WARN] 损坏的缓存分片，忽略: {f} ({e})", flush=True)
            continue
        names.extend(z["names"].tolist())
        embs.append(z["emb"])
        ps.append(z["p_info"])
    if not names:
        return {}, np.zeros((0, 1), np.float16), np.zeros((0,), np.float32)
    emb = np.concatenate(embs)
    p = np.concatenate(ps)
    name2row = {n: i for i, n in enumerate(names)}
    return name2row, emb, p


def class_embeddings(model, proc, dev, is_siglip, info_prompts=None):
    import torch
    with torch.no_grad():
        cls = []
        for prompts in (info_prompts or INFO_PROMPTS, NOINFO_PROMPTS):
            t = proc(text=prompts, return_tensors="pt",
                     padding="max_length" if is_siglip else True, truncation=True)
            t = {k: v.to(dev) for k, v in t.items()}
            e = _as_tensor(model.get_text_features(**t)).float()
            e = e / e.norm(dim=-1, keepdim=True)
            m = e.mean(0)
            cls.append(m / m.norm())
        cls = torch.stack(cls)
        scale = float(model.logit_scale.exp()) if hasattr(model, "logit_scale") else 100.0
    return cls, scale


def recompute_p_info(model_path: str, emb: np.ndarray, info_prompts=None) -> Optional[np.ndarray]:
    try:
        import torch
        from transformers import AutoModel, AutoProcessor
        model = AutoModel.from_pretrained(model_path).eval()
        proc = AutoProcessor.from_pretrained(model_path)
    except Exception as e:
        print(f"[Select] 无法加载 {model_path} 重算 p_info，沿用缓存值: {e}", flush=True)
        return None
    is_siglip = "siglip" in str(getattr(model.config, "model_type", "")).lower()
    cls, scale = class_embeddings(model, proc, "cpu", is_siglip, info_prompts)
    e = torch.from_numpy(emb.astype(np.float32))
    e = e / e.norm(dim=-1, keepdim=True)
    p = torch.softmax(scale * e @ cls.T, dim=-1)[:, 0].numpy()
    del model
    return p.astype(np.float32)


def score_images(args, basenames: List[str], index: Dict[str, str], cache_dir: Path):
    import torch
    from PIL import Image
    from transformers import AutoModel, AutoProcessor

    cache_dir.mkdir(parents=True, exist_ok=True)
    done, _, _ = load_cache(cache_dir)
    todo = [b for b in basenames if b not in done and b in index]
    n_missing = sum(1 for b in basenames if b not in index)
    print(f"[Score] unique street views={len(basenames)}  cached={len(done)}  "
          f"todo={len(todo)}  not_found_on_disk={n_missing}", flush=True)
    if not todo:
        return

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if dev == "cuda" else torch.float32
    print(f"[Score] loading {args.model} on {dev}", flush=True)
    model = AutoModel.from_pretrained(args.model, torch_dtype=dtype).to(dev).eval()
    proc = AutoProcessor.from_pretrained(args.model)
    is_siglip = "siglip" in str(getattr(model.config, "model_type", "")).lower()

    cls, scale = class_embeddings(model, proc, dev, is_siglip)

    def _load(b):
        try:
            with Image.open(index[b]) as im:
                return b, im.convert("RGB")
        except Exception as e:
            print(f"[WARN] load failed {b}: {e}", flush=True)
            return b, None

    shard_id = len(list(cache_dir.glob("shard_*.npz")))
    buf_n, buf_e, buf_p = [], [], []
    t0 = time.time()
    n_done = 0

    def dump():
        nonlocal shard_id, buf_n, buf_e, buf_p
        if not buf_n:
            return
        out = cache_dir / f"shard_{shard_id:05d}.npz"
        tmp = cache_dir / f"shard_{shard_id:05d}.tmp.npz"
        np.savez(tmp, names=np.array(buf_n), emb=np.concatenate(buf_e).astype(np.float16),
                 p_info=np.concatenate(buf_p).astype(np.float32))
        os.replace(tmp, out)
        shard_id += 1
        buf_n, buf_e, buf_p = [], [], []

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for s in range(0, len(todo), args.batch_size):
            chunk = todo[s:s + args.batch_size]
            loaded = [(b, im) for b, im in pool.map(_load, chunk) if im is not None]
            if not loaded:
                continue
            names = [b for b, _ in loaded]
            imgs = [im for _, im in loaded]
            with torch.no_grad():
                px = proc(images=imgs, return_tensors="pt")["pixel_values"].to(dev, dtype)
                e = _as_tensor(model.get_image_features(pixel_values=px)).float()
                e = e / e.norm(dim=-1, keepdim=True)
                p = torch.softmax(scale * e @ cls.T, dim=-1)[:, 0]
            del imgs, loaded, px
            buf_n.extend(names)
            buf_e.append(e.cpu().numpy())
            buf_p.append(p.cpu().numpy())
            n_done += len(names)
            if sum(len(x) for x in buf_e) >= args.shard_size:
                dump()
            if (s // args.batch_size) % 20 == 0:
                rate = n_done / max(time.time() - t0, 1e-6)
                eta = (len(todo) - n_done) / max(rate, 1e-6) / 60
                print(f"[Score] {n_done}/{len(todo)}  {rate:.0f} img/s  ETA {eta:.1f} min", flush=True)
    dump()
    print(f"[Score] done in {(time.time() - t0) / 60:.1f} min", flush=True)


def select_views(emb: np.ndarray, p: np.ndarray, k: int, lam: float,
                 info_thr: float, dup_thr: float, min_keep: int) -> List[int]:
    n = len(p)
    if n == 0:
        return []
    sims = emb.astype(np.float32) @ emb.astype(np.float32).T
    selected: List[int] = []
    max_sim = np.zeros(n, np.float32)
    avail = np.ones(n, bool)
    while len(selected) < min(k, n):
        allow = avail & (max_sim <= dup_thr)
        if len(selected) >= min_keep:
            allow &= p >= info_thr
        if not allow.any():
            break
        val = lam * p - (1 - lam) * max_sim
        val[~allow] = -np.inf
        i = int(np.argmax(val))
        selected.append(i)
        avail[i] = False
        max_sim = np.maximum(max_sim, sims[i])
    return sorted(selected)


def label_summary(task: str, recs: list) -> str:
    try:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from stage1_verify import get_gt_label
    except Exception:
        return "n/a"
    t = {"GC": "GC", "DR": "DR", "RIR": "BW"}[task]
    ys = [get_gt_label(r, t) for r in recs]
    ys = [y for y in ys if y is not None]
    if not ys:
        return "n/a"
    if t == "DR":
        c = Counter(ys)
        return ", ".join(f"{k}={v / len(ys) * 100:.1f}%" for k, v in c.most_common())
    a = np.array(ys, float)
    q = np.percentile(a, [10, 50, 90])
    return f"mean={a.mean():.3f} std={a.std():.3f} p10/50/90={q[0]:.3f}/{q[1]:.3f}/{q[2]:.3f}"


def make_viz(rec, sat_path, sv_paths, p_all, sel_set, index, out_png: Path, info_thr: float = 0.5,
             thumb=160):
    from PIL import Image, ImageDraw
    items = [(sat_path, None, True)] + [(sp, p_all.get(i), i in sel_set)
                                       for i, sp in enumerate(sv_paths)]
    cols = 7
    rows = (len(items) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * (thumb + 8), rows * (thumb + 24)), "white")
    d = ImageDraw.Draw(sheet)
    for j, (path, p, sel) in enumerate(items):
        x, y = (j % cols) * (thumb + 8), (j // cols) * (thumb + 24)
        b = os.path.basename(path)
        try:
            with Image.open(index[b]) as im:
                im = im.convert("RGB")
                im.thumbnail((thumb, thumb))
                sheet.paste(im, (x + 4, y + 4))
        except Exception:
            pass
        if j == 0:
            color, label = (40, 90, 200), "satellite"
        elif p is None:
            color, label = (150, 150, 150), "not found"
        else:
            if sel:
                color, tag = (30, 160, 60), "  kept"
            elif p >= info_thr:
                color, tag = (230, 140, 20), "  info"
            else:
                color, tag = (210, 50, 50), "  low"
            label = f"p={p:.2f}" + tag
        d.rectangle([x + 2, y + 2, x + thumb + 6, y + thumb + 6], outline=color, width=4)
        d.text((x + 4, y + thumb + 8), label, fill=color)
    sheet.save(out_png)


def run_select(args, index: Dict[str, str], cache_dir: Path, out_dir: Path):
    name2row, EMB, P = load_cache(cache_dir)
    print(f"[Select] cached embeddings: {len(name2row)}", flush=True)
    if not name2row:
        sys.exit("[Select] 没有缓存的打分结果，请先运行阶段 1（去掉 --skip-score）")

    newP = recompute_p_info(args.model, EMB)
    if newP is not None:
        P = newP
        print("[Select] p_info 已按当前 prompt 重算", flush=True)
    P_task = {}
    if args.task_prompts:
        want = {x.strip().upper() for x in args.task_prompts.split(",") if x.strip()}
        for t, extra in TASK_INFO_PROMPTS.items():
            if t not in want:
                continue
            pt = recompute_p_info(args.model, EMB, INFO_PROMPTS + extra)
            if pt is not None:
                P_task[t] = pt
                print(f"[Select] {t}: 用 {len(INFO_PROMPTS)}+{len(extra)} 个任务相关 prompt 重算 p_info", flush=True)

    if args.info_quantile > 0:
        args.info_thr = float(np.quantile(P, args.info_quantile))
        print(f"[Select] info_thr = {args.info_thr:.4f}  (p_info 的 {args.info_quantile:.0%} 分位数)",
              flush=True)

    rng = random.Random(0)
    lines = ["=" * 72, "View selection report",
             f"model={args.model}", f"k={args.k} lambda={args.lam} info_thr={args.info_thr} "
             f"dup_thr={args.dup_thr} min_keep={args.min_keep} "
             f"min_informative(train only)={args.min_informative}", "=" * 72]
    all_p = P.astype(np.float32)
    hist, edges = np.histogram(all_p, bins=10)
    lines.append("p_info 分布（全部唯一街景）:")
    lines += [f"  [{edges[i]:.3f},{edges[i + 1]:.3f}) {hist[i]:>8d}" for i in range(10)]
    lines.append(f"  p_info >= {args.info_thr}: {(all_p >= args.info_thr).mean() * 100:.1f}%")

    P_base = P
    for task, split, path, recs in iter_files(args.limit):
        P = P_task.get(task, P_base)
        if args.info_quantile > 0:
            args.info_thr = float(np.quantile(P, args.info_quantile))
        tdir = out_dir / task
        tdir.mkdir(parents=True, exist_ok=True)
        stem = Path(path).stem
        n_sv_before = n_sv_after = 0
        n_inf_hist: Counter = Counter()
        nn_sims: List[float] = []
        kept, meta = [], []
        n_drop = 0
        viz_left = args.viz if split == "train" else 0
        viz_ids = set(rng.sample(range(len(recs)), min(viz_left, len(recs))))

        for ri, rec in enumerate(recs):
            paths = get_paths(rec)
            sid = str(rec.get("sample_id", ri))
            if not paths:
                kept.append(rec)
                meta.append({"sid": sid, "note": "no_image_field"})
                continue
            sat, svs = paths[0], paths[1:]
            cand = [(i, name2row[os.path.basename(sp)]) for i, sp in enumerate(svs)
                    if os.path.basename(sp) in name2row]
            idx_local = [i for i, _ in cand]
            rows = [r for _, r in cand]
            emb = EMB[rows] if rows else np.zeros((0, EMB.shape[1]), np.float16)
            p = P[rows] if rows else np.zeros((0,), np.float32)
            if len(rows) >= 2:
                e32 = emb.astype(np.float32)
                e32 = e32 / np.linalg.norm(e32, axis=1, keepdims=True)
                sm = e32 @ e32.T
                np.fill_diagonal(sm, -1)
                nn_sims.extend(sm.max(1).tolist())
            sel_local = select_views(emb, p, args.k, args.lam, args.info_thr,
                                     args.dup_thr, args.min_keep)
            if args.sort_by_info:
                sel_local = sorted(sel_local, key=lambda j: -float(p[j]))
            sel = [idx_local[j] for j in sel_local]
            n_inf = int(sum(p[j] >= args.info_thr for j in sel_local))
            n_inf_hist[min(n_inf, args.k)] += 1
            n_sv_before += min(len(svs), args.orig_max_sv) if args.orig_max_sv > 0 else len(svs)
            n_sv_after += len(sel)

            drop = split == "train" and n_inf < args.min_informative
            p_map = {idx_local[j]: float(p[j]) for j in range(len(idx_local))}
            meta.append({"sid": sid, "n_sv": len(svs), "n_scored": len(idx_local),
                         "n_informative_selected": n_inf, "dropped": drop,
                         "selected": [os.path.basename(svs[i]) for i in sel],
                         "p_info": {os.path.basename(svs[i]): round(v, 4) for i, v in p_map.items()}})
            if ri in viz_ids:
                (out_dir / "viz").mkdir(exist_ok=True)
                make_viz(rec, sat, svs, p_map, set(sel), index,
                         out_dir / "viz" / f"{task}_{split}_{sid}.png", info_thr=args.info_thr)
            if drop:
                n_drop += 1
                continue
            new = copy.deepcopy(rec)
            new["image"] = [sat] + [svs[i] for i in sel]
            kept.append(new)

        with open(tdir / f"{stem}.json", "w", encoding="utf-8") as f:
            json.dump(kept, f, ensure_ascii=False, indent=2)
        with open(tdir / f"{stem}_meta.jsonl", "w", encoding="utf-8") as f:
            for m in meta:
                f.write(json.dumps(m, ensure_ascii=False) + "\n")

        n = len(recs)
        avg_b = n_sv_before / max(n, 1)
        avg_a = n_sv_after / max(n, 1)
        red = 1 - (1 + avg_a) / (1 + avg_b) if avg_b else 0
        lines += ["", f"[{task}/{split}] {path}",
                  f"  samples            : {n}  -> kept {len(kept)}  (dropped {n_drop})",
                  f"  street views/sample: {avg_b:.2f} -> {avg_a:.2f}   "
                  f"(含卫星图的图像数降幅 ≈ {red * 100:.1f}%)",
                  "  已选信息视图数分布 : " + "  ".join(f"{i}:{n_inf_hist.get(i, 0)}"
                                               for i in range(args.k + 1))]
        if nn_sims:
            q = np.percentile(nn_sims, [10, 25, 50, 75, 90])
            lines.append("  区域内最近邻 cos    : p10/25/50/75/90 = "
                         + "/".join(f"{x:.3f}" for x in q)
                         + f"   (当前 dup_thr={args.dup_thr}，超过它的视图对视为重复)")
        if split == "train" and args.min_informative > 0:
            lines.append(f"  标签分布 before    : {label_summary(task, recs)}")
            lines.append(f"  标签分布 after     : {label_summary(task, kept)}")
        print("\n".join(lines[-6:]), flush=True)

    (out_dir / "viewsel_report.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\n[Done] report -> {out_dir / 'viewsel_report.txt'}", flush=True)


def main():
    ap = argparse.ArgumentParser(description="CLIP-based informative & diverse street-view selection")
    ap.add_argument("--model", default="openai/clip-vit-large-patch14",
                    help="CLIP / SigLIP 模型名或本地 snapshot 路径")
    ap.add_argument("--image-root", default=f"{DATA}/all_images/UI-COT-images_full")
    ap.add_argument("--out", default=f"{BASE}/codes/outputs/view_select")
    ap.add_argument("--limit", type=int, default=0, help="每个 JSON 只处理前 N 条（试跑用）；0=全部")
    ap.add_argument("--skip-score", action="store_true", help="跳过阶段 1，只用缓存重跑选择")
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--workers", type=int, default=8, help="读图线程数（建议 = cpus-per-task）")
    ap.add_argument("--shard-size", type=int, default=4096, help="每多少张图写一个缓存分片")
    ap.add_argument("--sort-by-info", action="store_true",
                    help="选中的视图按信息分从高到低排列（默认保持原顺序）")
    ap.add_argument("--task-prompts", default="",
                    help="对这些任务加入额外的'有信息'prompt，如 DR 或 DR,GC,RIR（空 = 不启用）")
    ap.add_argument("--k", type=int, default=8, help="每个样本最多保留的街景数")
    ap.add_argument("--lam", type=float, default=0.7, help="MMR 中信息量 vs 多样性的权重")
    ap.add_argument("--info-thr", type=float, default=0.5,
                    help="p_info >= 阈值 视为信息视图（仅当 --info-quantile 0 时生效）")
    ap.add_argument("--info-quantile", type=float, default=0.3,
                    help="按全体街景 p_info 的该分位数设阈值（最低 30%% 视为无信息）；0=用 --info-thr")
    ap.add_argument("--dup-thr", type=float, default=0.95, help="与已选视图 cos 超过此值视为重复")
    ap.add_argument("--min-keep", type=int, default=3,
                    help="至少保留几张（信息视图不够时用低分视图补位）")
    ap.add_argument("--min-informative", type=int, default=0,
                    help="仅 train：已选信息视图 < m 的样本整条丢弃；0=不丢样本")
    ap.add_argument("--orig-max-sv", type=int, default=10,
                    help="报告里 'before' 的街景数按原训练设置截断（config.py max_sv_images）")
    ap.add_argument("--viz", type=int, default=0, help="每个 train 文件随机画 N 个样本的选择结果")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = out_dir / "score_cache"
    index = build_index(args.image_root)

    if not args.skip_score:
        basenames = []
        seen = set()
        for _, _, _, recs in iter_files(args.limit):
            for rec in recs:
                for sp in get_paths(rec)[1:]:
                    b = os.path.basename(sp)
                    if b not in seen:
                        seen.add(b)
                        basenames.append(b)
        score_images(args, basenames, index, cache_dir)

    run_select(args, index, cache_dir, out_dir)


if __name__ == "__main__":
    main()
