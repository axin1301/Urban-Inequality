#!/usr/bin/env python3

from __future__ import annotations

import argparse
import glob
import json
import os
import random
import sys
import time
from pathlib import Path

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

BASE = os.environ.get("CITYRIUI_BASE", "/path/to/CityRiUI")
TASKS = ["GC", "DR", "RIR"]

EVIDENCE_PROMPT = (
    "You are an urban analyst. You are given {n_img} images of the same US census tract: the first image "
    "is the satellite view (it is NOT numbered), followed by {n_sv} street-view images numbered View 1 to "
    "View {n_sv} in order.\n"
    "Describe the visual evidence that is most informative about the neighborhood's socioeconomic "
    "conditions and demographics: housing type, size and condition, businesses and signs (including their "
    "language), vehicles, infrastructure and maintenance, people.\n"
    "Rules:\n"
    "- Write 3 to 5 lines, each exactly in the form 'View i: <one sentence>' where i is a street-view number.\n"
    "- Each sentence must describe something concrete that is visible in THAT image and differs from the "
    "other lines. Do not describe the satellite image as a View.\n"
    "- Only describe what you see. Do NOT guess income, inequality or race.\n"
    "Output only the lines."
)

EVIDENCE_PROMPT_ALL = (
    "You are an urban analyst. You are given {n_img} images of the same US census tract: the first image "
    "is the satellite view (it is NOT numbered), followed by {n_sv} street-view images numbered View 1 to "
    "View {n_sv} in order.\n"
    "For EACH street view, describe the visual evidence in it that is most informative about the "
    "neighborhood's socioeconomic conditions and demographics: housing type, size and condition, businesses "
    "and signs (including their language), vehicles, infrastructure and maintenance, people.\n"
    "Rules:\n"
    "- Write exactly {n_sv} lines, one for each street view, in order from View 1 to View {n_sv}, each exactly "
    "in the form 'View i: <one sentence>'.\n"
    "- Each sentence must describe something concrete that is visible in THAT image and differs from the "
    "other lines. Do not describe the satellite image as a View.\n"
    "- Describe only what is visible. Do not add interpretations or conclusions (for example, do not use "
    "words such as 'suggests', 'indicates', or 'shows that'), and do NOT guess income, inequality or race.\n"
    "Output only the lines."
)

RATIONALE_PROMPT = {
    "GC": "the household income distribution across brackets (LEVEL1 <$10k ... LEVEL10 >=$200k) is: {line}; "
          "the Gini coefficient is {y:.2f}",
    "DR": "the population shares are: {line}; the dominant group is {y}",
    "RIR": "{line}; the Black-to-White median household income ratio is {y:.2f}",
}
RATIONALE_TMPL = (
    "Observations of a US census tract:\n{evidence}\n\n"
    "For this tract, {fact}.\n"
    "In one or two sentences, explain which of the observations above are consistent with this. "
    "Refer only to the observations listed; do not add new observations, and do not say that you were "
    "told the values. Output only the sentences."
)


def gt_line_and_answer(task: str, label, aux: dict):
    if task == "GC":
        d = aux.get("dist")
        if not d:
            return None
        line = ", ".join(f"LEVEL{i + 1}: {p:.2f}" for i, p in enumerate(d))
        return line, f"{float(label):.2f}"
    if task == "DR":
        p = aux.get("props")
        if not p:
            return None
        line = ", ".join(f"{k}: {p[k]:.2f}" for k in ("White", "Black", "Hispanic", "Asian"))
        return line, str(label)
    if task == "RIR":
        if "black_income" not in aux:
            return None
        line = (f"Black median income: {aux['black_income']:.0f}, "
                f"White median income: {aux['white_income']:.0f}")
        return line, f"{float(label):.2f}"
    raise ValueError(task)


def clean_evidence(text: str, max_lines: int = 5):
    from grounding import parse_view_claims
    claims = parse_view_claims(text)[:max_lines]
    return "\n".join(f"View {i}: {s}." for i, s in claims), claims


def load_teacher(path: str, min_pixels: int, max_pixels: int):
    import torch
    from transformers import AutoProcessor
    low = path.lower()
    if "qwen2.5-vl" in low or "qwen2_5_vl" in low:
        from transformers import Qwen2_5_VLForConditionalGeneration as M
    else:
        from transformers import Qwen2VLForConditionalGeneration as M
    proc = AutoProcessor.from_pretrained(path, min_pixels=min_pixels, max_pixels=max_pixels)
    proc.tokenizer.padding_side = "left"
    model = M.from_pretrained(path, torch_dtype=torch.bfloat16, device_map="auto",
                              attn_implementation="sdpa").eval()
    print(f"[teacher] loaded {path}", flush=True)
    return model, proc


def generate(model, proc, content: list, images: list, max_new_tokens: int, temperature: float) -> str:
    import torch
    msgs = [{"role": "user", "content": content}]
    text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    kw = dict(text=[text], return_tensors="pt", padding=True)
    if images:
        kw["images"] = images
    inputs = proc(**kw)
    inputs = {k: v.to(model.device) if hasattr(v, "to") else v for k, v in inputs.items()}
    gen = dict(max_new_tokens=max_new_tokens, repetition_penalty=1.05,
               pad_token_id=proc.tokenizer.pad_token_id or proc.tokenizer.eos_token_id)
    if temperature > 0:
        gen.update(do_sample=True, temperature=temperature, top_p=0.9)
    else:
        gen.update(do_sample=False)
    with torch.no_grad():
        out = model.generate(**inputs, **gen)
    n = inputs["input_ids"].shape[1]
    s = proc.batch_decode(out[:, n:], skip_special_tokens=True)[0]
    del inputs, out
    return s.strip()


def generate_many(model, proc, contents: list, images_list: list, max_new_tokens: int,
                  temperature: float) -> list:
    import torch
    texts = [proc.apply_chat_template([{"role": "user", "content": c}], tokenize=False,
                                      add_generation_prompt=True) for c in contents]
    flat = [im for ims in images_list for im in ims]
    inputs = proc(text=texts, images=flat or None, return_tensors="pt", padding=True)
    inputs = {k: v.to(model.device) if hasattr(v, "to") else v for k, v in inputs.items()}
    gen = dict(max_new_tokens=max_new_tokens, repetition_penalty=1.05,
               pad_token_id=proc.tokenizer.pad_token_id or proc.tokenizer.eos_token_id)
    gen.update(dict(do_sample=True, temperature=temperature, top_p=0.9) if temperature > 0
               else dict(do_sample=False))
    with torch.no_grad():
        out = model.generate(**inputs, **gen)
    n = inputs["input_ids"].shape[1]
    res = [x.strip() for x in proc.batch_decode(out[:, n:], skip_special_tokens=True)]
    del inputs, out
    return res


def raw_files(out_dir, task):
    main = out_dir / f"raw_{task}.jsonl"
    return ([main] if main.exists() else []) + sorted(out_dir.glob(f"raw_{task}.shard*.jsonl"))


def run(args):
    import dataset as D
    import aux_labels as A

    out_dir = Path(args.out)
    model, proc = load_teacher(args.teacher, args.min_pixels, args.max_pixels)
    image_index = D._build_image_index(D.IMAGE_ROOT)

    for task in (TASKS if args.task == "all" else [args.task]):
        files = sorted(f for f in glob.glob(os.path.join(args.data_dir, task, "*.json"))
                       if "train" in os.path.basename(f).lower())
        recs = []
        for f in files:
            d = json.load(open(f, encoding="utf-8"))
            recs.extend(d if isinstance(d, list) else [d])
        order = list(range(len(recs)))
        random.Random(args.seed).shuffle(order)
        order = order[:args.n_per_task] if args.n_per_task > 0 else order
        if args.ids_file:
            want = set()
            for l in open(args.ids_file, encoding="utf-8"):
                if l.strip():
                    r_ = json.loads(l)
                    if r_.get("task") == task:
                        want.add(str(r_.get("sample_id")))
            order = [i for i in range(len(recs)) if str(recs[i].get("sample_id", i)) in want]
            if not order:
                continue

        si, sn = (int(x) for x in args.shard.split("/"))
        prog = out_dir / (f"raw_{task}.jsonl" if sn == 1 else f"raw_{task}.shard{si}of{sn}.jsonl")
        done = set()
        for pf in raw_files(out_dir, task):
            for l in open(pf, encoding="utf-8"):
                try:
                    done.add(json.loads(l)["idx"])
                except Exception:
                    pass
        todo = [i for i in order[si::sn] if i not in done]
        print(f"[{task}] records={len(recs)} selected={len(order)} done={len(done & set(order))} "
              f"todo={len(todo)}", flush=True)
        if prog.exists() and prog.stat().st_size > 0:
            with open(prog, "rb") as fb:
                fb.seek(-1, os.SEEK_END)
                if fb.read(1) != b"\n":
                    with open(prog, "a", encoding="utf-8") as fa:
                        fa.write("\n")

        t0 = time.time()
        bs = max(1, args.batch)
        n_done = 0
        with open(prog, "a", encoding="utf-8") as fh:
            for b0 in range(0, len(todo), bs):
                pend = []
                for idx in todo[b0:b0 + bs]:
                    rec = recs[idx]
                    sid = str(rec.get("sample_id", idx))
                    label, aux = A.parse_targets(task, rec)
                    gl = gt_line_and_answer(task, label, aux) if label is not None else None
                    paths = [D._clean_path(p) for p in (rec.get("image") or []) if str(p).strip()]
                    sat = D._load_image(paths[0], D.IMAGE_ROOT, image_index) if paths else None
                    row = {"idx": idx, "sid": sid, "task": task}
                    if gl is None or sat is None:
                        row["skip"] = "no_label_or_image"
                        fh.write(json.dumps(row, ensure_ascii=False) + "\n")
                        continue
                    sv, vids = [], []
                    for p in paths[1:]:
                        if len(sv) >= args.max_sv:
                            break
                        im = D._load_image(p, D.IMAGE_ROOT, image_index)
                        if im is not None:
                            sv.append(im)
                            vids.append(os.path.basename(p))
                    images = [sat] + sv
                    content = ([{"type": "image"} for _ in images]
                               + [{"type": "text", "text": (EVIDENCE_PROMPT_ALL if args.all_views else EVIDENCE_PROMPT)
                                   .format(n_img=len(images), n_sv=len(sv))}])
                    row["view_ids"] = vids
                    pend.append((row, content, images, gl, label))
                if pend:
                    try:
                        raws = generate_many(model, proc, [p[1] for p in pend], [p[2] for p in pend],
                                             args.evid_tokens, args.temperature)
                    except Exception as e:
                        print(f"[WARN] batch failed ({type(e).__name__}), fallback to 1-by-1", flush=True)
                        import torch
                        torch.cuda.empty_cache()
                        raws = []
                        for p in pend:
                            try:
                                raws.append(generate(model, proc, p[1], p[2], args.evid_tokens, args.temperature))
                            except Exception as e2:
                                print(f"[WARN] evidence failed idx={p[0]['idx']}: {e2}", flush=True)
                                raws.append("")
                    for (row, _, _, gl, label), raw in zip(pend, raws):
                        evidence, claims = clean_evidence(raw, args.max_lines)
                        rationale = ""
                        if evidence and not args.no_rationale:
                            fact = RATIONALE_PROMPT[task].format(line=gl[0], y=label)
                            try:
                                rationale = generate(model, proc,
                                                     [{"type": "text", "text": RATIONALE_TMPL.format(evidence=evidence, fact=fact)}],
                                                     [], 120, 0.0)
                            except Exception as e:
                                print(f"[WARN] rationale failed idx={row['idx']}: {e}", flush=True)
                        rationale = " ".join(rationale.split())
                        row.update({"label": label, "raw_evidence": raw,
                                    "evidence": evidence, "rationale": rationale,
                                    "gt_line": gl[0], "answer": gl[1]})
                        fh.write(json.dumps(row, ensure_ascii=False) + "\n")
                    del pend
                fh.flush()
                n_done = min(len(todo), b0 + bs)
                rate = n_done / max(time.time() - t0, 1e-6)
                print(f"[{task}] {n_done}/{len(todo)}  {1 / rate:.1f}s/sample  "
                      f"ETA {(len(todo) - n_done) / rate / 60:.0f} min", flush=True)


def filter_and_report(args):
    from rewards import evidence_quality
    out_dir = Path(args.out)
    scorer = None
    if args.clip_path:
        import grounding
        scorer = grounding.GroundingScorer(args.clip_path, args.clip_cache)
    lines = ["=" * 64, "Visual-CoT distillation report",
             f"min_ground={args.min_ground}  grounding={'on' if scorer else 'OFF (no --clip-path)'}", "=" * 64]
    kept_all = []
    for task in TASKS:
        files = raw_files(out_dir, task)
        if not files:
            continue
        by_idx = {}
        for p in files:
            for l in open(p, encoding="utf-8"):
                try:
                    r = json.loads(l)
                except json.JSONDecodeError:
                    continue
                by_idx[r.get("idx")] = r
        rows = [r for r in by_idx.values() if "evidence" in r]
        n_ev = n_q = n_g = 0
        gs = []
        for r in rows:
            if not r["evidence"]:
                continue
            n_ev += 1
            q = evidence_quality(r["evidence"], max_lines=args.max_lines)
            if q < 1.0:
                continue
            if args.min_coverage > 0:
                import re as _re
                cited = {int(x) for x in _re.findall(r"View\s*(\d+)\s*:", r["evidence"])}
                kv = len(r.get("view_ids") or [])
                if kv == 0 or len({c for c in cited if 1 <= c <= kv}) / kv < args.min_coverage:
                    continue
            n_q += 1
            g = scorer.score(r["evidence"], r["view_ids"]) if scorer else None
            if g is not None:
                gs.append(g)
            if scorer and (g is None or g < args.min_ground):
                continue
            n_g += 1
            rat = r.get("rationale") or ""
            if rat and args.rationale_sentences > 0:
                import re as _re
                rat = " ".join(_re.split(r"(?<=[.!?])\s+", rat.strip())[:args.rationale_sentences])
            if args.rationale_sentences == 0:
                rat = ""
            body = r["evidence"] + ("\n" + rat if rat else "") + "\n" + r["gt_line"]
            kept_all.append({"task": task, "sample_id": r["sid"], "idx": r["idx"], "label": r.get("label"),
                             "target": f"<think>{body}</think>\n<answer>{r['answer']}</answer>",
                             "ground": g, "evid": q})
        import statistics as st
        lines += ["", f"[{task}] generated={len(rows)}  parsed_evidence={n_ev}  "
                      f"evidence_quality==1: {n_q}  kept: {n_g}"]
        labs = [k["label"] for k in kept_all if k["task"] == task]
        if task == "DR" and labs:
            from collections import Counter
            lines.append("  kept label distribution: " + ", ".join(f"{a}={b}" for a, b in Counter(labs).most_common()))
        elif labs:
            import statistics as _st
            lines.append(f"  kept label mean/std: {_st.mean(labs):.3f}/{_st.pstdev(labs):.3f}")
        if gs:
            qs = sorted(gs)
            lines.append(f"  grounding p25/50/75 = {qs[len(qs) // 4]:.3f}/{st.median(qs):.3f}/{qs[3 * len(qs) // 4]:.3f}")
    with open(out_dir / "sft_visual_cot.jsonl", "w", encoding="utf-8") as f:
        for k in kept_all:
            f.write(json.dumps(k, ensure_ascii=False) + "\n")
    lines.append(f"\nTOTAL kept for SFT: {len(kept_all)}  -> {out_dir / 'sft_visual_cot.jsonl'}")
    if kept_all:
        lines += ["", "example target:", kept_all[0]["target"]]
    text = "\n".join(lines) + "\n"
    (out_dir / "report.txt").write_text(text, encoding="utf-8")
    print(text)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher", help="教师 VLM snapshot（推荐 Qwen2.5-VL-32B-Instruct）")
    ap.add_argument("--data-dir", default=f"{BASE}/codes/outputs/view_select")
    ap.add_argument("--task", choices=TASKS + ["all"], default="all")
    ap.add_argument("--n-per-task", type=int, default=500)
    ap.add_argument("--max-sv", type=int, default=12)
    ap.add_argument("--min-pixels", type=int, default=128 * 28 * 28)
    ap.add_argument("--max-pixels", type=int, default=512 * 28 * 28)
    ap.add_argument("--temperature", type=float, default=0.0, help="证据生成温度；0 = greedy")
    ap.add_argument("--no-rationale", action="store_true", help="不生成衔接句（只有证据 + GT 中间量）")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batch", type=int, default=1, help="一次生成多少条证据（32B 跨 2 卡可试 8）")
    ap.add_argument("--clip-path", default=None, help="CLIP snapshot（与 view_select 相同）；不给则不做 grounding 过滤")
    ap.add_argument("--clip-cache", default=f"{BASE}/codes/outputs/view_select/score_cache")
    ap.add_argument("--min-ground", type=float, default=0.3, help="grounding 分阈值")
    ap.add_argument("--rationale-sentences", type=int, default=1,
                    help="衔接句只保留前 N 句（0 = 不要衔接句）；只影响过滤输出，不需重跑教师")
    ap.add_argument("--all-views", action="store_true",
                    help="要求教师为每张选中的街景各写一行（View 1..K_n），而不是 3-5 行")
    ap.add_argument("--max-lines", type=int, default=5, help="保留 / 允许的证据行数上限（全视图模式用 12）")
    ap.add_argument("--evid-tokens", type=int, default=256, help="证据生成的 max_new_tokens（全视图模式用 640）")
    ap.add_argument("--min-coverage", type=float, default=0.0,
                    help="过滤：引用的不同视图占全部视图的最低比例（全视图模式建议 0.8）")
    ap.add_argument("--report-only", action="store_true")
    ap.add_argument("--ids-file", default=None,
                    help="jsonl（含 task, sample_id）：只为这些区域生成证据，忽略 --n-per-task")
    ap.add_argument("--shard", default="0/1",
                    help="i/n：只处理未完成样本中的第 i 份（共 n 份），如两组卡分别用 0/2 和 1/2；可随时停下改分片数重跑")
    ap.add_argument("--out", default=f"{BASE}/codes/outputs/visual_cot")
    args = ap.parse_args()
    Path(args.out).mkdir(parents=True, exist_ok=True)
    if not args.report_only:
        if not args.teacher:
            sys.exit("需要 --teacher")
        run(args)
    filter_and_report(args)


if __name__ == "__main__":
    main()
