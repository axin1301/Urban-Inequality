#!/usr/bin/env python3
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import aux_tasks as AT


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip-path", default=os.environ.get("CLIP"))
    ap.add_argument("--clip-cache", default=os.path.join(HERE, "outputs", "view_select", "score_cache"))
    ap.add_argument("--data-dir", default=os.path.join(HERE, "outputs", "view_select"))
    ap.add_argument("--max-sv", type=int, default=12)
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", default=os.path.join(HERE, "outputs", "aux_labels.json"))
    args = ap.parse_args()
    if not args.clip_path:
        sys.exit("需要 --clip-path（或 export CLIP=...）")

    os.environ["CITYRIUI_DATA_DIR"] = args.data_dir
    os.environ["CITYRIUI_MAX_SV"] = str(args.max_sv)
    os.environ.pop("CITYRIUI_AUX_LABELS", None)
    import dataset
    from aux_labels import parse_targets
    from grounding import GroundingScorer

    sc = GroundingScorer(args.clip_path, args.clip_cache, min_views=3, device=args.device)
    names = list(AT.ATTR_PROMPTS)
    T = sc.encode_text([AT.ATTR_PROMPTS[n] for n in names])
    out = {"version": AT.VERSION, "thresholds": {"attrs": {}, "levels": {}}, "labels": {}}
    print(f"[aux] version {AT.VERSION}  levels: {AT.TASK_LEVELS}", flush=True)

    for task in ("GC", "DR", "RIR"):
        rows = {}
        for split in ("train", "test"):
            sub = dataset.UICoTDataset(task=task, split="test" if split == "test" else "train")
            recs = sub.records
            if split == "train":
                vs = dataset.UICoTDataset(task=task, split="val").records
                recs = list(recs) + list(vs)
            for i, r in enumerate(recs):
                sid = str(r.get("sample_id", i))
                _, sv = sub._paths(r)
                E, valid = sc.view_matrix([os.path.basename(p) for p in sv[:args.max_sv]])
                E = E[valid]
                a = {}
                if len(E):
                    s = (E @ T.T).mean(0)
                    a = {n: float(s[j]) for j, n in enumerate(names) if n in AT.TASK_ATTRS[task]}
                _, aux = parse_targets(task, r)
                rows[sid] = (split, a, AT.level_values(task, aux))
        tr = [v for v in rows.values() if v[0] == "train"]
        thr_a = {n: list(np.quantile([v[1][n] for v in tr if n in v[1]], [1 / 3, 2 / 3]))
                 for n in AT.TASK_ATTRS[task]}
        thr_l = {}
        for n in AT.TASK_LEVELS[task]:
            xs = [v[2][n] for v in tr if v[2].get(n) is not None]
            if n in AT.FIXED_THR.get(task, {}):
                thr_l[n] = list(AT.FIXED_THR[task][n])
            elif xs:
                thr_l[n] = list(np.quantile(xs, [1 / 3, 2 / 3]))
        out["thresholds"]["attrs"][task] = thr_a
        out["thresholds"]["levels"][task] = thr_l
        lab = {}
        for sid, (_, a, lv) in rows.items():
            lab[sid] = {"attrs": {n: AT.to_level(x, thr_a[n]) for n, x in a.items()},
                        "levels": {n: AT.to_level(x, thr_l[n]) for n, x in lv.items() if n in thr_l and x is not None}}
        out["labels"][task] = lab
        n_tr = sum(v[0] == "train" for v in rows.values())
        for n in thr_l:
            from collections import Counter
            cnt = Counter(lab[sid]["levels"].get(n) for sid, v in rows.items() if v[0] == "train")
            print(f"[aux]   {task} {n}: train level counts {dict(cnt)}", flush=True)
        print(f"[aux] {task}: {len(rows)} tracts ({n_tr} train, {len(rows) - n_tr} test)  "
              f"level thresholds: " + ", ".join(f"{n}={[round(t, 3) for t in thr_l[n]]}" for n in thr_l), flush=True)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    json.dump(out, open(args.out, "w"), indent=1)
    print(f"[aux] -> {args.out}")
    ex = next(iter(out["labels"]["GC"].items()))
    print("example GC:", ex[0], AT.attr_line("GC", ex[1]["attrs"]), "|", AT.levels_line("GC", ex[1]["levels"]))
    ex = next(((k, v) for k, v in out["labels"]["DR"].items() if v["levels"]), None)
    if ex:
        print("example DR:", ex[0], AT.attr_line("DR", ex[1]["attrs"]), "|", AT.levels_line("DR", ex[1]["levels"]))


if __name__ == "__main__":
    main()
