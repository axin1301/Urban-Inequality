#!/usr/bin/env python3
import argparse
import glob
import importlib
import json
import os
import re
import sys
import warnings
from collections import defaultdict

import numpy as np

warnings.filterwarnings("ignore", category=RuntimeWarning)
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from rewards import (_extract_first_float, _extract_tags, _race_from_answer,
                     parse_pred_dr_props, parse_pred_gc_dist, parse_pred_incomes)
from aux_labels import gini_from_dist

VIEW_IDX_RE = re.compile(r"\bview\s*#?\s*(\d{1,2})\b", re.I)
TASKS = ("GC", "DR", "RIR")


def load_rows(pattern):
    rows, seen = [], set()
    for f in sorted(glob.glob(pattern)):
        for l in open(f, encoding="utf-8"):
            try:
                r = json.loads(l)
            except json.JSONDecodeError:
                continue
            if r.get("error") is not None or (r["task"], r["index"]) in seen:
                continue
            seen.add((r["task"], r["index"]))
            rows.append(r)
    return rows


_VIEWS = {}


def views_for(data_dir):
    if data_dir in _VIEWS:
        return _VIEWS[data_dir]
    os.environ["CITYRIUI_DATA_DIR"] = data_dir
    os.environ["CITYRIUI_MAX_SV"] = "20"
    import config
    importlib.reload(config)
    import dataset
    importlib.reload(dataset)
    ds = dataset.MultiTaskDataset(split="test")
    out, st = {}, 0
    for sub, end in zip(ds.datasets, ds.cumulative_sizes):
        for k, rec in enumerate(sub.records):
            _, sv = sub._paths(rec)
            out[st + k] = [os.path.basename(p) for p in sv]
        st = end
    _VIEWS[data_dir] = out
    print(f"[faith] views from {data_dir}: {len(out)} samples", flush=True)
    return out


def valid_answer(task, ans):
    if task == "DR":
        return _race_from_answer(ans) is not None
    v = _extract_first_float(ans)
    if v is None:
        return False
    return (0.0 < v < 1.0) if task == "GC" else (0.0 <= v <= 2.0)


def consistent(task, think, ans):
    if task == "GC":
        d, p = parse_pred_gc_dist(think), _extract_first_float(ans)
        g = gini_from_dist(d) if d else None
        return None if (g is None or p is None) else abs(g - p) <= 0.03
    if task == "DR":
        pp, c = parse_pred_dr_props(think), _race_from_answer(ans)
        return None if (not pp or c is None) else max(pp, key=pp.get) == c
    b, w = parse_pred_incomes(think)
    p = _extract_first_float(ans)
    return None if (not b or not w or p is None) else abs(min(b / w, 2.0) - p) <= 0.05


def stats(rows, scorer, base):
    by = defaultdict(list)
    for r in rows:
        by[str(r["task"]).upper()].append(r)
    maxk = max((r.get("n_images") or 0) for r in rows) if rows else 0
    has_n = all(r.get("n_images") for r in rows)
    data_dir = base or os.path.join(HERE, "outputs", "view_select" if maxk <= 13 else "view_all_tr")
    max_sv = 20 if "view_all" in os.path.basename(data_dir.rstrip("/")) else 12
    vmap = views_for(data_dir) if (scorer is not None or not has_n) else {}
    res = {}
    for t in TASKS:
        rs = by.get(t, [])
        if not rs:
            continue
        parse = hall = evid = 0
        cons_ok = cons_n = 0
        grounds, ks, covs = [], [], []
        for r in rs:
            think, ans = _extract_tags(r.get("prediction", "") or "")
            parse += valid_answer(t, ans)
            if r.get("n_images"):
                k = r["n_images"] - 1
            else:
                k = min(len(vmap.get(r["index"], [])), max_sv)
            ks.append(k)
            idx = [int(x) for x in VIEW_IDX_RE.findall(think or "")]
            if idx:
                evid += 1
                hall += any(i < 1 or i > k for i in idx)
            if k:
                covs.append(len({i for i in idx if 1 <= i <= k}) / k)
            c = consistent(t, think, ans)
            if c is not None:
                cons_n += 1
                cons_ok += c
            if scorer is not None:
                vids = vmap.get(r["index"], [])[:k]
                g = scorer.score(think, vids) if idx else 0.0
                grounds.append(0.0 if g is None else g)
        n = len(rs)
        res[t] = {"n": n, "Parse": parse / n, "Halluc": hall / n, "Evid": evid / n,
                  "Cons": cons_ok / cons_n if cons_n else float("nan"), "Cons_n": cons_n / n,
                  "Ground": float(np.mean(grounds)) if grounds else float("nan"),
                  "K": float(np.mean(ks)) if ks else float("nan"),
                  "Cov": float(np.mean(covs)) if covs else float("nan")}
    avg = {m: float(np.nanmean([res[t][m] for t in res])) for m in ("Parse", "Halluc", "Ground", "Cons", "Cov", "Evid", "Cons_n")} if res else {}
    return res, avg, data_dir


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("models", nargs="+", help='"名字=glob" 或 "名字=glob@数据目录"')
    ap.add_argument("--clip-path", default=os.environ.get("CLIP"))
    ap.add_argument("--clip-cache", default=os.path.join(HERE, "outputs", "view_select", "score_cache"))
    ap.add_argument("--no-ground", action="store_true", help="不算 Ground.（不需要 CLIP）")
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args()

    scorer = None
    if not args.no_ground:
        if not args.clip_path:
            sys.exit("需要 --clip-path（或 export CLIP=...），或加 --no-ground")
        from grounding import GroundingScorer
        scorer = GroundingScorer(args.clip_path, args.clip_cache, min_views=3)

    allres = {}
    for spec in args.models:
        name, rest = spec.split("=", 1)
        pat, base = (rest.split("@", 1) + [None])[:2]
        rows = load_rows(pat)
        if not rows:
            print(f"[faith] {name}: no rows for {pat}", flush=True)
            continue
        res, avg, dd = stats(rows, scorer, base)
        allres[name] = {"per_task": res, "avg": avg, "data_dir": dd, "n": len(rows)}
        print(f"[faith] {name}: n={len(rows)}  " + "  ".join(
            f"{t}: K={v['K']:.1f} P={v['Parse']:.3f} H={v['Halluc']:.3f} G={v['Ground']:.3f} C={v['Cons']:.3f}"
            for t, v in res.items()), flush=True)

    print("\n== tab:faith (mean over GC/DR/RIR) ==")
    print(f"{'model':32s} {'Parse':>7s} {'Halluc':>7s} {'Ground':>7s} {'Cons':>7s} {'Cov':>7s} | {'Evid':>6s} {'Cons_n':>6s}")
    for name, v in allres.items():
        a = v["avg"]
        print(f"{name:32s} {a['Parse']:7.3f} {a['Halluc']:7.3f} {a['Ground']:7.3f} {a['Cons']:7.3f} {a['Cov']:7.3f} | "
              f"{a['Evid']:6.3f} {a['Cons_n']:6.3f}")
    if args.json_out:
        json.dump(allres, open(args.json_out, "w"), indent=1)
        print(f"-> {args.json_out}")


if __name__ == "__main__":
    main()
