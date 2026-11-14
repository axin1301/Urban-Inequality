#!/usr/bin/env python3

import argparse
import glob
import json
import math
import os
import sys
from collections import Counter, defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from rewards import (_extract_first_float, _extract_tags, _race_from_answer,
                     parse_pred_dr_props, parse_pred_gc_dist, parse_pred_incomes)
from aux_labels import gini_from_dist, parse_targets

RACES = ["White", "Black", "Asian", "Hispanic"]
EPS = 1e-10


def _load_split(root, split):
    recs = []
    if not root or not os.path.isdir(root):
        return recs
    for fn in sorted(os.listdir(root)):
        low = fn.lower()
        if not low.endswith(".json"):
            continue
        if (split == "test") != ("test" in low):
            continue
        with open(os.path.join(root, fn), encoding="utf-8") as f:
            d = json.load(f)
        recs.extend(x for x in (d if isinstance(d, list) else [d]) if isinstance(x, dict))
    return recs


def load_gt(data_roots, split="test"):
    from dataset import val_ids
    test_aux, stats = {}, {}
    for task, root in data_roots.items():
        test_aux[task] = {}
        train_recs = _load_split(root, "train")
        vids = val_ids(train_recs)
        src = _load_split(root, "test") if split == "test" else \
            [r for i, r in enumerate(train_recs) if str(r.get("sample_id", i)) in vids]
        for i, r in enumerate(src):
            _, aux = parse_targets(task, r)
            test_aux[task][str(r.get("sample_id", i))] = aux
        labels, auxs = [], []
        for i, r in enumerate(train_recs):
            if str(r.get("sample_id", i)) in vids:
                continue
            lab, aux = parse_targets(task, r)
            if lab is not None:
                labels.append(lab)
                auxs.append(aux)
        st = {}
        if labels:
            if task == "DR":
                st["majority"] = Counter(labels).most_common(1)[0][0]
                st["prior"] = {k: v / len(labels) for k, v in Counter(labels).items()}
            else:
                st["mean"] = float(np.mean(labels))
            if task == "GC":
                ds = [a["dist"] for a in auxs if a.get("dist")]
                if ds:
                    m = np.mean(ds, 0)
                    st["dist"] = (m / m.sum()).tolist()
            if task == "RIR":
                b = [a["black_income"] for a in auxs if a.get("black_income")]
                w = [a["white_income"] for a in auxs if a.get("white_income")]
                if b and w:
                    st["black_income"], st["white_income"] = float(np.mean(b)), float(np.mean(w))
        stats[task] = st
    return test_aux, stats


def load_dr_gt_csv(path):
    import csv
    out = {}
    with open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            row = {k.strip().lower(): v for k, v in row.items()}
            tid = str(row.get("tract_id") or row.get("tract") or row.get("sample_id"))
            try:
                v = np.array([float(row[r.lower()]) for r in RACES])
            except (KeyError, ValueError):
                continue
            if v.sum() > 0:
                out[tid] = (v / v.sum()).tolist()
    return out


def _rank(a):
    order = np.argsort(a, kind="mergesort")
    r = np.empty(len(a), float)
    r[order] = np.arange(len(a))
    _, inv, cnt = np.unique(a, return_inverse=True, return_counts=True)
    return (np.bincount(inv, weights=r) / cnt)[inv]


def _corr(a, b):
    return float(np.corrcoef(a, b)[0, 1]) if np.std(a) > 1e-12 and np.std(b) > 1e-12 else float("nan")


def kl_ui_cot(true, pred):
    t, p = np.asarray(true, float) + EPS, np.asarray(pred, float) + EPS
    return float(np.sum(t * np.log(t / p) - t + p))


def _filtered_mean(vals, lo=-2.0, hi=2.0):
    v = [x for x in vals if lo < x < hi]
    return float(np.mean(v)) if v else float("nan")


def regression_block(y, p, tol, const):
    y, p = np.asarray(y, float), np.asarray(p, float)
    err = p - y
    mse = float(np.mean(err ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2)) or 1e-12
    mse_c = float(np.mean((const - y) ** 2)) if const is not None else float("nan")
    out = {
        "RMSE": math.sqrt(mse), "MAE": float(np.abs(err).mean()),
        "Acc": float(np.mean(np.abs(err) <= tol + 1e-12)),
        "R2": 1 - float(np.sum(err ** 2)) / ss_tot,
        "Spearman": _corr(_rank(y), _rank(p)),
        "Skill": 1 - mse / mse_c if const is not None and mse_c > 0 else float("nan"),
        "pred_std/gt_std": float(p.std() / (y.std() or 1e-12)),
    }
    return out


def _set_base(m, rs, tol, const):
    if const is None:
        return
    y = np.array([float(r["label"]) for r in rs if r.get("label") is not None])
    if len(y) == 0:
        return
    ec = const - y
    m.setdefault("base_const", {}).update({"RMSE": float(np.sqrt(np.mean(ec ** 2))),
                                           "MAE": float(np.abs(ec).mean()),
                                           "Acc": float(np.mean(np.abs(ec) <= tol + 1e-12)),
                                           "value": const})


def prf(y, p, classes=RACES):
    y, p = np.asarray(y), np.asarray(p)
    P, R, F, sup, rec, prec = [], [], [], [], {}, {}
    for c in classes:
        tp = np.sum((p == c) & (y == c)); fp = np.sum((p == c) & (y != c)); fn = np.sum((p != c) & (y == c))
        pr = tp / (tp + fp) if tp + fp else 0.0
        rc = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * pr * rc / (pr + rc) if pr + rc else 0.0
        n = int(np.sum(y == c))
        if n == 0:
            continue
        P.append(pr); R.append(rc); F.append(f1); sup.append(n); rec[c] = round(float(rc), 3)
        prec[c] = f"{float(pr):.3f} ({int(tp)}/{int(tp + fp)})"
    w = np.asarray(sup, float) / max(1, sum(sup))
    return {"Precision": float(np.mean(P)), "Recall": float(np.mean(R)), "F1": float(np.mean(F)),
            "weighted_F1": float(np.dot(w, F)), "accuracy": float(np.mean(y == p)),
            "recall_per_class": rec, "precision_per_class": prec,
            "support": dict(zip([c for c in classes if (y == c).any()], sup))}


def evaluate(rows, test_aux, stats, dr_gt=None):
    by = defaultdict(list)
    for r in rows:
        by[str(r["task"]).upper()].append(r)
    out = {}

    rs = by.get("GC", [])
    if rs:
        st = stats.get("GC", {})
        ys, ps, kls, kls_base, gaps = [], [], [], [], []
        yd, pd_ = [], []
        for r in rs:
            think, ans = _extract_tags(r.get("prediction", ""))
            p = _extract_first_float(ans)
            if p is None or not (0.0 < p < 1.0) or r.get("label") is None:
                continue
            ys.append(float(r["label"])); ps.append(p)
            gt = test_aux.get("GC", {}).get(str(r.get("tract_id")), {}).get("dist")
            d = parse_pred_gc_dist(think)
            if r.get("samples"):
                ds = [x for x in (parse_pred_gc_dist(_extract_tags(o)[0]) for o in r["samples"]) if x]
                if ds:
                    m_ = np.mean(ds, 0)
                    d = (m_ / m_.sum()).tolist()
            if gt and d:
                kls.append(kl_ui_cot(gt, d))
            if gt and st.get("dist"):
                kls_base.append(kl_ui_cot(gt, st["dist"]))
            if d:
                g = gini_from_dist(d)
                if g is not None:
                    gaps.append(abs(g - p))
                    yd.append(float(r["label"])); pd_.append(g)
        m = {"n": len(rs), "coverage": len(ps) / len(rs)}
        if len(ps) >= 3:
            m.update(regression_block(ys, ps, 0.05, st.get("mean")))
        m["KL"] = _filtered_mean(kls)
        m["KL_n"] = len(kls)
        _set_base(m, rs, 0.05, st.get("mean"))
        kls_base = [kl_ui_cot(a["dist"], st["dist"]) for a in
                    (test_aux.get("GC", {}).get(str(r.get("tract_id")), {}) for r in rs)
                    if a.get("dist") and st.get("dist")]
        if kls_base:
            m.setdefault("base_const", {})["KL"] = _filtered_mean(kls_base)
        if gaps:
            m["dist_answer_gap_median"] = float(np.median(gaps))
        if len(pd_) >= 3:
            dm = regression_block(yd, pd_, 0.05, st.get("mean"))
            m["from_dist"] = {k: dm[k] for k in ("RMSE", "MAE", "Acc", "Spearman", "Skill")}
            m["from_dist"]["n"] = len(pd_)
        out["GC"] = m

    rs = by.get("RIR", [])
    if rs:
        st = stats.get("RIR", {})
        ys, ps, rels, rels_base = [], [], [], []
        yi, pi_ = [], []
        for r in rs:
            think, ans = _extract_tags(r.get("prediction", ""))
            p = _extract_first_float(ans)
            if p is None or not (0.0 <= p <= 2.0) or r.get("label") is None:
                continue
            ys.append(float(r["label"])); ps.append(p)
            gt = test_aux.get("RIR", {}).get(str(r.get("tract_id")), {})
            ab, aw = gt.get("black_income"), gt.get("white_income")
            qb, qw = parse_pred_incomes(think)
            if qb and qw and 0 <= qb / qw <= 2:
                yi.append(float(r["label"])); pi_.append(qb / qw)
            if ab and aw:
                pb, pw = qb, qw
                if pb and pw:
                    rels.append((abs(pw - aw) / aw + abs(pb - ab) / ab) / 2)
                if st.get("black_income"):
                    rels_base.append((abs(st["white_income"] - aw) / aw
                                      + abs(st["black_income"] - ab) / ab) / 2)
        m = {"n": len(rs), "coverage": len(ps) / len(rs)}
        if len(ps) >= 3:
            m.update(regression_block(ys, ps, 0.2, st.get("mean")))
        m["Rel"] = float(np.mean(rels)) if rels else float("nan")
        if len(pi_) >= 3:
            dm = regression_block(yi, pi_, 0.2, st.get("mean"))
            m["from_income"] = {k: dm[k] for k in ("RMSE", "MAE", "Acc", "Spearman", "Skill")}
            m["from_income"]["n"] = len(pi_)
        m["Rel_n"] = len(rels)
        _set_base(m, rs, 0.2, st.get("mean"))
        rels_base = []
        for r in rs:
            a = test_aux.get("RIR", {}).get(str(r.get("tract_id")), {})
            ab, aw = a.get("black_income"), a.get("white_income")
            if ab and aw and st.get("black_income"):
                rels_base.append((abs(st["white_income"] - aw) / aw + abs(st["black_income"] - ab) / ab) / 2)
        if rels_base:
            m.setdefault("base_const", {})["Rel"] = float(np.mean(rels_base))
        out["RIR"] = m

    rs = by.get("DR", [])
    if rs:
        st = stats.get("DR", {})
        y, p, divs = [], [], []
        for r in rs:
            think, ans = _extract_tags(r.get("prediction", ""))
            c = _race_from_answer(ans)
            if c is None or r.get("label") is None:
                continue
            y.append(str(r["label"])); p.append(c)
            tid = str(r.get("tract_id"))
            gt = (dr_gt or {}).get(tid)
            if gt is None:
                a = test_aux.get("DR", {}).get(tid, {}).get("props")
                if a:
                    v = np.array([a.get(k, 0.0) for k in RACES], float)
                    gt = (v / v.sum()).tolist() if v.sum() > 0 else None
            if gt is not None:
                pp = parse_pred_dr_props(think)
                if pp:
                    divs.append(float(np.sum(np.abs(np.array(gt) - np.array([pp[k] for k in RACES])))))
        m = {"n": len(rs), "coverage": len(p) / len(rs)}
        if p:
            m.update(prf(y, p))
            m["pred_distribution"] = {k: round(v / len(p), 3) for k, v in Counter(p).most_common()}
            y_all = [str(r["label"]) for r in rs if r.get("label") is not None]
            p_all = []
            for r in rs:
                if r.get("label") is None:
                    continue
                c = _race_from_answer(_extract_tags(r.get("prediction", ""))[1])
                p_all.append(c or "INVALID")
            sa = prf(y_all, p_all)
            m["F1_strict"], m["acc_strict"] = sa["F1"], sa["accuracy"]
            if st.get("majority"):
                y = y_all
                b = prf(y, [st["majority"]] * len(y))
                m["base_const"] = {"Precision": b["Precision"], "Recall": b["Recall"], "F1": b["F1"],
                                   "accuracy": b["accuracy"], "value": st["majority"]}
        m["Div"] = float(np.mean(divs)) if divs else float("nan")
        prior = stats.get("DR", {}).get("prior")
        if prior:
            for tau in (0.0, 1.0):
                yy, pp_ = [], []
                for r in rs:
                    if r.get("label") is None:
                        continue
                    props = parse_pred_dr_props(_extract_tags(r.get("prediction", ""))[0])
                    yy.append(str(r["label"]))
                    if props:
                        pp_.append(max(RACES, key=lambda c: props.get(c, 0.0) / max(prior.get(c, 1e-3), 1e-3) ** tau))
                    else:
                        pp_.append(_race_from_answer(_extract_tags(r.get("prediction", ""))[1]) or "INVALID")
                b = prf(yy, pp_)
                m[f"from_props_tau{int(tau)}"] = {"P": b["Precision"], "R": b["Recall"], "F1": b["F1"],
                                                  "acc": b["accuracy"],
                                                  "pred": dict(Counter(pp_).most_common())}
        m["Div_n"] = len(divs)
        out["DR"] = m
    return out


MAIN = {"GC": ["RMSE", "MAE", "Acc", "KL"], "DR": ["Precision", "Recall", "F1", "Div"],
        "RIR": ["RMSE", "MAE", "Acc", "Rel"]}
SHORT = {"Precision": "P", "Recall": "R"}
EXTRA = {"GC": ["R2", "Spearman", "Skill", "pred_std/gt_std", "coverage"],
         "DR": ["accuracy", "weighted_F1", "F1_strict", "acc_strict", "coverage"],
         "RIR": ["R2", "Spearman", "Skill", "pred_std/gt_std", "coverage"]}


def _f(v):
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "-"
    return f"{v:.4f}" if isinstance(v, float) else str(v)


def print_table(results):
    tasks = [t for t in ("GC", "DR", "RIR") if any(t in r for r in results.values())]
    names = {k: os.path.basename(k.rstrip("/")) or k for k in results}
    w = max([len(n) for n in names.values()] + [14])
    cw = 7

    def line(label, getter):
        cells = [" ".join(f"{_f(getter(t, c)):>{cw}}" for c in MAIN[t]) for t in tasks]
        return f"{label:<{w}} | " + " | ".join(cells)

    print("\n== work3 (UI-CoT) 指标，与论文表格同口径 ==")
    print(" " * w + " | " + " | ".join(f"{t:^{(cw + 1) * len(MAIN[t]) - 1}}" for t in tasks))
    hdr = " " * w + " | " + " | ".join(" ".join(f"{SHORT.get(c, c):>{cw}}" for c in MAIN[t]) for t in tasks)
    print(hdr)
    print("-" * len(hdr))
    for k, res in results.items():
        print(line(names[k], lambda t, c, res=res: res.get(t, {}).get(c)))
    first = next(iter(results.values()))
    print(line("const/majority", lambda t, c: first.get(t, {}).get("base_const", {}).get(c)))

    print("\n== 补充指标 ==")
    for k, res in results.items():
        parts = []
        for t in tasks:
            m = res.get(t, {})
            parts.append(f"[{t}] " + " ".join(f"{c}={_f(m.get(c))}" for c in EXTRA[t]))
        print(f"{names[k]:<{w}}  " + "  ".join(parts))
        for t, key in (("GC", "from_dist"), ("RIR", "from_income"),
                       ("DR", "from_props_tau0"), ("DR", "from_props_tau1")):
            if res.get(t, {}).get(key):
                print(f"{'':<{w}}  [{t} {key}] " + " ".join(f"{c}={_f(v)}" for c, v in res[t][key].items()))
        if res.get("DR", {}).get("pred_distribution"):
            print(f"{'':<{w}}  DR pred={res['DR']['pred_distribution']}  "
                  f"recall={res['DR'].get('recall_per_class')}")
            print(f"{'':<{w}}  DR precision(对/预测数)={res['DR'].get('precision_per_class')}  "
                  f"真实各类数={res['DR'].get('support')}")
    print("\n注：同 work3，只在有效预测上计算；const 行 = 训练集均值/平均分布/多数类（不看图像，"
          "在全部样本上计算）；DR F1_strict 把无效回答算错。")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--compare", action="store_true", help="按父目录分组，每个目录 = 一个模型")
    ap.add_argument("--data-root", default=None,
                    help="含 GC/ DR/ RIR/ 子目录的数据根（默认用 config.DATA_ROOTS）")
    ap.add_argument("--split", choices=["test", "val"], default="test")
    ap.add_argument("--dr-gt", default=None, help="DR 种族占比 GT CSV: tract_id,White,Black,Asian,Hispanic")
    ap.add_argument("--json-out", default=None)
    ap.add_argument("--common", action="store_true",
                    help="只在所有模型都评估过的样本（task+tract_id 交集）上比较，用于子集评估")
    args = ap.parse_args()

    if args.data_root:
        roots = {t: os.path.join(args.data_root, t) for t in ("GC", "DR", "RIR")}
    else:
        from config import DATA_ROOTS
        roots = dict(DATA_ROOTS)
    test_aux, stats = load_gt(roots, args.split)
    print("[metrics] GT test aux: " + ", ".join(
        f"{t}={sum(bool(a) for a in d.values())}/{len(d)}" for t, d in test_aux.items()))
    dr_gt = load_dr_gt_csv(args.dr_gt) if args.dr_gt else None

    groups = defaultdict(list)
    for pat in args.files:
        for f in sorted(glob.glob(pat)):
            groups[os.path.dirname(os.path.abspath(f)) if args.compare else "all"].append(f)
    missing = [pat for pat in args.files if not glob.glob(pat)]
    if missing:
        print("[metrics] 没有匹配到文件（检查路径 / 当前目录）:", *missing, sep="\n  ")
    if not groups:
        sys.exit(1)
    allrows = {}
    for key, fs in groups.items():
        rows = []
        for f in fs:
            with open(f, encoding="utf-8") as fh:
                rows.extend(json.loads(l) for l in fh if l.strip())
        allrows[key] = rows
    if args.common:
        keysets = [{(str(r["task"]).upper(), str(r.get("tract_id"))) for r in rs} for rs in allrows.values()]
        common = set.intersection(*keysets)
        print(f"[metrics] --common: {len(common)} samples shared by all models")
        allrows = {k: [r for r in rs if (str(r["task"]).upper(), str(r.get("tract_id"))) in common]
                   for k, rs in allrows.items()}
    results = {}
    for key, rows in allrows.items():
        results[key] = evaluate(rows, test_aux, stats, dr_gt)
    print_table(results)
    if args.json_out:
        json.dump(results, open(args.json_out, "w"), indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
