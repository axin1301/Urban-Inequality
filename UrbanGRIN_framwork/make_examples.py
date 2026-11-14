#!/usr/bin/env python3
import argparse
import glob
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
os.environ.setdefault("CITYRIUI_PROMPT", "aux")
os.environ.setdefault("CITYRIUI_AUX_V", "3")

TASK_FILES = {
    "GC": "UI-COT-data/CoT_and_Original_Data/CoT_Data/gc_CoT_{split}.json",
    "DR": "UI-COT-data/dr/CoT/dr_CoT_{split}_change_prompt.json",
    "RIR": "UI-COT-data/bw/CoT/bw_CoT_{split}_change_prompt.json",
}


def load(path):
    d = json.load(open(path, encoding="utf-8"))
    return d if isinstance(d, list) else [d]


def sid(rec, i):
    return str(rec.get("sample_id", i))


def clean_paths(rec):
    rec = json.loads(json.dumps(rec))
    ims = rec.get("image")
    if isinstance(ims, str):
        rec["image"] = "images/" + os.path.basename(ims.strip().strip("'\""))
    elif isinstance(ims, list):
        rec["image"] = ["images/" + os.path.basename(str(p).strip().strip("'\"")) for p in ims]
    return rec


def to_urbangrin(task, rec, i, judgments):
    from aux_labels import parse_targets
    from dataset import build_prompt_text
    ims = clean_paths(rec)["image"]
    label, aux = parse_targets(task, rec)
    out = {"sample_id": sid(rec, i), "task": task, "image": ims,
           "prompt": build_prompt_text(task, len(ims) - 1), "label": label}
    if task == "GC" and aux.get("dist"):
        out["income_distribution"] = [round(x, 4) for x in aux["dist"]]
    elif task == "DR" and aux.get("props"):
        out["racial_shares"] = {k: round(v, 4) for k, v in aux["props"].items()}
    elif task == "RIR" and aux.get("black_income"):
        out["median_income"] = {"Black": aux["black_income"], "White": aux["white_income"]}
    j = judgments.get(task, {}).get(sid(rec, i))
    if j:
        out["judgment_labels"] = {"attributes": j.get("attrs", {}), **({"levels": j["levels"]} if j.get("levels") else {})}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=os.environ.get("CITYRIUI_BASE"))
    ap.add_argument("--reduced", required=True)
    ap.add_argument("--split", default="test", choices=["train", "test"])
    ap.add_argument("--n", type=int, default=2)
    ap.add_argument("--aux-labels", default=None, help="aux_labels_v3.json; if given, the Attributes and Levels labels are included")
    ap.add_argument("--out", default=os.path.join(HERE, "examples"))
    args = ap.parse_args()
    judgments = json.load(open(args.aux_labels))["labels"] if args.aux_labels else {}

    for task, pat in TASK_FILES.items():
        rel = pat.format(split=args.split)
        orig = load(os.path.join(args.base, rel))
        stem = os.path.splitext(os.path.basename(rel))[0]
        red = {sid(r, i): (i, r) for i, r in enumerate(load(os.path.join(args.reduced, task, stem + ".json")))}
        meta = {}
        for f in glob.glob(os.path.join(args.reduced, task, stem + "_meta.jsonl")):
            for l in open(f, encoding="utf-8"):
                m = json.loads(l)
                meta[str(m["sid"])] = m
        picked = []
        for i, r in enumerate(orig):
            s = sid(r, i)
            ims = r.get("image") or []
            if s in red and s in meta and isinstance(ims, list) and len(ims) >= 21:
                picked.append((s, r))
            if len(picked) >= args.n:
                break
        d = os.path.join(args.out, task)
        os.makedirs(d, exist_ok=True)
        json.dump([clean_paths(r) for _, r in picked], open(os.path.join(d, "original.json"), "w"),
                  indent=2, ensure_ascii=False)
        json.dump([to_urbangrin(task, red[s][1], red[s][0], judgments) for s, _ in picked],
                  open(os.path.join(d, "reduced.json"), "w"), indent=2, ensure_ascii=False)
        sel = []
        for s, r in picked:
            m = meta[s]
            names = [os.path.basename(str(p).strip().strip("'\"")) for p in r["image"][1:]]
            chosen = set(m.get("selected", []))
            sel.append({"sample_id": s, "n_street_views": len(names), "n_selected": len(chosen),
                        "views": [{"image": "images/" + n, "p_info": m.get("p_info", {}).get(n),
                                   "selected": n in chosen} for n in names]})
        json.dump(sel, open(os.path.join(d, "selection.json"), "w"), indent=2, ensure_ascii=False)
        print(f"[examples] {task}: {[s for s, _ in picked]} -> {d}")


if __name__ == "__main__":
    main()
