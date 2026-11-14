#!/usr/bin/env python3
import argparse
import json
import os
import re
import sys
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import aux_tasks as AT

TGT_RE = re.compile(r"^<think>(.*)</think>\s*(<answer>.*</answer>)\s*$", re.S)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sft", required=True)
    ap.add_argument("--aux", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    J = json.load(open(args.aux, encoding="utf-8"))
    if str(J.get("version", "1")) != AT.VERSION:
        raise SystemExit(f"标签版本 {J.get('version', '1')} 与 CITYRIUI_AUX_V={AT.VERSION} 不一致")
    labels = J["labels"]
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    kept, skip = Counter(), Counter()
    example = None
    with open(args.out, "w", encoding="utf-8") as fo:
        for l in open(args.sft, encoding="utf-8"):
            if not l.strip():
                continue
            r = json.loads(l)
            t, sid = r["task"], str(r["sample_id"])
            lab = labels.get(t, {}).get(sid)
            m = TGT_RE.match(r["target"].strip())
            need_lv = bool(AT.TASK_LEVELS[t])
            if not lab or not m or not lab.get("attrs") or (need_lv and not lab.get("levels")):
                skip[t] += 1
                continue
            lines = [x for x in m.group(1).strip().split("\n") if x.strip()]
            mid = [AT.attr_line(t, lab["attrs"])] + ([AT.levels_line(t, lab["levels"])] if need_lv else [])
            body = lines[:-1] + mid + [lines[-1]]
            r["target"] = "<think>" + "\n".join(body) + "</think>\n" + m.group(2)
            fo.write(json.dumps(r, ensure_ascii=False) + "\n")
            kept[t] += 1
            if example is None:
                example = r["target"]
    print(f"[aux-sft] kept {dict(kept)}  skipped {dict(skip)}  -> {args.out}")
    print("example target:\n" + (example or ""))


if __name__ == "__main__":
    main()
