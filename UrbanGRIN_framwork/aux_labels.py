from __future__ import annotations

import math
import re
from typing import Dict, List, Optional, Tuple


GC_REP_INCOME = [5000, 12500, 20000, 30000, 42500, 62500, 87500, 125000, 175000, 350000]

RACES = ["White", "Black", "Asian", "Hispanic"]

_OPTION_PREFIX = [
    ("White", re.compile(r"^\s*White\b", re.I)),
    ("Black", re.compile(r"^\s*Black\b", re.I)),
    ("Asian", re.compile(r"^\s*Asian\b", re.I)),
    ("Hispanic", re.compile(r"^\s*Hispanic\b", re.I)),
]

_DR_LINE = {
    "White": re.compile(r"(?m)^\s*-\s*White alone, not Hispanic or Latino\s*:\s*([0-9.]+)\s*%"),
    "Black": re.compile(r"(?m)^\s*-\s*Black or African American\s*:\s*([0-9.]+)\s*%"),
    "Asian": re.compile(r"(?m)^\s*-\s*Asian\s*:\s*([0-9.]+)\s*%"),
    "Hispanic": re.compile(r"(?m)^\s*-\s*Hispanic or Latino\s*:\s*([0-9.]+)\s*%"),
}

_NUM = r"[0-9]*\.?[0-9]+"


def _msg(rec: dict, role: str) -> str:
    roles = {"human", "user"} if role == "human" else {"gpt", "assistant"}
    out = []
    for m in rec.get("conversations", []) or []:
        if str(m.get("from", "")).strip().lower() in roles:
            out.append(str(m.get("value", "")))
    return "\n".join(out)


def _answer_block(text: str) -> str:
    i = text.rfind("<Answer>")
    return text[i:] if i >= 0 else text


def _normalize(p: List[float]) -> Optional[List[float]]:
    s = sum(p)
    if not (s > 0 and math.isfinite(s)):
        return None
    return [x / s for x in p]


def gini_from_dist(p: List[float], rep_income: List[float] = GC_REP_INCOME) -> Optional[float]:
    if p is None or len(p) != len(rep_income):
        return None
    p = _normalize([max(0.0, float(x)) for x in p])
    if p is None:
        return None
    inc = [ri * pi for ri, pi in zip(rep_income, p)]
    tot = sum(inc)
    if tot <= 0:
        return None
    g, prev, cum = 0.0, 0.0, 0.0
    for pi, ii in zip(p, inc):
        cum += ii / tot
        g += pi * (cum + prev)
        prev = cum
    return 1.0 - g


def parse_gc(rec: dict) -> Tuple[Optional[float], Dict]:
    text = _msg(rec, "gpt")
    label = None
    for pat in (r"Gini\s+coefficient\s*:\s*(" + _NUM + ")", r"G\s*=\s*(" + _NUM + r")\s*$"):
        ms = re.findall(pat, _answer_block(text), re.I | re.M)
        if ms:
            v = float(ms[-1])
            if 0.0 <= v <= 1.0:
                label = v
                break

    dist = None
    m = re.search(r"population proportions \(given\).*?:\s*(LEVEL1\s*:.*?LEVEL10\s*:\s*" + _NUM + ")",
                  text, re.I | re.S)
    if m is None:
        m = re.search(r"(LEVEL1\s*:.*?LEVEL10\s*:\s*" + _NUM + ")", _answer_block(text), re.I | re.S)
    if m:
        vals = re.findall(r"LEVEL(\d+)\s*:\s*(" + _NUM + ")", m.group(1), re.I)
        by_lvl = {int(k): float(v) for k, v in vals}
        if all(i in by_lvl for i in range(1, 11)):
            dist = _normalize([by_lvl[i] for i in range(1, 11)])
    return label, ({"dist": dist} if dist else {})


def _canon_option(text: str) -> Optional[str]:
    for name, pat in _OPTION_PREFIX:
        if pat.search(text):
            return name
    return None


def dr_option_map(rec: dict) -> Dict[str, str]:
    human = _msg(rec, "human")
    out = {}
    for letter, opt in re.findall(r"(?m)^\s*([ABCD])\.\s*(.+?)\s*$", human):
        race = _canon_option(opt)
        if race:
            out[letter] = race
    return out


def parse_dr(rec: dict) -> Tuple[Optional[str], Dict]:
    text = _msg(rec, "gpt")
    ans = _answer_block(text)
    label = None
    m = re.search(r"Choice\s*:\s*\**\s*([ABCD])\b", ans)
    if m:
        label = dr_option_map(rec).get(m.group(1))
    if label is None:
        s = text.strip().strip("'\"")
        if s in {"A", "B", "C", "D"}:
            label = dr_option_map(rec).get(s)

    props = {}
    for race, pat in _DR_LINE.items():
        mm = pat.search(ans)
        if mm:
            props[race] = float(mm.group(1)) / 100.0
    aux = {"props": props} if len(props) == 4 else {}
    return label, aux


def parse_rir(rec: dict) -> Tuple[Optional[float], Dict]:
    ans = _answer_block(_msg(rec, "gpt"))
    label = None
    m = re.search(r"Income\s+Ratio\s*:\s*(" + _NUM + ")", ans, re.I)
    if m:
        v = float(m.group(1))
        if 0.0 <= v <= 2.0:
            label = v
    b = re.search(r"Black\s+Median\s+Income\s*:\s*\$?([0-9,]+(?:\.[0-9]+)?)", ans, re.I)
    w = re.search(r"White\s+Median\s+Income\s*:\s*\$?([0-9,]+(?:\.[0-9]+)?)", ans, re.I)
    aux = {}
    if b and w:
        bi, wi = float(b.group(1).replace(",", "")), float(w.group(1).replace(",", ""))
        if bi > 0 and wi > 0:
            aux = {"black_income": bi, "white_income": wi}
    return label, aux


_PARSERS = {"GC": parse_gc, "DR": parse_dr, "RIR": parse_rir, "BW": parse_rir}


def parse_targets(task: str, rec: dict) -> Tuple[Optional[object], Dict]:
    return _PARSERS[task](rec)
