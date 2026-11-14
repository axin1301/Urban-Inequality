import os
import re
from typing import Dict, Optional

VERSION = os.environ.get("CITYRIUI_AUX_V", "1").strip()

LEVELS3 = ("low", "medium", "high")

ATTR_PROMPTS = {
    "vacancy":            "a street view photo of a vacant lot or boarded-up building",
    "luxury homes":       "a street view photo of large luxury houses with landscaped yards",
    "new housing":        "a street view photo of newly built modern houses",
    "suburban homes":     "a street view photo of a quiet suburban street with single-family homes",
    "run-down buildings": "a street view photo of run-down houses with peeling paint and broken windows",
    "industrial":         "a street view photo of industrial buildings and warehouses",
    "commercial":         "a street view photo of a commercial street with shops and signs",
    "Spanish signs":      "a street view photo of shops with Spanish-language signs",
    "Asian signs":        "a street view photo of shops with Chinese, Korean or Vietnamese signs",
    "apartments":         "a street view photo of apartment buildings",
    "mobile homes":       "a street view photo of mobile homes or trailers",
    "parking lots":       "a street view photo of a large parking lot",
}

_AFFLUENCE = ["vacancy", "luxury homes", "new housing", "suburban homes", "run-down buildings",
              "industrial", "commercial"]
TASK_ATTRS = {
    "GC": _AFFLUENCE,
    "RIR": _AFFLUENCE,
    "DR": ["Spanish signs", "Asian signs", "apartments", "mobile homes", "parking lots", "vacancy"],
}
if VERSION == "3":
    TASK_LEVELS = {"GC": ["low-income share", "high-income share"], "RIR": [], "DR": []}
    FIXED_THR = {}
elif VERSION == "2":
    TASK_LEVELS = {
        "GC": ["low-income share", "high-income share"],
        "RIR": [],
        "DR": ["White share", "Black share", "Hispanic share", "Asian share"],
    }
    FIXED_THR = {"DR": {n: [0.10, 0.40] for n in TASK_LEVELS["DR"]}}
else:
    TASK_LEVELS = {
        "GC": ["low-income share", "high-income share"],
        "RIR": ["Black income", "White income"],
        "DR": ["Black share", "Hispanic share", "Asian share"],
    }
    FIXED_THR = {}
LEVEL_DESC = {
    "low-income share": "share of households earning under $25k (LEVEL1-3)",
    "high-income share": "share of households earning $100k or more (LEVEL8-10)",
    "Black income": "median household income of Black residents",
    "White income": "median household income of White residents",
    "White share": "White population share",
    "Black share": "Black population share",
    "Hispanic share": "Hispanic population share",
    "Asian share": "Asian population share",
}


def level_values(task: str, aux: dict) -> Dict[str, Optional[float]]:
    aux = aux or {}
    if task == "GC":
        d = aux.get("dist")
        if not d:
            return {}
        return {"low-income share": float(sum(d[:3])), "high-income share": float(sum(d[7:]))}
    if task == "RIR":
        if not aux.get("black_income") or not aux.get("white_income"):
            return {}
        return {"Black income": float(aux["black_income"]), "White income": float(aux["white_income"])}
    p = aux.get("props")
    if not p:
        return {}
    s = sum(p.values()) or 1.0
    return {"White share": p.get("White", 0.0) / s, "Black share": p.get("Black", 0.0) / s, "Hispanic share": p.get("Hispanic", 0.0) / s,
            "Asian share": p.get("Asian", 0.0) / s}


def to_level(x: float, thr) -> str:
    return LEVELS3[0] if x < thr[0] else (LEVELS3[1] if x < thr[1] else LEVELS3[2])


def format_line(prefix: str, names, values: Dict[str, str]) -> str:
    return f"{prefix}: " + ", ".join(f"{n}={values[n]}" for n in names if n in values)


def attr_line(task: str, attrs: Dict[str, str]) -> str:
    return format_line("Attributes", TASK_ATTRS[task], attrs)


def levels_line(task: str, levels: Dict[str, str]) -> str:
    return format_line("Levels", TASK_LEVELS[task], levels)


_PAIR_RE = re.compile(r"([A-Za-z][A-Za-z \-]*?)\s*=\s*(low|medium|high)\b", re.I)


def parse_line(think: Optional[str], prefix: str) -> Dict[str, str]:
    if not think:
        return {}
    m = re.search(rf"^\s*{prefix}\s*:\s*(.+)$", think, re.I | re.M)
    if not m:
        return {}
    return {k.strip().lower(): v.lower() for k, v in _PAIR_RE.findall(m.group(1))}


def match_score(pred: Dict[str, str], gold: Dict[str, str]) -> Optional[float]:
    if not gold:
        return None
    tot = 0.0
    for name, g in gold.items():
        p = pred.get(name.lower())
        if p is None:
            continue
        d = abs(LEVELS3.index(p) - LEVELS3.index(g))
        tot += 1.0 if d == 0 else (0.5 if d == 1 else 0.0)
    return tot / len(gold)


def exact_acc(pred: Dict[str, str], gold: Dict[str, str]) -> Optional[float]:
    if not gold:
        return None
    return sum(pred.get(n.lower()) == g for n, g in gold.items()) / len(gold)


def prompt_block(task: str) -> str:
    attrs = ", ".join(TASK_ATTRS[task])
    ex_a = attr_line(task, {n: "low" for n in TASK_ATTRS[task]})
    out = (f"After the evidence, summarize what the views show in ONE line rating each of these features as "
           f"low, medium or high across the tract: {attrs}. Form: '{ex_a}'.\n")
    if not TASK_LEVELS[task]:
        return out
    ex_l = levels_line(task, {n: "medium" for n in TASK_LEVELS[task]})
    if task in FIXED_THR:
        lo, hi = next(iter(FIXED_THR[task].values()))
        levs = ", ".join(LEVEL_DESC[n] for n in TASK_LEVELS[task])
        return out + (f"Then rate, in ONE line, each of: {levs}, as low (under {lo:.0%}), medium "
                      f"({lo:.0%} to {hi:.0%}) or high (over {hi:.0%}). Form: '{ex_l}'.\n")
    levs = "; ".join(f"{n} ({LEVEL_DESC[n]})" for n in TASK_LEVELS[task])
    return out + (f"Then rate, in ONE line, each of: {levs}, as low, medium or high relative to typical "
                  f"US urban tracts. Form: '{ex_l}'.\n")


def format_hint(task: str) -> str:
    return "Attributes: ...\n" + ("Levels: ...\n" if TASK_LEVELS[task] else "")
