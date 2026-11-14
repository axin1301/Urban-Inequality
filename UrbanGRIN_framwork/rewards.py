import math
import re
from typing import List, Optional, Tuple


_NUMBER_PATTERN = r"[+-]?(?:\d+(?:\.\d+)?|\.\d+)"
_NUMBER_RE = re.compile(_NUMBER_PATTERN)


def _extract_first_float(text) -> Optional[float]:
    if text is None:
        return None

    m = _NUMBER_RE.search(str(text))
    if m is None:
        return None

    try:
        value = float(m.group(0))
    except (TypeError, ValueError):
        return None

    if not math.isfinite(value):
        return None

    return value


def _to_finite_float(value) -> Optional[float]:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None

    return out if math.isfinite(out) else None


def _completion_text(c) -> str:
    if c is None:
        return ""
    if isinstance(c, str):
        return c
    if isinstance(c, dict):
        content = c.get("content", c.get("text", ""))
        return _completion_text(content)
    if isinstance(c, (list, tuple)):
        return "\n".join(_completion_text(x) for x in c)
    return str(c)


def _extract_tags(text):
    if text is None:
        return None, None

    text = _completion_text(text)

    think = re.search(r"<think>(.*?)</think>", text, re.DOTALL)
    answer = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL)

    return (
        think.group(1).strip() if think else None,
        answer.group(1).strip() if answer else None,
    )


def format_reward(completions: List[str], **_) -> List[float]:
    scores = []

    for c in completions:
        think, answer = _extract_tags(c)
        scores.append(
            1.0 if (think is not None and answer is not None) else 0.0
        )

    return scores


def gc_distribution_reward(completions: List[str], **_) -> List[float]:
    scores = []

    level_re = re.compile(
        rf"LEVEL\d+\s*:\s*({_NUMBER_PATTERN})",
        re.IGNORECASE,
    )

    for c in completions:
        think, _ = _extract_tags(c)

        if think is None:
            scores.append(0.0)
            continue

        matches = level_re.findall(think)

        if not matches:
            scores.append(0.0)
            continue

        values = []

        for raw in matches:
            try:
                value = float(raw)
            except (TypeError, ValueError):
                continue

            if math.isfinite(value):
                values.append(value)

        if not values:
            scores.append(0.0)
            continue

        total = sum(values)
        scores.append(
            1.0 if 0.8 <= total <= 1.2 else 0.0
        )

    return scores


def gc_accuracy_reward(
    completions: List[str],
    labels: List[float],
    **_,
) -> List[float]:
    scores = []

    for c, gt in zip(completions, labels):
        _, answer = _extract_tags(c)

        pred = _extract_first_float(answer)
        gt_value = _to_finite_float(gt)

        if pred is None or gt_value is None:
            scores.append(0.0)
            continue

        scores.append(
            1.0 if abs(pred - gt_value) < 0.05 else 0.0
        )

    return scores


def gc_reward(
    completions: List[str],
    labels: List[float],
    **kwargs,
) -> List[float]:
    fmt = format_reward(completions)
    dist = gc_distribution_reward(completions)
    acc = gc_accuracy_reward(completions, labels)

    return [
        0.2 * f + 0.3 * d + 0.5 * a
        for f, d, a in zip(fmt, dist, acc)
    ]


DR_GROUPS = {"white", "black", "hispanic", "asian"}


def dr_proportion_reward(completions: List[str], **_) -> List[float]:
    scores = []

    for c in completions:
        think, _ = _extract_tags(c)

        if think is None:
            scores.append(0.0)
            continue

        text_lower = think.lower()
        found = sum(
            1
            for g in DR_GROUPS
            if g in text_lower
        )

        scores.append(
            1.0 if found == 4 else found / 4.0
        )

    return scores


def dr_accuracy_reward(
    completions: List[str],
    labels: List[str],
    **_,
) -> List[float]:
    scores = []

    for c, gt in zip(completions, labels):
        _, answer = _extract_tags(c)

        if answer is None:
            scores.append(0.0)
        else:
            scores.append(
                1.0
                if answer.strip().lower() == str(gt).strip().lower()
                else 0.0
            )

    return scores


def dr_reward(
    completions: List[str],
    labels: List[str],
    **kwargs,
) -> List[float]:
    fmt = format_reward(completions)
    prop = dr_proportion_reward(completions)
    acc = dr_accuracy_reward(completions, labels)

    return [
        0.2 * f + 0.3 * p + 0.5 * a
        for f, p, a in zip(fmt, prop, acc)
    ]


def rir_accuracy_reward(
    completions: List[str],
    labels: List[float],
    **_,
) -> List[float]:
    scores = []

    for c, gt in zip(completions, labels):
        _, answer = _extract_tags(c)

        pred = _extract_first_float(answer)
        gt_value = _to_finite_float(gt)

        if pred is None or gt_value is None:
            scores.append(0.0)
            continue

        scores.append(
            1.0 if abs(pred - gt_value) < 0.2 else 0.0
        )

    return scores


def rir_reward(
    completions: List[str],
    labels: List[float],
    **kwargs,
) -> List[float]:
    fmt = format_reward(completions)
    acc = rir_accuracy_reward(completions, labels)

    return [
        0.2 * f + 0.8 * a
        for f, a in zip(fmt, acc)
    ]


REWARD_FN_V1 = {
    "GC": gc_reward,
    "DR": dr_reward,
    "RIR": rir_reward,
}


from aux_labels import GC_REP_INCOME, gini_from_dist

SCALE = {
    "gc_acc": 0.072, "gc_dist": 1.5, "gc_cons": 0.03,
    "dr_prop": 0.9,
    "rir_acc": 0.44, "rir_logb": 0.75, "rir_logw": 0.56, "rir_cons": 0.05,
}

WEIGHTS_V2 = {
    "GC":  {"format": 0.1, "evid": 0.15, "dist": 0.3, "cons": 0.1, "acc": 0.5, "ground": 0.25, "attr": 0.0, "lvl": 0.0},
    "DR":  {"format": 0.1, "evid": 0.15, "prop": 0.3, "cons": 0.1, "acc": 0.5, "ground": 0.25, "attr": 0.0, "lvl": 0.0},
    "RIR": {"format": 0.1, "evid": 0.15, "income": 0.2, "cons": 0.1, "acc": 0.6, "ground": 0.25, "attr": 0.0, "lvl": 0.0},
}


def add_aux_terms(comps: dict, think, aux) -> dict:
    aux = aux or {}
    if aux.get("attrs") or aux.get("levels"):
        import aux_tasks as AT
        comps["attr"] = AT.match_score(AT.parse_line(think, "Attributes"), aux.get("attrs") or {})
        comps["lvl"] = AT.match_score(AT.parse_line(think, "Levels"), aux.get("levels") or {})
    return comps


def _close(d: Optional[float], s: float) -> float:
    if d is None or not math.isfinite(d):
        return 0.0
    return max(0.0, 1.0 - d / s)


def _combine(task: str, comps: dict) -> float:
    w = WEIGHTS_V2[task]
    num = den = 0.0
    for k, wk in w.items():
        v = comps.get(k)
        if v is None:
            continue
        num += wk * float(v)
        den += wk
    return num / den if den > 0 else 0.0


_LEVEL_RE = re.compile(rf"LEVEL\s*(\d+)\s*[:=]\s*({_NUMBER_PATTERN})\s*(%?)", re.IGNORECASE)


def parse_pred_gc_dist(think: Optional[str]) -> Optional[List[float]]:
    if not think:
        return None
    vals = {}
    for lvl, num, pct in _LEVEL_RE.findall(think):
        i = int(lvl)
        if 1 <= i <= 10 and i not in vals:
            v = float(num) / (100.0 if pct else 1.0)
            if math.isfinite(v) and v >= 0:
                vals[i] = v
    if len(vals) < 8:
        return None
    p = [vals.get(i, 0.0) for i in range(1, 11)]
    s = sum(p)
    return [x / s for x in p] if s > 0 else None


_DR_PRED_RE = {
    "White": re.compile(rf"\bWhite\b[^:=\n]{{0,40}}[:=]\s*({_NUMBER_PATTERN})\s*(%?)", re.I),
    "Black": re.compile(rf"\bBlack\b[^:=\n]{{0,40}}[:=]\s*({_NUMBER_PATTERN})\s*(%?)", re.I),
    "Asian": re.compile(rf"\bAsian\b[^:=\n]{{0,40}}[:=]\s*({_NUMBER_PATTERN})\s*(%?)", re.I),
    "Hispanic": re.compile(rf"(?<!not )\bHispanic\b[^:=\n]{{0,40}}[:=]\s*({_NUMBER_PATTERN})\s*(%?)", re.I),
}


def parse_pred_dr_props(think: Optional[str]) -> Optional[dict]:
    if not think:
        return None
    out = {}
    for race, pat in _DR_PRED_RE.items():
        m = pat.search(think)
        if m:
            v = float(m.group(1))
            v = v / 100.0 if (m.group(2) or v > 1.0) else v
            if math.isfinite(v) and v >= 0:
                out[race] = v
    if len(out) != 4 or sum(out.values()) < 0.5:
        return None
    return out


_INCOME_RE = {
    "black": re.compile(r"Black[^0-9\n:=]{0,30}income[^0-9\n:=]{0,10}[:=]\s*\$?\s*([0-9][0-9,]*(?:\.[0-9]+)?)\s*(k|K)?"),
    "white": re.compile(r"White[^0-9\n:=]{0,30}income[^0-9\n:=]{0,10}[:=]\s*\$?\s*([0-9][0-9,]*(?:\.[0-9]+)?)\s*(k|K)?"),
}


def parse_pred_incomes(think: Optional[str]) -> Tuple[Optional[float], Optional[float]]:
    if not think:
        return None, None
    res = []
    for key in ("black", "white"):
        m = _INCOME_RE[key].search(think)
        if not m:
            res.append(None)
            continue
        v = float(m.group(1).replace(",", "")) * (1000.0 if m.group(2) else 1.0)
        res.append(v if v > 0 else None)
    return res[0], res[1]


def _race_from_answer(answer: Optional[str]) -> Optional[str]:
    if not answer:
        return None
    low = answer.lower()
    hits = [r for r in ("white", "black", "hispanic", "asian") if re.search(rf"\b{r}\b", low)]
    return hits[0].capitalize() if len(hits) == 1 else None


_WORD_RE = re.compile(r"[a-z]+")


EVID_MAX_LINES = 5


def evidence_quality(think: Optional[str], min_views: int = 3, max_lines: Optional[int] = None) -> float:
    try:
        from grounding import parse_view_claims
    except ImportError:
        return 1.0
    seen: List[set] = []
    n_ok = 0
    claims = parse_view_claims(think)
    for _, sent in claims:
        low = sent.lower()
        if "satellite" in low or "aerial" in low:
            continue
        words = set(_WORD_RE.findall(low)) - {"a", "an", "the", "of", "with", "and", "in", "is", "view", "street", "shows", "image"}
        if not words:
            continue
        if any(len(words & w) / len(words | w) >= 0.6 for w in seen):
            continue
        seen.append(words)
        n_ok += 1
    q = min(1.0, n_ok / max(1, min_views))
    ml = EVID_MAX_LINES if max_lines is None else max_lines
    if len(claims) > ml:
        q *= float(ml) / len(claims)
    return q


def _format_score(think, answer, structure_ok: bool) -> float:
    tags = 1.0 if (think is not None and answer is not None) else 0.0
    return 0.5 * tags + 0.5 * (1.0 if structure_ok else 0.0)


def gc_components(completion, label, aux: Optional[dict] = None) -> dict:
    think, answer = _extract_tags(completion)
    pred = _extract_first_float(answer)
    gt = _to_finite_float(label)
    acc = _close(abs(pred - gt), SCALE["gc_acc"]) if (pred is not None and gt is not None) else 0.0

    pd = parse_pred_gc_dist(think)
    fmt = _format_score(think, answer, pd is not None)
    gt_dist = (aux or {}).get("dist")
    if gt_dist:
        if pd is None:
            dist = 0.0
        else:
            cum = 0.0
            w1 = 0.0
            for a, b in zip(pd, gt_dist):
                cum += a - b
                w1 += abs(cum)
            dist = _close(w1, SCALE["gc_dist"])
    else:
        dist = None

    if pd is not None and pred is not None:
        g_self = gini_from_dist(pd)
        cons = _close(abs(g_self - pred), SCALE["gc_cons"]) if g_self is not None else 0.0
    else:
        cons = 0.0
    return {"format": fmt, "dist": dist, "cons": cons, "acc": acc, "pred": pred}


def dr_components(completion, label, aux: Optional[dict] = None) -> dict:
    think, answer = _extract_tags(completion)
    pred = _race_from_answer(answer)
    acc = 1.0 if (pred is not None and label is not None
                  and pred.lower() == str(label).strip().lower()) else 0.0

    pp = parse_pred_dr_props(think)
    fmt = _format_score(think, answer, pp is not None)
    gt_props = (aux or {}).get("props")
    if gt_props:
        if pp is None:
            prop = 0.0
        else:
            s = sum(pp.values()) or 1.0
            l1 = sum(abs(pp[r] / s - gt_props.get(r, 0.0)) for r in gt_props)
            prop = _close(l1, SCALE["dr_prop"])
    else:
        prop = None

    cons = 1.0 if (pp is not None and pred is not None and max(pp, key=pp.get) == pred) else 0.0
    return {"format": fmt, "prop": prop, "cons": cons, "acc": acc, "pred": pred}


def rir_components(completion, label, aux: Optional[dict] = None) -> dict:
    think, answer = _extract_tags(completion)
    pred = _extract_first_float(answer)
    gt = _to_finite_float(label)
    acc = _close(abs(pred - gt), SCALE["rir_acc"]) if (pred is not None and gt is not None) else 0.0

    b, w = parse_pred_incomes(think)
    fmt = _format_score(think, answer, b is not None and w is not None)
    aux = aux or {}
    if "black_income" in aux and "white_income" in aux:
        if b is None or w is None:
            income = 0.0
        else:
            income = 0.5 * (_close(abs(math.log(b) - math.log(aux["black_income"])), SCALE["rir_logb"])
                            + _close(abs(math.log(w) - math.log(aux["white_income"])), SCALE["rir_logw"]))
    else:
        income = None

    if b is not None and w is not None and pred is not None:
        cons = _close(abs(min(b / w, 2.0) - pred), SCALE["rir_cons"])
    else:
        cons = 0.0
    return {"format": fmt, "income": income, "cons": cons, "acc": acc, "pred": pred}


COMPONENTS = {"GC": gc_components, "DR": dr_components, "RIR": rir_components}


def grounding_component(completion, view_ids) -> Optional[float]:
    try:
        import grounding
    except ImportError:
        return None
    scorer = grounding.get_scorer()
    if scorer is None:
        return None
    think, _ = _extract_tags(completion)
    return scorer.score(think, view_ids)


def all_components(task, completion, label, aux=None, view_ids=None) -> dict:
    comps = COMPONENTS[task](completion, label, aux)
    think, _ = _extract_tags(completion)
    comps["evid"] = evidence_quality(think)
    comps["ground"] = grounding_component(completion, view_ids)
    return comps


def _v2(task):
    def fn(completions, labels, aux=None, view_ids=None, **_):
        n = len(completions)
        aux = aux if aux is not None else [None] * n
        view_ids = view_ids if view_ids is not None else [None] * n
        return [_combine(task, all_components(task, c, y, a, v))
                for c, y, a, v in zip(completions, labels, aux, view_ids)]
    fn.__name__ = f"{task.lower()}_reward_v2"
    return fn


gc_reward_v2 = _v2("GC")
dr_reward_v2 = _v2("DR")
rir_reward_v2 = _v2("RIR")

REWARD_FN_V2 = {"GC": gc_reward_v2, "DR": dr_reward_v2, "RIR": rir_reward_v2}

DR_CLASS_W = {"White": 0.62, "Black": 0.97, "Hispanic": 2.05, "Asian": 3.95}
DR_TRAIN_PRIOR = {"White": 3290 / 5003, "Black": 1336 / 5003, "Hispanic": 297 / 5003, "Asian": 80 / 5003}


def set_dr_class_power(power: float, cap: float = 6.0):
    raw = {c: p ** (-power) for c, p in DR_TRAIN_PRIOR.items()}
    z = sum(DR_TRAIN_PRIOR[c] * raw[c] for c in raw)
    for c in raw:
        DR_CLASS_W[c] = min(cap, raw[c] / z)
    return dict(DR_CLASS_W)
SCALE_V3 = {"gc_kl": 0.26, "rir_rel": 0.73}

OUTCOME_MODE = "hit"
HALLUC_PEN = 0.0


def _outcome(d, eps, s, cont):
    if OUTCOME_MODE == "band":
        return 1.0 if d <= eps else max(0.0, 1.0 - (d - eps) / s)
    return 0.5 * (1.0 if d <= eps else 0.0) + 0.5 * cont


def _kl_ui_cot(true, pred, eps=1e-10):
    return sum((t + eps) * math.log((t + eps) / (p + eps)) - (t + eps) + (p + eps)
               for t, p in zip(true, pred))


def gc_components_v3(completion, label, aux=None):
    c = gc_components(completion, label, aux)
    think, answer = _extract_tags(completion)
    pred, gt = c["pred"], _to_finite_float(label)
    if pred is not None and gt is not None:
        c["acc"] = _outcome(abs(pred - gt), 0.05, SCALE["gc_acc"], c["acc"])
    gt_dist = (aux or {}).get("dist")
    if gt_dist:
        pd = parse_pred_gc_dist(think)
        c["dist"] = _close(_kl_ui_cot(gt_dist, pd), SCALE_V3["gc_kl"]) if pd else 0.0
    return c


DR_FN_PEN = {"White": 0.0, "Black": 0.0, "Hispanic": 0.0, "Asian": 0.0}
DR_FP_PEN = {"White": 0.0, "Black": 0.0, "Hispanic": 0.0, "Asian": 0.0}


def dr_components_v3(completion, label, aux=None):
    c = dr_components(completion, label, aux)
    y = str(label).strip().capitalize()
    if c["acc"] > 0:
        c["acc"] = DR_CLASS_W.get(y, 1.0)
    elif c["pred"] is not None:
        p = str(c["pred"]).strip().capitalize()
        c["acc"] = -(DR_FN_PEN.get(y, 0.0) + DR_FP_PEN.get(p, 0.0))
    return c


def scale_aux_weights(factor: float):
    for t in WEIGHTS_V2:
        for k in ("format", "evid", "ground"):
            if k in WEIGHTS_V2[t]:
                WEIGHTS_V2[t][k] *= factor
    return WEIGHTS_V2


def rir_components_v3(completion, label, aux=None):
    c = rir_components(completion, label, aux)
    think, _ = _extract_tags(completion)
    pred, gt = c["pred"], _to_finite_float(label)
    if pred is not None and gt is not None:
        c["acc"] = _outcome(abs(pred - gt), 0.2, SCALE["rir_acc"], c["acc"])
    aux = aux or {}
    if aux.get("black_income") and aux.get("white_income"):
        b, w = parse_pred_incomes(think)
        if b and w:
            rel = (abs(w - aux["white_income"]) / aux["white_income"]
                   + abs(b - aux["black_income"]) / aux["black_income"]) / 2
            c["income"] = _close(rel, SCALE_V3["rir_rel"])
        else:
            c["income"] = 0.0
    return c


COMPONENTS_V3 = {"GC": gc_components_v3, "DR": dr_components_v3, "RIR": rir_components_v3}


def _v3(task):
    def fn(completions, labels, aux=None, view_ids=None, **_):
        n = len(completions)
        aux = aux if aux is not None else [None] * n
        view_ids = view_ids if view_ids is not None else [None] * n
        out = []
        for c, y, a, v in zip(completions, labels, aux, view_ids):
            comps = COMPONENTS_V3[task](c, y, a)
            think, _ = _extract_tags(c)
            comps["evid"] = evidence_quality(think)
            comps["ground"] = grounding_component(c, v)
            add_aux_terms(comps, think, a)
            if v:
                cited = [int(x) for x in re.findall(r"View\s*(\d+)\s*:", think or "")]
                if cited:
                    ok = sum(1 for x in cited if 1 <= x <= len(v)) / len(cited)
                    if ok < 1.0:
                        comps["format"] = 0.0
                        comps["evid"] = comps["evid"] * ok
            score = _combine(task, comps)
            if HALLUC_PEN and v:
                cited = [int(x) for x in re.findall(r"View\s*(\d+)\s*:", think or "")]
                if cited:
                    score -= HALLUC_PEN * sum(1 for x in cited if not 1 <= x <= len(v)) / len(cited)
            out.append(score)
        return out
    fn.__name__ = f"{task.lower()}_reward_v3"
    return fn


REWARD_FN_V3 = {"GC": _v3("GC"), "DR": _v3("DR"), "RIR": _v3("RIR")}

REWARD_VERSIONS = {"v1": REWARD_FN_V1, "v2": REWARD_FN_V2, "v3": REWARD_FN_V3}
REWARD_FN = REWARD_FN_V2
