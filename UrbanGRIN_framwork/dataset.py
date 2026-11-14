import ast
import json
import os
import re
from pathlib import Path
from typing import Dict, List, Optional

from PIL import Image
from torch.utils.data import ConcatDataset, Dataset

from config import DATA_ROOTS, IMAGE_ROOT, TRAIN_CFG
from aux_labels import parse_targets


TASK_DESC = {
    "GC": (
        "Task: estimate the Gini coefficient of household income (0 to 1, higher = more unequal) "
        "for this US census tract.\n"
        "Inside <think>:\n"
        "1. Visual evidence: one short sentence for each of 3-5 street views, citing it by number and "
        "describing what is specific to that image, e.g. 'View 3: two-story brick houses with detached garages.'\n"
        "2. Estimate the share of households in each income bracket: LEVEL1 <$10k, LEVEL2 $10-15k, "
        "LEVEL3 $15-25k, LEVEL4 $25-35k, LEVEL5 $35-50k, LEVEL6 $50-75k, LEVEL7 $75-100k, "
        "LEVEL8 $100-150k, LEVEL9 $150-200k, LEVEL10 >=$200k. Shares sum to 1.\n"
        "3. The LAST line inside <think> MUST have exactly this form (all ten levels, numbers only):\n"
        "LEVEL1: 0.xx, LEVEL2: 0.xx, LEVEL3: 0.xx, LEVEL4: 0.xx, LEVEL5: 0.xx, LEVEL6: 0.xx, "
        "LEVEL7: 0.xx, LEVEL8: 0.xx, LEVEL9: 0.xx, LEVEL10: 0.xx\n"
        "Then give the Gini implied by this distribution "
        "(representative incomes: 5000, 12500, 20000, 30000, 42500, 62500, 87500, 125000, 175000, 350000).\n"
        "Output format:\n"
        "<think>View i: ...\nView j: ...\nView k: ...\nLEVEL1: 0.xx, ..., LEVEL10: 0.xx</think>\n"
        "<answer>0.xx</answer>"
    ),
    "DR": (
        "Task: identify the dominant racial group (White / Black / Hispanic / Asian) "
        "of this US census tract.\n"
        "Inside <think>:\n"
        "1. Visual evidence: one short sentence for each of 3-5 street views, citing it by number and "
        "describing what is specific to that image, e.g. 'View 3: Spanish-language signs on a strip mall.'\n"
        "2. The LAST line inside <think> MUST have exactly this form (population shares, numbers only, "
        "summing to about 1):\n"
        "White: 0.xx, Black: 0.xx, Hispanic: 0.xx, Asian: 0.xx\n"
        "The answer is the group with the largest share.\n"
        "Output format:\n"
        "<think>View i: ...\nView j: ...\nView k: ...\nWhite: 0.xx, Black: 0.xx, Hispanic: 0.xx, Asian: 0.xx</think>\n"
        "<answer>White|Black|Hispanic|Asian</answer>"
    ),
    "RIR": (
        "Task: estimate the Black-to-White median household income ratio (0 to 2; "
        "< 1 means Black households earn less) for this US census tract.\n"
        "Inside <think>:\n"
        "1. Visual evidence: one short sentence for each of 3-5 street views, citing it by number and "
        "describing what is specific to that image, e.g. 'View 3: older frame houses with peeling paint.'\n"
        "2. The LAST line inside <think> MUST have exactly this form (USD, numbers only):\n"
        "Black median income: xxxxx, White median income: xxxxx\n"
        "The answer is Black median income divided by White median income.\n"
        "Output format:\n"
        "<think>View i: ...\nView j: ...\nView k: ...\nBlack median income: xxxxx, White median income: xxxxx</think>\n"
        "<answer>0.xx</answer>"
    ),
}


_AUX_LAST = {
    "GC": ("Estimate the share of households in each income bracket: LEVEL1 <$10k, LEVEL2 $10-15k, "
           "LEVEL3 $15-25k, LEVEL4 $25-35k, LEVEL5 $35-50k, LEVEL6 $50-75k, LEVEL7 $75-100k, "
           "LEVEL8 $100-150k, LEVEL9 $150-200k, LEVEL10 >=$200k (shares sum to 1). The LAST line inside "
           "<think> MUST have exactly this form (numbers only):\n"
           "LEVEL1: 0.xx, LEVEL2: 0.xx, LEVEL3: 0.xx, LEVEL4: 0.xx, LEVEL5: 0.xx, LEVEL6: 0.xx, "
           "LEVEL7: 0.xx, LEVEL8: 0.xx, LEVEL9: 0.xx, LEVEL10: 0.xx\n"
           "Then give the Gini implied by this distribution (representative incomes: 5000, 12500, 20000, "
           "30000, 42500, 62500, 87500, 125000, 175000, 350000).\n",
           "<answer>0.xx</answer>"),
    "DR": ("The LAST line inside <think> MUST have exactly this form (population shares, numbers only, "
           "summing to about 1):\nWhite: 0.xx, Black: 0.xx, Hispanic: 0.xx, Asian: 0.xx\n"
           "The answer is the group with the largest share.\n",
           "<answer>White|Black|Hispanic|Asian</answer>"),
    "RIR": ("The LAST line inside <think> MUST have exactly this form (USD, numbers only):\n"
            "Black median income: xxxxx, White median income: xxxxx\n"
            "The answer is Black median income divided by White median income.\n",
            "<answer>0.xx</answer>"),
}
_AUX_HEAD = {
    "GC": "Task: estimate the Gini coefficient of household income (0 to 1, higher = more unequal) for this US census tract.\n",
    "DR": "Task: identify the dominant racial group (White / Black / Hispanic / Asian) of this US census tract.\n",
    "RIR": ("Task: estimate the Black-to-White median household income ratio (0 to 2; < 1 means Black households "
            "earn less) for this US census tract.\n"),
}


def _aux_prompt(task: str, n_sv: int) -> str:
    import aux_tasks as AT
    legend = (
        f"You are given {n_sv + 1} images of the same US census tract: the first image is the "
        f"satellite view (it is NOT numbered), followed by {n_sv} street-view images numbered "
        f"View 1 to View {n_sv} in order. View 1 is the first street-view image, not the satellite image.\n\n"
    )
    last, ans = _AUX_LAST[task]
    return (legend + _AUX_HEAD[task] + "Inside <think>:\n"
            f"1. Visual evidence: exactly one short sentence for EACH of the {n_sv} street views, in order from "
            f"View 1 to View {n_sv}, each in the form 'View i: ...' describing what is specific to that image.\n"
            "2. " + AT.prompt_block(task) + "3. " + last +
            "Output format:\n<think>View 1: ...\n...\n" + AT.format_hint(task) + "<last line></think>\n" + ans)


def build_prompt_text(task: str, n_sv: int) -> str:
    if os.environ.get("CITYRIUI_PROMPT", "").lower() == "aux":
        return _aux_prompt(task, n_sv)
    legend = (
        f"You are given {n_sv + 1} images of the same US census tract: the first image is the "
        f"satellite view (it is NOT numbered), followed by {n_sv} street-view images numbered "
        f"View 1 to View {n_sv} in order. View 1 is the first street-view image, not the satellite image.\n"
        "Rules for the evidence: pick the 3-5 most informative street views; each sentence must describe "
        "something different and specific to that image (housing type and condition, businesses, signs, "
        "vehicles, people); never repeat the same sentence.\n\n"
    )
    return legend + TASK_DESC[task]


_IMAGE_INDEX_CACHE: Dict[str, Dict[str, str]] = {}
_AUX_LABELS = None


def _aux_labels() -> dict:
    global _AUX_LABELS
    if _AUX_LABELS is None:
        p = os.environ.get("CITYRIUI_AUX_LABELS")
        J = json.load(open(p, encoding="utf-8")) if p and os.path.isfile(p) else {}
        if J:
            import aux_tasks as AT
            if str(J.get("version", "1")) != AT.VERSION:
                raise RuntimeError(f"aux 标签版本 {J.get('version', '1')} 与 CITYRIUI_AUX_V={AT.VERSION} 不一致")
        _AUX_LABELS = J.get("labels", {})
        if _AUX_LABELS:
            print(f"[Dataset] aux labels v{J.get('version', '1')} from {p}: " + ", ".join(f"{t}={len(v)}" for t, v in _AUX_LABELS.items()),
                  flush=True)
    return _AUX_LABELS


def _clean_path(path) -> str:
    if path is None:
        return ""
    return str(path).strip().strip("'\\\"").replace("\\\\", "/")


def _build_image_index(image_root: str) -> Dict[str, str]:
    root = str(Path(image_root).resolve())
    if root in _IMAGE_INDEX_CACHE:
        return _IMAGE_INDEX_CACHE[root]

    root_path = Path(root)
    if not root_path.is_dir():
        raise FileNotFoundError(f"IMAGE_ROOT does not exist: {root}")

    valid = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    index = {}
    dup = 0

    for p in root_path.rglob("*"):
        if not p.is_file() or p.suffix.lower() not in valid:
            continue
        if p.name in index and index[p.name] != str(p):
            dup += 1
            continue
        index[p.name] = str(p)

    print(
        f"[ImageIndex] root={root} images={len(index)} duplicate_basenames={dup}",
        flush=True,
    )
    _IMAGE_INDEX_CACHE[root] = index
    return index


def _resolve_image_path(
    old_path: str,
    image_root: str,
    image_index: Dict[str, str],
) -> Optional[str]:
    old_path = _clean_path(old_path)
    if not old_path:
        return None

    if os.path.isfile(old_path):
        return old_path

    name = os.path.basename(old_path)

    direct = os.path.join(image_root, name)
    if os.path.isfile(direct):
        return direct

    return image_index.get(name)


def _load_image(
    old_path: str,
    image_root: str,
    image_index: Dict[str, str],
) -> Optional[Image.Image]:
    p = _resolve_image_path(old_path, image_root, image_index)
    if p is None:
        return None

    try:
        with Image.open(p) as img:
            return img.convert("RGB").copy()
    except Exception as e:
        print(f"[WARN] image load failed: {p!r}: {e}", flush=True)
        return None


def _message_text(rec: dict, role: str) -> str:
    if role == "human":
        roles = {"human", "user"}
    elif role == "gpt":
        roles = {"gpt", "assistant"}
    else:
        raise ValueError(role)

    values = []
    for msg in rec.get("conversations", []) or []:
        who = str(msg.get("from", "")).strip().lower()
        if who in roles:
            values.append(str(msg.get("value", "")))
    return "\n".join(values).strip()


RACE = {
    "white": "White",
    "black": "Black",
    "hispanic": "Hispanic",
    "asian": "Asian",
}


def _parse_gc_label(rec: dict):
    text = _message_text(rec, "gpt")
    for pat in [
        r"Gini\s+coefficient\s*:\s*([0-9]*\.?[0-9]+)",
        r"Gini\s*:\s*([0-9]*\.?[0-9]+)",
    ]:
        m = re.search(pat, text, re.I)
        if m:
            v = float(m.group(1))
            if 0.0 <= v <= 1.0:
                return v
    return None


def _parse_rir_label(rec: dict):
    text = _message_text(rec, "gpt")
    for pat in [
        r"Income\s+Ratio\s*:\s*([0-9]*\.?[0-9]+)",
        r"Black[-\s]*to[-\s]*White\s+(?:Income\s+)?Ratio\s*:\s*([0-9]*\.?[0-9]+)",
        r"Ratio\s*:\s*([0-9]*\.?[0-9]+)",
    ]:
        m = re.search(pat, text, re.I)
        if m:
            v = float(m.group(1))
            if 0.0 <= v <= 2.0:
                return v
    return None


def _canonical_race(text: str) -> Optional[str]:
    low = str(text).lower()
    for key, value in RACE.items():
        if re.search(rf"\b{key}\b", low):
            return value
    return None


def _parse_choice_mapping_from_text(text: str) -> Dict[str, str]:
    out = {}

    patterns = [
        r"(?im)^\s*[\(\[]?([ABCD])[\)\].:\-]\s*(White|Black|Hispanic|Asian)\b",
        r"(?i)\b([ABCD])\s*[:=\-]\s*(White|Black|Hispanic|Asian)\b",
    ]
    for pat in patterns:
        for letter, group in re.findall(pat, text):
            out[letter.upper()] = RACE[group.lower()]

    return out


def _parse_choice_list(value) -> Dict[str, str]:
    if value is None:
        return {}

    if isinstance(value, dict):
        out = {}
        for k, v in value.items():
            key = str(k).strip().upper()
            race = _canonical_race(v)
            if key in {"A", "B", "C", "D"} and race:
                out[key] = race
        return out

    if isinstance(value, (list, tuple)):
        out = {}
        for i, v in enumerate(value[:4]):
            race = _canonical_race(v)
            if race:
                out[chr(ord("A") + i)] = race
        return out

    text = str(value).strip()
    if not text or text.lower() in {"null", "none"}:
        return {}

    for parser in (json.loads, ast.literal_eval):
        try:
            parsed = parser(text)
            if parsed != value:
                out = _parse_choice_list(parsed)
                if out:
                    return out
        except Exception:
            pass

    out = _parse_choice_mapping_from_text(text)
    if out:
        return out

    found = re.findall(
        r"\b(White|Black|Hispanic|Asian)\b",
        text,
        re.I,
    )
    if len(found) >= 4:
        return {
            chr(ord("A") + i): RACE[group.lower()]
            for i, group in enumerate(found[:4])
        }

    return {}


def _parse_dr_label(rec: dict):
    label, _ = parse_targets("DR", rec)
    return label


def _parse_label(task: str, rec: dict):
    if task == "GC":
        return _parse_gc_label(rec)
    if task == "DR":
        return _parse_dr_label(rec)
    if task == "RIR":
        return _parse_rir_label(rec)
    raise ValueError(task)


VAL_PER_TASK = int(os.environ.get("CITYRIUI_VAL_PER_TASK", 300))


def _sid(rec, i=None):
    return str(rec.get("sample_id", i))


def val_ids(train_records) -> set:
    import random as _random
    order = list(range(len(train_records)))
    _random.Random(0).shuffle(order)
    return {_sid(train_records[i], i) for i in order[-VAL_PER_TASK:]} if VAL_PER_TASK > 0 else set()


class UICoTDataset(Dataset):
    def __init__(self, task: str, split: str = "train"):
        if task not in DATA_ROOTS:
            raise ValueError(f"Unknown task: {task}")
        if split not in {"train", "test", "val"}:
            raise ValueError(f"Unknown split: {split}")

        self.task = task
        self.split = split
        self.image_root = IMAGE_ROOT
        self.max_sv = int(TRAIN_CFG["max_sv_images"])
        self.task_desc = TASK_DESC[task]
        self.image_index = _build_image_index(self.image_root)

        if split == "test":
            self.records = self._load_records(DATA_ROOTS[task], "test")
        else:
            recs = self._load_records(DATA_ROOTS[task], "train")
            vids = val_ids(recs)
            keep = (lambda r: _sid(r) in vids) if split == "val" else (lambda r: _sid(r) not in vids)
            self.records = [r for r in recs if keep(r)]

        print(
            f"[Dataset] task={task} split={split} records={len(self.records)}",
            flush=True,
        )
        self._health_check()

    def _load_records(self, root: str, split: str):
        root = str(root)
        if not os.path.isdir(root):
            raise FileNotFoundError(root)

        records = []

        for fname in sorted(os.listdir(root)):
            if not fname.lower().endswith(".json"):
                continue

            low = fname.lower()
            if split == "train" and "test" in low:
                continue
            if split == "test" and "test" not in low:
                continue

            fpath = os.path.join(root, fname)
            with open(fpath, "r", encoding="utf-8") as f:
                data = json.load(f)

            batch = data if isinstance(data, list) else [data]
            records.extend(x for x in batch if isinstance(x, dict))

        return records

    def _paths(self, rec: dict):
        paths = rec.get("image", [])
        if paths is None:
            paths = []
        if isinstance(paths, str):
            paths = [paths]

        paths = [_clean_path(p) for p in paths if str(p).strip()]

        sat = paths[0] if paths else None
        sv = paths[1:] if len(paths) > 1 else []
        return sat, sv

    def _build_prompt(self, n_sv: int) -> list:
        return [{"role": "user", "content": build_prompt_text(self.task, n_sv)}]

    def _health_check(self):
        n = min(20, len(self.records))
        if n == 0:
            return

        labels = 0
        sats = 0
        sv_any = 0

        for rec in self.records[:n]:
            if _parse_label(self.task, rec) not in (None, ""):
                labels += 1

            sat, sv = self._paths(rec)

            if sat and _resolve_image_path(
                sat, self.image_root, self.image_index
            ):
                sats += 1

            if any(
                _resolve_image_path(p, self.image_root, self.image_index)
                for p in sv[:self.max_sv]
            ):
                sv_any += 1

        print(
            f"[DatasetCheck] {self.task}/{self.split}: "
            f"labels={labels}/{n}, satellite={sats}/{n}, "
            f"street_any={sv_any}/{n}",
            flush=True,
        )

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]

        tract_id = str(rec.get("sample_id", idx))
        label = _parse_label(self.task, rec)
        _, aux = parse_targets(self.task, rec)
        extra = _aux_labels().get(self.task, {}).get(tract_id)
        if extra:
            aux = dict(aux or {})
            aux.update(extra)

        if label in (None, ""):
            raise ValueError(
                f"Cannot parse label: task={self.task} split={self.split} "
                f"idx={idx} sample_id={tract_id!r} "
                f"gpt={_message_text(rec, 'gpt')[:300]!r}"
            )

        sat_path, sv_paths = self._paths(rec)

        if not sat_path:
            raise FileNotFoundError(
                f"No satellite path: task={self.task} split={self.split} "
                f"idx={idx} sample_id={tract_id!r}"
            )

        sat_img = _load_image(
            sat_path,
            self.image_root,
            self.image_index,
        )
        if sat_img is None:
            raise FileNotFoundError(
                f"Satellite image not found: sample_id={tract_id!r} "
                f"path={sat_path!r} basename={os.path.basename(sat_path)!r}"
            )

        sv_imgs: List[Image.Image] = []
        view_ids: List[str] = []
        for p in sv_paths:
            if len(sv_imgs) >= self.max_sv:
                break

            img = _load_image(
                p,
                self.image_root,
                self.image_index,
            )
            if img is not None:
                sv_imgs.append(img)
                view_ids.append(os.path.basename(p))

        images = [sat_img] + sv_imgs
        prompt = self._build_prompt(len(sv_imgs))

        return {
            "prompt": prompt,
            "images": images,
            "label": label,
            "aux": aux,
            "view_ids": view_ids,
            "task": self.task,
            "tract_id": tract_id,
        }


class MultiTaskDataset(ConcatDataset):
    def __init__(self, split: str = "train"):
        datasets = [
            UICoTDataset(task=t, split=split)
            for t in ["GC", "DR", "RIR"]
        ]
        super().__init__(datasets)

        print(
            f"[MultiTaskDataset] split={split} total={sum(len(d) for d in datasets)} "
            f"(GC={len(datasets[0])}, DR={len(datasets[1])}, "
            f"RIR={len(datasets[2])})",
            flush=True,
        )
