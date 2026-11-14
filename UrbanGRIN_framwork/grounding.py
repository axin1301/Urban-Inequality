from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

_VIEW_RE = re.compile(
    r"(?:^|\n|;|\.\s)\s*[-*]?\s*(?:street\s*)?view\s*#?\s*(\d{1,2})\s*[:\-–)]\s*([^\n;]+)",
    re.IGNORECASE,
)


def parse_view_claims(think: Optional[str], max_claims: int = 20) -> List[tuple]:
    if not think:
        return []
    out, seen = [], set()
    for idx, sent in _VIEW_RE.findall(think):
        i = int(idx)
        sent = sent.strip().strip(".").strip()
        if i in seen or len(sent) < 8:
            continue
        if re.search(r"\b(satellite|aerial)\b", sent, re.I):
            continue
        seen.add(i)
        out.append((i, sent[:300]))
        if len(out) >= max_claims:
            break
    return out


class GroundingScorer:

    def __init__(self, clip_path: str, cache_dir: str, min_views: int = 3,
                 device: Optional[str] = None):
        self.clip_path = clip_path
        self.cache_dir = Path(cache_dir)
        self.min_views = min_views
        self.device = device
        self._model = self._proc = None
        self._name2row: Dict[str, int] = {}
        self._emb: Optional[np.ndarray] = None
        self._scale = 100.0

    def _ensure(self):
        if self._model is not None:
            return
        import torch
        from transformers import AutoModel, AutoProcessor

        names, embs = [], []
        for f in sorted(self.cache_dir.glob("shard_*.npz")):
            z = np.load(f, allow_pickle=False)
            names.extend(z["names"].tolist())
            embs.append(z["emb"])
        if not names:
            raise FileNotFoundError(f"[grounding] 没有找到嵌入缓存: {self.cache_dir}/shard_*.npz")
        emb = np.concatenate(embs).astype(np.float32)
        emb /= np.linalg.norm(emb, axis=1, keepdims=True) + 1e-8
        self._emb = emb
        self._name2row = {n: i for i, n in enumerate(names)}

        if self.device is None:
            if torch.cuda.is_available():
                self.device = f"cuda:{torch.cuda.current_device()}"
            else:
                self.device = "cpu"
        dtype = torch.float16 if self.device.startswith("cuda") else torch.float32
        self._model = AutoModel.from_pretrained(self.clip_path, torch_dtype=dtype).to(self.device).eval()
        self._proc = AutoProcessor.from_pretrained(self.clip_path)
        self._is_siglip = "siglip" in str(getattr(self._model.config, "model_type", "")).lower()
        if hasattr(self._model, "logit_scale"):
            self._scale = float(self._model.logit_scale.exp())
        print(f"[grounding] loaded {len(names)} view embeddings, CLIP on {self.device}", flush=True)

    def encode_text(self, sents: List[str]) -> np.ndarray:
        import torch
        self._ensure()
        t = self._proc(text=sents, return_tensors="pt",
                       padding="max_length" if self._is_siglip else True, truncation=True)
        t = {k: v.to(self.device) for k, v in t.items()}
        with torch.no_grad():
            e = self._model.get_text_features(**t)
            if not isinstance(e, torch.Tensor):
                for attr in ("text_embeds", "pooler_output"):
                    v = getattr(e, attr, None)
                    if isinstance(v, torch.Tensor):
                        e = v
                        break
                else:
                    e = e[0]
            e = e.float()
            e = e / e.norm(dim=-1, keepdim=True)
        return e.cpu().numpy()

    def view_matrix(self, view_ids: List[str]) -> tuple:
        self._ensure()
        rows = [self._name2row.get(os.path.basename(v)) for v in view_ids]
        valid = np.array([r is not None for r in rows])
        E = np.zeros((len(rows), self._emb.shape[1]), np.float32)
        for j, r in enumerate(rows):
            if r is not None:
                E[j] = self._emb[r]
        return E, valid

    def score(self, think: Optional[str], view_ids: Optional[List[str]]) -> Optional[float]:
        if not view_ids:
            return None
        E, valid = self.view_matrix(view_ids)
        k = int(valid.sum())
        if k < 2:
            return None
        claims = parse_view_claims(think)
        if not claims:
            return 0.0
        T = self.encode_text([s for _, s in claims])
        logits = self._scale * (T @ E.T)
        logits[:, ~valid] = -np.inf
        logits -= logits.max(1, keepdims=True)
        prob = np.exp(logits)
        prob /= prob.sum(1, keepdims=True)
        chance = 1.0 / k
        gs = []
        for row, (i, _) in enumerate(claims):
            j = i - 1
            if j < 0 or j >= len(view_ids) or not valid[j]:
                gs.append(0.0)
                continue
            gs.append(float(np.clip((prob[row, j] - chance) / (1 - chance), 0.0, 1.0)))
        coverage = min(1.0, len({i for i, _ in claims}) / max(1, self.min_views))
        return float(np.mean(gs)) * coverage


_SCORER: Optional[GroundingScorer] = None


def enable(clip_path: str, cache_dir: str, min_views: int = 3):
    global _SCORER
    _SCORER = GroundingScorer(clip_path, cache_dir, min_views=min_views)
    return _SCORER


def get_scorer() -> Optional[GroundingScorer]:
    return _SCORER
