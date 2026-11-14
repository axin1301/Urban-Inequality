# UrbanGRIN

Code for **UrbanGRIN: Reinforcing Multimodal Large Language Models for Evidence-Grounded Urban Inequality Estimation**.

UrbanGRIN estimates three tract-level inequality indicators from one satellite image and a set of street view images: the Gini coefficient (GC), the dominant race (DR), and the Black/White income ratio (RIR). The pipeline has three stages:

1. **Input reduction** (`view_select.py`): CLIP informativeness scoring and MMR selection keep an informative, non-redundant subset of the street views.
2. **Evidence curation** (`build_visual_cot.py`, `make_aux_labels.py`, `build_aux_sft.py`): a teacher MLLM writes one evidence line per selected view; traces are filtered by quality, view coverage, and CLIP grounding, and intermediate judgments (Attributes, Levels) are added. The curated traces form the cold-start set (`sft_warmup.py`).
3. **Verifiable refinement** (`train_grpo.py`, `rewards.py`, `grounding.py`): group-relative policy optimization with outcome, intermediate-quantity, consistency, judgment, evidence, and format rewards.

`eval_model.py`, `metrics.py`, and `faith_stats.py` evaluate the indicators, the intermediate quantities, and the faithfulness of the reasoning.

## Requirements

- One node with 8 GPUs (we used 4x AMD MI250X, i.e. 8 GCDs with 64 GB each, under ROCm). The 72B teacher runs as two copies on 4 GPUs each.
- Python packages: see `requirements.txt` (key versions) and `ENVIRONMENT.md`. In particular, TRL 0.25.1.
- Local snapshots of Qwen2-VL-7B-Instruct (policy), Qwen2.5-VL-72B-Instruct (teacher), and CLIP ViT-L/14.

## Data

We use the urban inequality benchmark of Zhang et al. (WWW 2025). Place it as

```
$CITYRIUI_BASE/UI-COT-data/
    CoT_and_Original_Data/CoT_Data/gc_CoT_{train,test}.json
    dr/CoT/dr_CoT_{train,test}_change_prompt.json
    bw/CoT/bw_CoT_{train,test}_change_prompt.json
    all_images/UI-COT-images_full/        # satellite and street view images
```

## Run

```bash
export CITYRIUI_BASE=/path/to/CityRiUI
export POLICY=/path/to/Qwen2-VL-7B-Instruct
export TEACHER=/path/to/Qwen2.5-VL-72B-Instruct
export CLIP=/path/to/clip-vit-large-patch14
bash run_urbangrin.sh                       # all stages
bash run_urbangrin.sh eval                  # a single stage: select labels teacher filter sft grpo eval
```

Outputs are written to `./outputs` (override with `CITYRIUI_OUTPUT_DIR`). Finished stages are skipped, so the script can be rerun after an interruption; GRPO resumes from its latest checkpoint.

## Files

| File | Role |
|---|---|
| `view_select.py` | CLIP scoring and MMR view selection |
| `make_aux_labels.py`, `aux_tasks.py`, `aux_labels.py` | Labels and parsing of intermediate judgments and quantities |
| `build_visual_cot.py` | Teacher evidence generation and curation filters |
| `build_aux_sft.py` | Adds Attributes and Levels to the curated traces |
| `sft_warmup.py` | Cold-start supervised fine-tuning (LoRA) |
| `train_grpo.py`, `trl_multiimage_patch.py` | Policy optimization with TRL (multi-image patch) |
| `rewards.py`, `grounding.py` | Reward functions and contrastive CLIP grounding |
| `dataset.py`, `config.py`, `vlm_base.py` | Data loading, paths, and model loading |
| `eval_model.py`, `metrics.py`, `faith_stats.py` | Evaluation, indicator metrics, faithfulness metrics |
| `stage1_verify.py` | Label parsing helpers used by view selection |
