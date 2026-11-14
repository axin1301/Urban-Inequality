#!/bin/bash
set -u
cd "$(dirname "$0")" || exit 1
: "${CITYRIUI_BASE:?}" "${POLICY:?}" "${TEACHER:?}" "${CLIP:?}"
OUT=$PWD/outputs; mkdir -p $OUT logs
VS=$OUT/view_select; VC=$OUT/visual_cot; LAB=$OUT/aux_labels_v3.json
SFT=sft_urbangrin; GRPO=urbangrin_grpo600; PIX=$((512*28*28))
export CITYRIUI_MODEL=$POLICY CITYRIUI_DATA_DIR=$VS CITYRIUI_MAX_SV=12
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=6
AUXENV="CITYRIUI_PROMPT=aux CITYRIUI_AUX_V=3 CITYRIUI_AUX_LABELS=$LAB"
NOAUX="env -u CITYRIUI_PROMPT -u CITYRIUI_AUX_V -u CITYRIUI_AUX_LABELS"
WANT=" ${*:-select labels teacher filter sft grpo eval} "
want() { [[ "$WANT" == *" $1 "* ]]; }
log() { echo "[$(date '+%m-%d %H:%M:%S')] $*"; }

if want select && [ ! -f $VS/viewsel_report.txt ]; then
  log "view selection"
  python view_select.py --model $CLIP --k 12 --lam 0.7 --dup-thr 0.9 --info-quantile 0.3 --min-keep 3 \
    --out $VS > logs/view_select.log 2>&1 || { log "failed, see logs/view_select.log"; exit 1; }
fi

if want labels && [ ! -f $LAB ]; then
  log "intermediate-judgment labels"
  CITYRIUI_AUX_V=3 python make_aux_labels.py --clip-path $CLIP --data-dir $VS --max-sv 12 --out $LAB \
    > logs/aux_labels.log 2>&1 || { log "failed, see logs/aux_labels.log"; exit 1; }
fi

if want teacher && [ ! -f $VC/.gen_done ]; then
  for T in GC:1000 DR:2000 RIR:1000; do
    log "teacher ${T%%:*} (n=${T##*:})"
    for S in 0 1; do
      $NOAUX ROCR_VISIBLE_DEVICES=$((4*S)),$((4*S+1)),$((4*S+2)),$((4*S+3)) CUDA_VISIBLE_DEVICES=$((4*S)),$((4*S+1)),$((4*S+2)),$((4*S+3)) \
        python build_visual_cot.py --teacher $TEACHER --all-views --max-lines 12 --evid-tokens 640 \
          --data-dir $VS --max-sv 12 --no-rationale --batch 2 \
          --task ${T%%:*} --n-per-task ${T##*:} --shard $S/2 --out $VC >> logs/teacher_$S.log 2>&1 &
    done
    wait
  done
  grep -a -q -E "Traceback|OutOfMemory" logs/teacher_*.log && { log "teacher failed, see logs/teacher_*.log"; exit 1; }
  touch $VC/.gen_done
fi

if want filter && [ ! -f $OUT/visual_cot_aux3/sft_visual_cot.jsonl ]; then
  log "filter teacher traces"
  python build_visual_cot.py --report-only --all-views --max-lines 12 --min-coverage 0.8 \
    --clip-path $CLIP --clip-cache $VS/score_cache --min-ground 0.2 --out $VC > logs/filter.log 2>&1
  cat $VC/report.txt
  env $AUXENV python build_aux_sft.py --sft $VC/sft_visual_cot.jsonl --aux $LAB \
    --out $OUT/visual_cot_aux3/sft_visual_cot.jsonl > logs/build_aux_sft.log 2>&1
fi

if want sft && [ ! -f $OUT/$SFT/adapter_model.safetensors ]; then
  log "cold-start SFT"
  env $AUXENV torchrun --standalone --nproc_per_node=8 sft_warmup.py \
    --sft-data $OUT/visual_cot_aux3/sft_visual_cot.jsonl --output_name $SFT \
    --epochs 3 --lr 1e-4 --vision_lora --per_device_train_batch_size 1 --gradient_accumulation_steps 2 \
    --max_pixels $PIX --gradient_checkpointing --resume >> logs/sft.log 2>&1
  [ -f $OUT/$SFT/adapter_model.safetensors ] || { log "SFT failed, see logs/sft.log"; exit 1; }
fi

if want grpo && [ ! -f $OUT/$GRPO/adapter_model.safetensors ]; then
  log "GRPO"
  env $AUXENV torchrun --standalone --nproc_per_node=8 train_grpo.py --init_adapter $OUT/$SFT \
    --reward_version v3 --scale_rewards none --dr_class_power 1 --dr_class_cap 2 \
    --gc_dist_weight 0.6 --gc_cons_weight 0.3 --aux_scale 0.3 \
    --dr_fn_pen Black=0.5 --dr_fp_pen Hispanic=0.5,Asian=0.5 --evid_max_lines 12 \
    --attr_weight 0.2 --level_weight 0.2 --learning_rate 1e-6 \
    --output_name $GRPO --per_device_train_batch_size 2 --gradient_accumulation_steps 2 --num_generations 8 \
    --max_pixels $PIX --gradient_checkpointing --temperature 1.0 \
    --data_fraction 0.4 --max_steps 600 --resume_from_checkpoint auto --report_to none >> logs/grpo.log 2>&1
  [ -f $OUT/$GRPO/adapter_model.safetensors ] || { log "GRPO failed, see logs/grpo.log"; exit 1; }
fi

if want eval; then
  D=$OUT/eval/$GRPO
  log "evaluation -> $D"
  for S in 0 1 2 3 4 5 6 7; do
    env $AUXENV ROCR_VISIBLE_DEVICES=$S CUDA_VISIBLE_DEVICES=$S python eval_model.py --checkpoint $OUT/$GRPO \
      --tasks GC,DR,RIR --shard $S/8 --batch_size 4 --resume --tag s$S --max_pixels $PIX \
      --output_dir $D > logs/eval_$S.log 2>&1 &
  done
  wait
  python metrics.py --compare $D/*_predictions.jsonl 2>&1 | tee $OUT/metrics_$GRPO.txt
  python faith_stats.py --clip-path $CLIP --clip-cache $VS/score_cache "UrbanGRIN=$D/*_predictions.jsonl" \
    2>&1 | tee $OUT/faith_$GRPO.txt
fi
log "done"
