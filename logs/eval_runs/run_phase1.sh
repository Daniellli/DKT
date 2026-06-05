#!/bin/bash
# Phase 1: same-protocol baselines — serial, synchronous, GPU 4.
# Each eval runs in the foreground; moves to the next when it exits.

cd /workspace/DKT

PROTOCOL="--clearpose_root /workspace/datasets/clearpose/set2 \
  --depth_suffix=-depth_true.png \
  --depth_scale 1.0 \
  --max_depth 10000.0 \
  --max_frames 100 \
  --stride 40 \
  --transparent_class_ids 24,26,21,47"

LOGDIR=logs/eval_runs
SUMMARY="$LOGDIR/phase1_summary.txt"
echo "Phase 1 start: $(date)" > "$SUMMARY"

for refine in none moge_affine moge_affine_mask moge_affine_mask_median; do
  LOG="$LOGDIR/v3_${refine}.log"
  echo "" | tee -a "$SUMMARY"
  echo "=== START $refine $(date +%T) ===" | tee -a "$SUMMARY"

  CUDA_VISIBLE_DEVICES=4 python -u tools/eval_clearpose.py $PROTOCOL \
    --refine "$refine" > "$LOG" 2>&1
  EXIT=$?

  echo "=== DONE  $refine $(date +%T)  exit=$EXIT ===" | tee -a "$SUMMARY"
  grep -E 'REL=|OVERALL|delta<1\.(05|10|25)|coverage|WARNING|Traceback' "$LOG" \
    | tail -15 | tee -a "$SUMMARY"
done

echo "" | tee -a "$SUMMARY"
echo "===== ALL FOUR DONE $(date) =====" | tee -a "$SUMMARY"
echo "" | tee -a "$SUMMARY"

for refine in none moge_affine moge_affine_mask moge_affine_mask_median; do
  LOG="$LOGDIR/v3_${refine}.log"
  echo "--- $refine ---" | tee -a "$SUMMARY"
  grep 'REL=' "$LOG" 2>/dev/null | tee -a "$SUMMARY" || echo "  (no REL lines)" | tee -a "$SUMMARY"
  grep -E 'delta<1\.05' "$LOG" 2>/dev/null | tee -a "$SUMMARY" || true
  echo "" | tee -a "$SUMMARY"
done

echo "Phase 1 complete. Full summary in $SUMMARY"
cat "$SUMMARY"
