#!/usr/bin/env bash

python script/who_hi_extractor_group_violate_target_filter_seasonfix.py \
  --excel "WHO_data/*.xlsx" \
  --fasta nextclade.cds_translation.HA1.fasta \
  --out WHO_H1N1_HI_long_format.csv \
  --log WHO_H1N1_sheet_log.csv \
  --score-out  WHO_H1N1_HI_long_format_with_score_filtered.csv \
  --score-log WHO_H1N1_score_log.csv \
  --titre-log-diff-violate-cutoff -2.0 \
  --titre-log-diff-violate-rate-threshold 0.3 \
  --low-self-titre-log-margin 1.0
 