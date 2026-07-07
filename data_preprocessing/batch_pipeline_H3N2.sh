#!/usr/bin/env bash

python script/fix_external_hi_date.py \
  --input WIC-HI-H3N2.csv \
  --output WIC-HI-H3N2_datefixed.csv \
  --old-date 2003-02-15 \
  --new-date 2013-02-15
  

python script/who_hi_extractor_group_violate_target_filter_seasonfix.py \
  --excel "WHO_data/*.xlsx" \
  --fasta nextclade.cds_translation.HA1.fasta \
  --out WHO_H3N2_HI_long_format.csv \
  --log WHO_H3N2_sheet_log.csv \
  --score-out  WHO_H3N2_HI_long_format_with_score_filtered.csv \
  --score-log WHO_H3N2_score_log.csv \
  --titre-log-diff-violate-cutoff -2.0 \
  --titre-log-diff-violate-rate-threshold 0.3 \
  --low-self-titre-log-margin 1.0

 
python script/who_hi_combiner_group_violate_target_filter_seasonfix.py \
  --external-hi WIC-HI-H3N2_datefixed.csv \
  --who-score-csv WHO_H3N2_HI_long_format_with_score_filtered.csv \
  --fasta nextclade.cds_translation.HA1.fasta \
  --out WHO_H3N2_HI_long_format_with_score_filtered_combined.csv  \
  --external-out WIC-HI-H3N2_with_score.csv \
  --missing-fasta-out external_H3N2_missing_fasta_matches.csv \
  --log combined_HI_score_log.csv \
  --titre-log-diff-violate-cutoff -2.0 \
  --titre-log-diff-violate-rate-threshold 0.3 \
 --low-self-titre-log-margin 1.0
  


