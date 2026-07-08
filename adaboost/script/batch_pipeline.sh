#!/usr/bin/env bash

conda activate adaboost

season=SH2013

python script/adaboost_run_PLANT_time_split_bayes.py \
        --prefix adaboost_aaindex \
        --directory . \
        --paired-csv ./dataset_new/H3N2_HI/WHO_H3N2_HI_long_format_with_score_filtered_combined.csv \
        --aaindex-csv ./AAindex_GIAG010101_table.csv \
        --split-mode season \
        --cutoff-season ${season} \
        --n-iter 50 \
        --tune-on train_val --final-fit-on train_val \
        --n-jobs 6
