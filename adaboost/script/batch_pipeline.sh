#!/usr/bin/env bash

conda activate adaboost
for season in SH2013 NH2014 SH2014 NH2015 SH2015 NH2016 SH2016 NH2017 SH2017 NH2018 SH2018 NH2019 SH2019 NH2020 SH2020 NH2021 SH2021 NH2022 SH2022 NH2023 SH2023 NH2024
do
  python script/adaboost_run_PLANT_time_split_bayes.py \
        --prefix adaboost_aaindex \
        --directory . \
        --paired-csv /Users/jumpeiito/Dropbox/論文/antigenicity/dataset_new/H3N2_HI/WHO_H3N2_HI_long_format_with_score_filtered_combined.csv \
        --aaindex-csv ./AAindex_GIAG010101_table.csv \
        --split-mode season \
        --cutoff-season ${season} \
        --n-iter 50 \
        --tune-on train_val --final-fit-on train_val \
        --n-jobs 6
done