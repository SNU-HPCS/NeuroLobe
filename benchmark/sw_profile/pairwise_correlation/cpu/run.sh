#!/bin/bash
target_cell_nums="1600" # 1600 or 3200
bin_widths="10"
corr_periods="10"
num_threads="4"

for target_cell_num in $target_cell_nums; do
    for bin_width in $bin_widths; do
        for corr_period in $corr_periods; do
            for num_thread in $num_threads; do
                path=./performance_log_target_cell_num_"$target_cell_num"_bin_width_"$bin_width"_corr_period_"$corr_period"_num_thread_"$num_thread".txt
                python PairwiseCorrelation.py --target_cell_num $target_cell_num --bin_width $bin_width --corr_period $corr_period\
                                                          --num_thread $num_thread \
                                                          > $path
            done
        done
    done
done
