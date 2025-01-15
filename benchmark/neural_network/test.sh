#!/bin/bash

set -e
# set -x

lr_delay="4e-0"
bins="120"

epochs_weight=100
epochs_delay=100
h_dim="800"
split="1"
original_num="182"
target_cell_nums="1600"
fr_scale="1"

inference=1

models=FC
batch_size=1
lr_weight=1e-3
gaussian=0

# Slayer Configuration
neuron="cuba"
lr_step="200"
delay=1

test_dir="debug_"$h_dim
num_data=400

if [[ $inference == 0 ]]
then
    dev=0
    for model in $models; do
        for target_cell_num in $target_cell_nums; do
            for bin in $bins; do
                for gauss in $gaussian; do
                    dev=$(($dev % 5))
                    duplicate=$((($target_cell_num - 1) / $original_num + 1))
                    mkdir -p ./logs/$test_dir
                    path=./logs/"$test_dir"/"$model"_bin_"$bin"_gaussian_"$gauss"_batch_"$batch_size"_lr_"$lr_weight"_step_"$lr_step"_target_cell_num_"$target_cell_num".txt
                    python -u main.py --dev $dev \
                                      --model $model \
                                      --neuron $neuron \
                                      --batch_size $batch_size \
                                      --bins $bin \
                                      --gaussian $gauss \
                                      --split $split \
                                      --ep_wgt $epochs_weight \
                                      --ep_delay $epochs_delay \
                                      --delay $delay \
                                      --h_dim $h_dim \
                                      --original_num $original_num \
                                      --lr_weight $lr_weight \
                                      --lr_delay $lr_delay \
                                      --lr_step $lr_step \
                                      --inference $inference \
                                      --duplicate $duplicate \
                                      --target_cell_num $target_cell_num \
                                      --fr_scale $fr_scale \
                                      --num_data $num_data \
                                      | tee $path
                    dev=$(($dev + 1))
                done
            done	
        done
    done
elif [[ $inference == 1 ]]
then
    # Fix the batch size to one
    batch_size=1
    dev=0
    for model in $models; do
        for target_cell_num in $target_cell_nums; do
            for bin in $bins; do
                for gauss in $gaussian; do
                    duplicate=$((($target_cell_num - 1) / $original_num + 1))
                    python -u main.py --dev $dev \
                                      --model $model \
                                      --neuron $neuron \
                                      --batch_size $batch_size \
                                      --bins $bin \
                                      --gaussian $gauss \
                                      --split $split \
                                      --ep_wgt $epochs_weight \
                                      --ep_delay $epochs_delay \
                                      --delay $delay \
                                      --h_dim $h_dim \
                                      --original_num $original_num \
                                      --lr_weight $lr_weight \
                                      --lr_delay $lr_delay \
                                      --lr_step $lr_step \
                                      --inference $inference \
                                      --duplicate $duplicate \
                                      --target_cell_num $target_cell_num \
                                      --fr_scale $fr_scale \
                                      --num_data $num_data
                done
            done
        done
    done	
fi
