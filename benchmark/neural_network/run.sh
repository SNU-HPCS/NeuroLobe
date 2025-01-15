#!/bin/bash

model="FC"

# This code runs the full process of training + inference with the given model.
modes="0 1"
fr_scale="8 16"
for fr in $fr_scale; do
    sed -i "s|fr_scale=.*|fr_scale=$fr|" test.sh
    for mode in $modes; do
        # Update test.sh
        sed -i "s|inference=.*|inference=$mode|" test.sh
            
        if [ "$model" == "SlayerFCSNN" ] || [ "$model" == "TorchFCSNN" ]; then
            sed -i "s|models=.*|models=$model|" test.sh
            sed -i "s|lr_weight=.*|lr_weight="1e-2"|" test.sh
            sed -i "s|batch_size=.*|batch_size=64|" test.sh
            sed -i "s|gaussian=.*|gaussian="0"|" test.sh

        elif [ "$model" == "FC" ] || [ "$model" == "RNNTANH" ] || [ "$model" == "RNNRELU" ] || [ "$model" == "LSTM" ]; then
            sed -i "s|models=.*|models=$model|" test.sh
            sed -i "s|lr_weight=.*|lr_weight="1e-3"|" test.sh
            sed -i "s|batch_size=.*|batch_size=1|" test.sh
            sed -i "s|gaussian=.*|gaussian="0"|" test.sh

        elif [ "$model" == "SVM" ]; then
            sed -i "s|models=.*|models=$model|" test.sh
            sed -i "s|lr_weight=.*|lr_weight="1e-1"|" test.sh
            sed -i "s|batch_size=.*|batch_size=1|" test.sh
            sed -i "s|gaussian=.*|gaussian="0"|" test.sh

        elif [ "$model" == "SVMSNN" ]; then
            sed -i "s|models=.*|models=$model|" test.sh
            sed -i "s|lr_weight=.*|lr_weight="1e-2"|" test.sh
            sed -i "s|batch_size=.*|batch_size=64|" test.sh
            sed -i "s|gaussian=.*|gaussian="0"|" test.sh

        else
            echo "Model not supported"
        fi

        ./test.sh
    done
done