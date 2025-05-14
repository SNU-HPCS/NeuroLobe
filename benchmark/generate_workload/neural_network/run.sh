#!/bin/bash

# FC / SlayerFCSNN
model="FC"

# This code runs the full process of training + inference with the given model.
modes="0 1"
for mode in $modes; do
    # Update test.sh
    sed -i "s|inference=.*|inference=$mode|" test.sh
        
    if [ "$model" == "SlayerFCSNN" ]; then
        sed -i "s|models=.*|models=$model|" test.sh
        sed -i "s|lr_weight=.*|lr_weight="1e-2"|" test.sh
        sed -i "s|batch_size=.*|batch_size=64|" test.sh
        sed -i "s|gaussian=.*|gaussian="0"|" test.sh

    elif [ "$model" == "FC" ]; then
        sed -i "s|models=.*|models=$model|" test.sh
        sed -i "s|lr_weight=.*|lr_weight="1e-3"|" test.sh
        sed -i "s|batch_size=.*|batch_size=1|" test.sh
        sed -i "s|gaussian=.*|gaussian="0"|" test.sh
    else
        echo "Model not supported"
    fi

    ./test.sh
done
