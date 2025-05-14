#!/bin/bash
# This code runs the full process of training + inference.

modes="training inference"

mkdir -p spykingcircus_output

for mode in $modes; do
    # Update ss.params with the current mode
    sed -i "s|mode              = .*|mode              = $mode|" ss.params

    # Run spike sorting with the updated configuration
    python SpikeSorting.py
done

# Parse the data to generate the dataset
python Parse.py
