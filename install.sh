#!/bin/bash

# Dependencies for HW simulator 
#pip install numpy cython pandas

# Dependencies for SW simulator & benchmark
pip install cython pandas h5py nlb-tools dandi elephant networkx LFPy MEArec npy_append_array yappi
pip install cupy

# conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 -f https://download.pytorch.org/whl/torch_stable.html

cd benchmark/slayerFCSNN/custom_slayer/cuda_toolkit
python install -e .
cd ../../../../

cd benchmark/spike_sorting/custom_spikeinterface
pip install -e .
cd ../custom_spyking_circus
pip install -e .
cd ../python-neo
pip install -e .
