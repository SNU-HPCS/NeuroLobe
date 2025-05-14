#!/bin/bash

# Dependencies for HW simulator
pip install numpy==1.24.4 cython==3.0.9 pandas==1.3.4

# Dependencies for SW simulator & benchmark
pip install h5py==3.10.0 statsmodels==0.14.1 nlb-tools==0.0.3 dandi==0.60.0 elephant==1.0.0 networkx==3.2.1 MEArec==1.9.0 npy_append_array==0.9.16 yappi==1.6.0 LFPy==2.3 pyqt5==5.15.10

# check CUDA version
pip install cupy-cuda113 torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 -f https://download.pytorch.org/whl/torch_stable.html

cd benchmark/generate_workload/neural_network/custom_slayer/cuda_toolkit
pip install -e .
cd ../../../../../

cd benchmark/generate_workload/spike_sorting/custom_spikeinterface
pip install -e .
cd ../custom_spyking_circus
pip install -e .
cd ../python-neo
pip install -e .