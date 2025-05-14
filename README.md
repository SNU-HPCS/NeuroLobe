## Requirements

Run `./install.sh` to install all the following requirements.  
  
  
#### Dependencies for HW simulator

- python=3.9  
- numpy=1.24.4  
- cython=3.0.9  
- pandas=1.3.4  



#### Dependencies for SW simulator & benchmark generation

- h5py=3.10.0  
- statsmodels=0.14.1
- nlb-tools=0.0.3  
- dandi=0.60.0  
- elephant=1.0.0  
- networkx=3.2.1  
- MEArec=1.9.0  
- npy-append-array=0.9.16  
- yappi=1.6.0  
- lfpy=2.3  
- pyqt5=5.15.10  
- cupy-cuda113=10.6.0  
- torch=1.12.1+cu113  
- torchvision=0.13.1+cu113  
- torchaudio=0.12.1  

- custom_slayer  

    ```
    cd benchmark/generate_workload/neural_network/custom_slayer/cuda_toolkit  
    pip install -e .  
    ```  
  
- custom_spikeinterface  

    ```
    cd benchmark/generate_workload/spike_sorting/custom_spikeinterface  
    pip install -e .  
    ```  
  
- custom_spyking_circus  

    ```
    cd benchmark/generate_workload/spike_sorting/custom_spyking_circus  
    pip install -e .  
    ```  
  
- neo  

    ```
    cd benchmark/generate_workload/spike_sorting/python-neo  
    pip install -e .  
    ```  
  

##
## Creating benchmark data

Following steps create benchmark data and save them in  `benchmark/dataset`.

  
```
cd benchmark/generate_workload
```

  
#### Pairwise Correlation  
1. `cd spike_generator`  
2. Set parameters in `gen.params`.  
3. `python spike_generator.py`  
4. `cd ../pairwise_correlation`  
5. Set parameters in  `pc.params`.  
6. `python PairwiseCorrelation.py`  

  
#### Template Matching  
1. `cd spike_generator`  
2. Set parameters in `gen.params`.  
3. `python spike_generator.py`  
4. `python template_generator.py`  
5. `cd ../template_matching`  
6. Set parameters in `tm.params`.  
7. `python TemplateMatching.py`  

  
#### Neural Network  
1. `cd neural_network`    
2. Set parameters in `gen.params`.  
3. `python preprocess_maze.py`  
4. `python postproess_maze.py`  
5. Set hyperparameters in `test.sh`.  
6. Set model (ANN / SNN) in `run.sh`.  
7. `./run.sh`  

  
#### Spike Sorting  
1. `cd spike_sorting`  
2. Set parameters in `gen.params`.  
3. `python gen_mearec_dataset.py`.  
4. Set parameters in `ss.params`.  
5. `./SpikeSorting.sh`  

##
## Running the SW Implementation

Change directory to the target benchmark and set parameters in `{algorithm}.params`.  
```
cd benchmark/sw_profile/{algorithm}
```

### GPU  
```
cd gpu
python {algorithm}_gpu.py
```
  
### CPU
```
cd cpu  
./compile.sh  
python {algorithm}.py
```

  
##
## Running the Simulator

The users should follow the procedures below to simulate the target BCI algorithms.

1. Define the synchronization pattern in the `simulator/sync_pattern.py` and run `python simulator/sync_pattern.py`.  

2. Program the instructions for the event-driven computations and pre/post-processing for the *.inst files in `simulator/instruction_pipelined`.  

3. Set simulation parameters in `simulator.cfg` and `example_{algorithm}.cfg`.  

4. Compile with `compile.py` and simulate with `simulate.py`.
```
python compile.py simulator.cfg example_snn.cfg example_tm.cfg ...  
python simulate.py
```
  
  
### Functionality Check

The correct results are saved in `benchmark/dataset`.  
Compare them with the simulations results.

### Publications

You can refer to the following publication for detailed descriptions of NeuroLobe architecture.

["Rearchitecting a Neuromorphic Processor for Spike-Driven Brain-Computer Interfacing," in *(MICRO24).*](https://ieeexplore.ieee.org/document/10764483)

------------------------------------------------------

If you have any questions or comments, please contact us via email.

Hunjun Lee <hunjunlee@hanyang.ac.kr>

Yeongwoo Jang <yeongwoo.jang@snu.ac.kr>

Daye Jung <daye.jung@snu.ac.kr>

Seunghyun Song <seunghyun.song@snu.ac.kr>
