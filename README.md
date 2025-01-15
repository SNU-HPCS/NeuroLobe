## Requirements

#### Dependencies for HW simulator

- python=3.9  
- numpy  
- cython  
- pandas  



#### Dependencies for SW simulator & benchmark generation

- python=3.9  
- numpy  
- cupy  
- h5py  
- nlb-tools  
- dandi  
- elephant  
- networkx  
- MEArec  
- LFPy
- npy_append_array  
- yappi
- custom_slayer  

    ```
    cd benchmark/slayerFCSNN/custom_slayer/cuda_toolkit  
    pip install -e .  
    ```  
  
- custom_spikeinterface  

    ```
    cd benchmark/spike_sorting/custom_spikeinterface  
    pip install -e .  
    ```  
  
- custom_spyking_circus  

    ```
    cd benchmark/spike_sorting/custom_spyking_circus  
    pip install -e .  
    ```  
  
- neo  

    ```
    cd benchmark/spike_sorting/python-neo  
    pip install -e .  
    ```  
  

##
## Creating benchmark data

Following steps create benchmark data and save them in  `benchmark/dataset`.

  
```
cd benchmark
```

  
#### Pairwise Correlation  
1. `cd spike_geneartor`  
2. Set parameters in `gen.params`.  
3. `python spike_generator.py`  
4. `cd ../pairwise_correlation`  
3. Set parameters in  `pc.params`.  
4. `python PairwiseCorrelation.py`  

  
#### Template Matching  
1. `cd spike_geneartor`  
2. Set parameters in `gen.params`.  
3. `python spike_generator.py`  
4. `python template_geneartor.py`  
5. `cd ../template_matching`  
6. Set parameters in `tm.params`.  
7. `python TemplateMatching.py`  

  
#### Neural Network  
1. `cd neural_network`    
3. `python preprocess_maze.py`  
4. `python postproess_maze.py`  
5. Set hyperparameters in `test.sh`.  
6. Set model (ANN / SNN) in `run.sh`.  
6. `./run.sh`  

  
#### Spike Sorting  
1. `cd spike_sorting`  
2. Set parameters in `gen.params`.  
3. `python gen_mearec_dataset.py`.  
4. Set parameters in `ss.params`.  
5. `./SpikeSorting.sh`  

##
## Running the Simulator

### SW simulation
Change directory to the target benchmark and set parameters in `{algorithm}.params`.  
```
cd benchmark/SW_profile/{algorithm}
```

#### GPU  
```
python {algorithm}_gpu.py
```
  
#### CPU
```
cd cpu  
./compile.sh  
python {algorithm}.py
```

  

### HW simulation

Set simulation parameters in `simulator.cfg` and `example_{algorithm}.cfg`.  
Compile with `compile.py` and simulate with `simulate.py`.
```
python compile.py simualtor.cfg example_snn.cfg example_tm.cfg ...  
python simulate.py
```
  
To add new algorithm, run the task compiler before running the simulator.
```
python task_compiler.py
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
