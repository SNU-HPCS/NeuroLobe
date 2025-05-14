Custom SpyKING CIRCUS
==============

Refer to the original github for the user manual.

I added some comments with "Comment:" tag.
You can read through the whole code and refer to the Comment: tag to further understand the code.
If you cannot understand the code, please ask me!

What we should do is to modify the original Spyking CIRCUS to meet the following goals.

#. We should apply *spatial sparsity* in the tempaltes.
    * In the current version, Spyking Circus generates templates using waveforms from *all the electrodes*.
    * I added comments with FIXME (Spatial) tag to indicate the code segments of interest.

#. We should enable real-time spike sorting.
    * In the current version, the template matching procedure adopts greedy algorithm that sorts the spikes with the highest similarity to the template.
    * Instead, the algorithm should sort the spikes in the FCFS manner.
    * I added comments with FIXME (RealTime) tag to indicate the code segments of interest.

We may need to modify the circus/{clustering.py,fitting.py}, circus/shared/files.py, circus/files/{datafile.py,hdf5.py,npy.py}.
