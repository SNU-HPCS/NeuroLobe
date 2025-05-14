###################################################
###Python File for Pairwise Correlation Analysis###
###################################################
from circus.shared.utils import *
import circus.shared.files as io
from circus.shared.files import get_dead_times
from circus.shared.probes import get_nodes_and_edges
from circus.shared.messages import print_and_log, init_logging, print_debug
from circus.shared.mpi import detect_memory
import time
import numpy as np
import scipy

def main(params, nb_cpu, nb_gpu, use_gpu):

    bin_width = 5 * 45
    window = 4

    #CODE FOR Result
    file_out_suff = params.get('data','file_out_suff')
    hdf5_compress = params.getboolean('data', 'hdf5_compress')

    spikefile = h5py.File(file_out_suff + '.result.hdf5', mode='r', libver = 'earliest')

    spiketimes = spikefile['spiketimes']
    amplitudes = spikefile['amplitudes']
    duration = spikefile['info']['duration'][0] #
    assert(len(spiketimes) == len(amplitudes))
    
    N = len(spiketimes) #N: number of template(neurons)
    print(N)

    # #Construct the data into a matrix of size (neurons, template_width)
    # spike_mat = numpy.zeros((N,int(duration*1000)), dtype=numpy.uint32)
    # for n in range(N):
    #     spikes = spiketimes['temp_'+str(n)][:] // 32
    #     for t in spikes:
    #         spike_mat[n,t] += 1

    #Construct the data into a matrix of size (neurons, template_width)
    spike_mat = numpy.zeros((N, int(duration*1000*32)), dtype=numpy.uint32)
    for n in range(N):
        for t in spiketimes['temp_'+str(n)][:]:
            spike_mat[n,t] = 1


    print(spike_mat.shape)


    corr = [[[0 for _ in range(window * 2 + 1)] for _ in range(N)] for _ in range(N)] # Calculate through HW-like method
    debug_corr = [[[0 for _ in range(window * 2 + 1)] for _ in range(N)] for _ in range(N)] # Direct calculation for debugging

    spike_total = spike_mat.shape[1]
    print(spike_total)

    partial_corr = [[[0 for _ in range(window + 1)] for _ in range(N)] for _ in range(N)] # Calculate through HW-like method
    history = bin_width*window + bin_width//2

    # Make random connection between neurons
    np.random.seed(0)
    conn_list = np.random.randint(0, 2, (N, N))

    print("\nHW-like Method")
    # Calculate pairwise correlation with partial correlograms
    for t in range(spike_total):
        same_time = [[False for _ in range(N)] for _ in range(N)]
        for reference in range(N):
            ref = spike_mat[reference]
            spiked = ref[t]
            # spike at reference neuron
            if spiked > 0:
                for target in range(N):
                    if target != reference and conn_list[reference][target]:
                        tar = spike_mat[target]

                        for i in range(max(0,t-history+1),t):
                            # spike history at target neuron
                            if tar[i]:
                                #print_debug()
                                tar_idx = ((t-i) + (bin_width//2)) // bin_width
                                partial_corr[target][reference][tar_idx] += 1
                                print("t =", t, ":", target, reference, partial_corr[target][reference])

                        if tar[t]:
                            if True:
                            # if not same_time[target][reference]:
                                partial_corr[target][reference][0] += 1
                                same_time[reference][target] = True
                                same_time[target][reference] = True
                                print("t =", t, ":", target, reference, partial_corr[target][reference], "simultaneous")


    # partial correlation concatenation
    for reference in range(N):
        for target in range(N):
            #print_debug()
            past = list(reversed(partial_corr[reference][target]))
            future = partial_corr[target][reference]
            #print(len(past),len(future))
            for i in range(len(past)-1):
                corr[target][reference][i] = past[i]
            corr[target][reference][len(past)-1] = past[-1]+future[0]
            for i in range(len(future)-1):
                corr[target][reference][i+len(past)] = future[i+1]
    print_debug()

    print("correlation")
    for n in range(N):
        print(corr[n])


    # print("\nDirect Calculation")
    # Calculate pairwise correlation for debugging

    # for reference in range(N):
    #     for target in range(N):
    #         for timestep in range(spike_mat.shape[-1]):
    #             if spike_mat[reference][timestep] > 0:
    #                 # if reference neuron spiked
    #                 target_spike = []
    #                 for t in range(timestep - window * bin_width - bin_width // 2, timestep + window * bin_width + (bin_width - bin_width // 2)):
    #                     if t < 0 or t >= spike_mat.shape[-1]:
    #                         target_spike.append(0)
    #                     else:
    #                         target_spike.append(spike_mat[target][t])
    #                 binned_target_spike = np.array(target_spike, dtype=numpy.uint32).reshape(-1, bin_width).sum(axis = 1)
    #                 for w in range(window * 2 + 1):
    #                     debug_corr[reference][target][w] += int(binned_target_spike[w])

    # print("correlation")
    # for n in range(N):
    #     print(debug_corr[n])

    # print("\nDifference")
    # diff = numpy.nan_to_num(corr, nan=0, posinf=0, neginf=0) - numpy.nan_to_num(debug_corr, nan=0, posinf=0, neginf=0)
    # print(numpy.argwhere(abs(diff) > 0.000000001))
    # # print(diff)
