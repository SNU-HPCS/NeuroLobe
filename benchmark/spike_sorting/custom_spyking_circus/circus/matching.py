##############################################################
###Python File for template matching (correlation analysis)###
##############################################################

from circus.shared.utils import *
import circus.shared.files as io
from circus.shared.files import get_dead_times
from circus.shared.probes import get_nodes_and_edges
from circus.shared.messages import print_and_log, init_logging, print_debug
from circus.shared.mpi import detect_memory
import time
import numpy
import scipy

def main(params, nb_cpu, nb_gpu, use_gpu):
    
    #CODE FOR Result
    file_out_suff = params.get('data','file_out_suff')
    hdf5_compress = params.getboolean('data', 'hdf5_compress')

    spikefile = h5py.File(file_out_suff + '.result.hdf5', mode='r', libver = 'earliest')
    # print(spikefile.keys()) #for debugging

    spiketimes = spikefile['spiketimes']
    amplitudes = spikefile['amplitudes']
    duration = spikefile['info']['duration'][0] #
    assert(len(spiketimes) == len(amplitudes))
    
    N = len(spiketimes) #N: number of template(neurons)
    print(N)

    #Construct the data into a matrix of size (neurons, template_width)
    spike_mat = numpy.zeros((N,int(duration*1000*32)), dtype=numpy.uint32)
    for n in range(N):
        spikes = spiketimes['temp_'+str(n)][:]
        for t in spikes:
            spike_mat[n,t] += 1

    #print(spike_mat[:,:100])

    #CODE FOR Templates

    tempfile = h5py.File(file_out_suff + '.synthetic_templates.hdf5', mode='r', libver = 'earliest')
     
    matchfile = h5py.File(file_out_suff + '.template_matching_result.hdf5', 'w', libver='earliest')
    matchfile.close()

    print_debug()
    temp_x = tempfile.get('temp_x')[:]
    temp_y = tempfile.get('temp_y')[:]
    temp_shape = tempfile.get('temp_shape')[:]
    _temp_neurons = tempfile.get('temp_neurons')[:]

    # Get shape for each template
    temp_neurons = []
    n_tm = len(temp_shape)
    _, n_t = temp_shape[0]
    for tm in range(n_tm):
        num_neu_in_temp = temp_shape[tm][0]
        temp_neurons.append(_temp_neurons[tm][:num_neu_in_temp])

    print(temp_neurons)

    # Get template data
    template_mat = numpy.zeros((n_tm, N, n_t), dtype=numpy.uint32)
    
    print_debug()
    print(template_mat.shape)
    print(temp_x.shape)
    print(temp_y.shape)

    i = 0
    for elec in range(N):
        if i >= len(temp_x):
            break
        while True:
            template_mat[temp_y[i], elec, temp_x[i] - elec * n_t] = 1
            i = i + 1
            if i >= len(temp_x) or temp_x[i] >= (elec + 1) * n_t:
                break

    # print_debug()
    print(template_mat)


    # Perform template matching using correlation analysis
    # Pearson correlation analysis between spike_mat & template_mat

    # Binning data
    print_debug()
    #spike_mat = spike_mat[:,:175]
    #spike_mat = spike_mat[:, :1000]
    print(spike_mat.shape)
    print(template_mat.shape)
    # print(spike_mat[:,:50])
    # print(template_mat[0])
    B = 5
    binned_spike_mat = spike_mat.reshape(N, -1, B).sum(axis=2)
    binned_template_mat = template_mat.reshape(n_tm, N, -1, B).sum(axis=3)

    print_debug()
    print(binned_spike_mat.shape)
    print(binned_spike_mat)
    print(binned_template_mat.shape)
    print(binned_template_mat)

    M = binned_template_mat.shape[-1]

    corr = [[] for _ in range(n_tm)] # Calculation using Reformulated Pearson's Correlation (from Noema)
    debug_corr = [[] for _ in range(n_tm)] # Direct calculation

    C1 = [0 for _ in range(n_tm)]
    C2 = [0 for _ in range(n_tm)]
    C3 = [0 for _ in range(n_tm)]
    S1 = [0 for _ in range(n_tm)]
    S2 = [0 for _ in range(n_tm)]
    S3 = [0 for _ in range(n_tm)]
    R2 = [[0 for _ in range(M)] for _ in range(n_tm)]
    R3 = [[0 for _ in range(M)] for _ in range(n_tm)]

    # list to save summation results in matchfile
    S1_list = []
    S2_list = []
    S3_list = []
     
    # print_debug()
    # print(N, M)

    # Compute Constants
    for tm in range(n_tm):
        C1[tm] = M * len(temp_neurons[tm])
        for m in range(M):
            for n in range (N):
                for i in range(len(temp_neurons[tm])):
                    if n == temp_neurons[tm][i]:
                        C2[tm] += binned_template_mat[tm, i, m]
                        C3[tm] += binned_template_mat[tm, i, m] ** 2
        C3[tm] = C1[tm] * C3[tm] - C2[tm] ** 2

    print_debug()
    print("C1:", C1)
    print("C2:", C2)
    print("C3:", C3)


    print("\nCalculate with Constants and Summations")
    # Initialize R2, R3, S2, S3
    for t in range(M - 1):
        for tm in range(n_tm):
            P2 = 0
            P3 = 0
            for n in range(N):
                if n in temp_neurons[tm]:
                    w = binned_spike_mat[n][t]
                    P2 += w
                    P3 += w ** 2
            S2[tm] += P2
            R2[tm][t] = P2
            S3[tm] += P3
            R3[tm][t] = P3

    # Compute Summations
    for t in range(M - 1, binned_spike_mat.shape[-1]):
        target_spike = binned_spike_mat[:,t-M+1:t+1]
        for tm in range(n_tm):
            P2 = 0
            P3 = 0
            S1[tm] = 0

            for n in range(N):
                for i in range(len(temp_neurons[tm])):
                    if n == temp_neurons[tm][i]:
                        S1[tm] += numpy.sum(binned_template_mat[tm][i] * target_spike[n])
                        w = binned_spike_mat[n][t]
                        P2 += w
                        P3 += w ** 2
            S2[tm] = S2[tm] + P2 - R2[tm][t % M]
            R2[tm][t % M] = P2
            S3[tm] = S3[tm] + P3 - R3[tm][t % M]
            R3[tm][t % M] = P3  

        #print(t, ":", "P2", P2, "P3", P3, "S1", S1, "S2", S2, "S3", S3)

        #print("## t =", t, "##")
        #print(S1, S2, S3)
        #print("S1:", S1)
        #print("S2:", S2)
        #print("S3:", S3)
        # S1_list.append(S1)
        # S2_list.append(S2)
        # S3_list.append(S3)

        for tm in range(n_tm):
            x = ((C1[tm] * S1[tm] - C2[tm] * S2[tm]) ** 2) \
                / (C3[tm] * (C1[tm] * S3[tm] - S2[tm] ** 2))
            corr[tm].append(x)
    
    corr = numpy.array(corr)
    show_corr = corr[numpy.logical_not(numpy.isnan(corr))]
    show_index = numpy.argwhere(numpy.logical_not(np.isnan(corr)))
    print("correlation")
    print(show_corr)
    print("index")
    print(show_index)

    print([max(c) for c in numpy.nan_to_num(corr, nan=0, posinf=0, neginf=0)])


    print("\nCaclulate Correlation Directly")
    for t in range(M - 1, binned_spike_mat.shape[-1]):
        target_spike = binned_spike_mat[:,t-M+1:t+1]

        for tm in range(n_tm):
            # data for neurons actually taking part in the template
            temp_target_spike = []
            temp_target_template = []
            for n in range(N):
                for i in range(len(temp_neurons[tm])):
                    if n == temp_neurons[tm][i]:
                        temp_target_spike.append(list(target_spike[n]))
                        temp_target_template.append(list(binned_template_mat[tm][i]))
            # print(numpy.array(temp_target_spike).shape)
            # print(numpy.array(temp_target_template).shape)

            temp_target_spike = numpy.array(temp_target_spike).flatten()
            temp_target_template = numpy.array(temp_target_template).flatten()
            # print(temp_target_spike, temp_target_template)

            x, _ = scipy.stats.pearsonr(temp_target_spike, temp_target_template)
            debug_corr[tm].append(x)  


    debug_corr = numpy.array(debug_corr) ** 2

    show_corr = debug_corr[numpy.logical_not(numpy.isnan(debug_corr))]
    show_index = numpy.argwhere(numpy.logical_not(np.isnan(debug_corr)))
    print("correlation")
    print(show_corr)
    print("index")
    print(show_index)

    print([max(c) for c in numpy.nan_to_num(debug_corr, nan=0, posinf=0, neginf=0)])

    print("\nDifference")
    diff = numpy.nan_to_num(corr, nan=0, posinf=0, neginf=0) - numpy.nan_to_num(debug_corr, nan=0, posinf=0, neginf=0)
    print(numpy.argwhere(abs(diff) > 0.000000001))
    # print(diff)


    if comm.rank == 0:
        print_debug()
        matchfile = h5py.File(file_out_suff + '.template_matching_result.hdf5', 'r+', libver='earliest')
        if hdf5_compress:
            # print_debug()
            matchfile.create_dataset('C1', data=C1, compression='gzip')
            matchfile.create_dataset('C2', data=C2, compression='gzip')
            matchfile.create_dataset('C3', data=C3, compression='gzip')
            matchfile.create_dataset('S1', data=S1_list, compression='gzip')
            matchfile.create_dataset('S2', data=S2_list, compression='gzip')
            matchfile.create_dataset('S3', data=S3_list, compression='gzip')
        else:
            # print_debug()
            matchfile.create_dataset('C1', data=C1)
            matchfile.create_dataset('C2', data=C2)
            matchfile.create_dataset('C3', data=C3)
            matchfile.create_dataset('S1', data=S1_list)
            matchfile.create_dataset('S2', data=S2_list)
            matchfile.create_dataset('S3', data=S3_list)

        matchfile.flush()
        matchfile.close()

    tempfile.close()    
    spikefile.close()

