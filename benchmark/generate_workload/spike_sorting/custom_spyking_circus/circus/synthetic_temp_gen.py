###############################################################################
# This file creates synthetic templates for template matching.
# Template Size: (number of recorded neurons, time width, number of templates)
# - number of recorded neurons - determined from the spike sorting result
# - number of templates - predefined value
# The created synthetic templates are converted to csc matrix 
# and saved in the file "recording.synthetic_templates.hdf5".
###############################################################################

from circus.shared.utils import *
import circus.shared.files as io
import circus.shared.algorithms as algo
from circus.shared import plot
import warnings
import scipy
import scipy.optimize
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=FutureWarning)
    import h5py
from circus.shared.probes import get_nodes_and_edges, get_nodes_and_positions
from circus.shared.files import get_dead_times
from circus.shared.messages import print_and_log, init_logging, print_debug
from circus.shared.utils import get_parallel_hdf5_flag
from circus.shared.mpi import detect_memory
import scipy
import scipy.optimize

def main(params, nb_cpu, nb_gpu, use_gpu):
    
    parallel_hdf5 = get_parallel_hdf5_flag(params)
    _ = init_logging(params.logfile)
    logger = logging.getLogger('circus.clustering')
    #################################################################
    SHARED_MEMORY = get_shared_memory_flag(params)
    sparse_threshold = params.getfloat('fitting', 'sparse_thresh')
    file_out_suff = params.get('data', 'file_out_suff')
    hdf5_compress = params.getboolean('data', 'hdf5_compress')

    # FIXME: should configure the number of templates elsewhere
    n_tm = 2
    n_t = 10

    if SHARED_MEMORY:
        templates, _ = io.load_data_memshared(params, 'templates', sparse_threshold=sparse_threshold)
        _, n_n = templates.shape
    else:
        templates = io.load_data(params, 'templates')
        _, n_n = templates.shape

    # n_n is the number of neurons detected from spike sorting
    n_n = n_n // 2
    # print_debug()
    # print(n_n)

    #template size: (neuron # x time_width x temp #)
    temp_x = []
    temp_y = []
    temp_data = []
    temp_neurons = []

    debug = False
    np.random.seed(0)
    neu_in_temp = np.random.randint(1, n_n, n_tm)
    print("Neurons in template: ")
    print(neu_in_temp)

    for tm in range(n_tm):
        temp_neu = neu_in_temp[tm]
        templates = np.random.randint(0, 2, (temp_neu, n_t))
        
        print_debug()
        neu_idx = np.array([n_n for _ in range(n_n)])
        print_debug()
        neu_idx[:temp_neu] = np.random.choice(n_n, temp_neu, replace=False)
        print_debug()

        if debug:
            if tm == 0:
                templates[3:,:] = 0
            elif tm == 1:
                templates[4:,:] = 0
                templates[0, :] = 0
                
        print_debug()
        print("Templates: ")
        print(templates)

        templates = templates.ravel()
        
        # FIXME: no need for temp_data - always 1

        dx = templates.nonzero()[0].astype(numpy.uint32)
        temp_x.append(dx)
        temp_y.append(tm * numpy.ones(len(dx), dtype=numpy.uint32))
        temp_data.append(templates[dx])
        
        temp_neurons.append(sorted(neu_idx))
    
    temp_neurons = np.asarray(temp_neurons)
    print("Temp neurons: ")
    print(temp_neurons)
    # print_debug()
    # print(temp_x)
    # print_debug()
    # print(temp_y)

    temp_x = numpy.concatenate(temp_x)
    temp_y = numpy.concatenate(temp_y)
    temp_data = numpy.concatenate(temp_data)

    # print_debug()
    # print(temp_x)
    # print_debug()
    # print(temp_y)

    sort_idx = np.argsort(temp_x, kind='stable')
    temp_x = [temp_x[i] for i in sort_idx]
    temp_y = [temp_y[i] for i in sort_idx]

    # print_debug()
    # print(temp_x)
    # print_debug()
    # print(temp_y)

    temp_shape = [[n, n_t] for n in neu_in_temp]
    print("Temp shape: ")
    print(temp_shape)

    hfile = h5py.File(file_out_suff + '.synthetic_templates.hdf5', 'w', libver='earliest')
    hfile.close()

    if comm.rank == 0:
        print_debug()
        hfile = h5py.File(file_out_suff + '.synthetic_templates.hdf5', 'r+', libver='earliest')
        if hdf5_compress:
            # save template here
            print_debug()
            hfile.create_dataset('temp_x', data=temp_x, compression='gzip')
            print_debug()
            hfile.create_dataset('temp_y', data=temp_y, compression='gzip')
            print_debug()
            hfile.create_dataset('temp_data', data=temp_data, compression='gzip')
            print_debug()
            hfile.create_dataset('temp_neurons', data=temp_neurons, compression='gzip')
            print_debug()
        else:
            print_debug()
            hfile.create_dataset('temp_x', data=temp_x)
            hfile.create_dataset('temp_y', data=temp_y)
            hfile.create_dataset('temp_data', data=temp_data)
            hfile.create_dataset('temp_neurons', data=temp_neurons, compression='gzip')

        print_debug()
        hfile.create_dataset('temp_shape', data=numpy.array(temp_shape, dtype=numpy.int32))
        hfile.flush()
        hfile.close()
