from circus.shared.utils import *
import circus.shared.files as io
from circus.shared.files import get_dead_times
from circus.shared.probes import get_nodes_and_edges, get_nodes_and_positions
from circus.shared.messages import print_and_log, init_logging
from circus.shared.mpi import detect_memory
import time
import sys

def main(params, nb_cpu, nb_gpu, use_gpu):

    #################################################################
    # params = detect_memory(params)
    _ = init_logging(params.logfile)
    SHARED_MEMORY = get_shared_memory_flag(params)
    logger = logging.getLogger('circus.fitting')
    data_file = params.data_file
    n_e = params.getint('data', 'N_e')
    n_total = params.nb_channels
    n_t = params.getint('detection', 'N_t')
    template_shift = params.getint('detection', 'template_shift')
    # file_out = params.get('data', 'file_out')
    file_out_suff = params.get('data', 'file_out_suff')
    sign_peaks = params.get('detection', 'peaks')
    matched_filter = params.getboolean('detection', 'matched-filter')
    # spike_thresh = params.getfloat('detection', 'spike_thresh')
    ratio_thresh = params.getfloat('fitting', 'ratio_thresh')
    two_components = params.getboolean('fitting', 'two_components')
    sparse_threshold = params.getfloat('fitting', 'sparse_thresh')
    # spike_width = params.getfloat('detection', 'spike_width')
    # dist_peaks = params.getint('detection', 'dist_peaks')
    do_temporal_whitening = params.getboolean('whitening', 'temporal')
    do_spatial_whitening = params.getboolean('whitening', 'spatial')
    templates_normalization = params.getboolean('clustering', 'templates_normalization')  # TODO test, switch, test!
    chunk_size = detect_memory(params, fitting=True)
    gpu_only = params.getboolean('fitting', 'gpu_only')
    nodes, edges = get_nodes_and_edges(params)
    tmp_limits = params.get('fitting', 'amp_limits').replace('(', '').replace(')', '').split(',')
    tmp_limits = [float(v) for v in tmp_limits]
    amp_auto = params.getboolean('fitting', 'amp_auto')
    max_chunk = params.getfloat('fitting', 'max_chunk')
    # noise_thr = params.getfloat('clustering', 'noise_thr')
    collect_all = params.getboolean('fitting', 'collect_all')
    min_second_component = params.getfloat('fitting', 'min_second_component')
    debug = params.getboolean('fitting', 'debug')
    ignore_dead_times = params.getboolean('triggers', 'ignore_times')
    inv_nodes = numpy.zeros(n_total, dtype=numpy.int32)
    inv_nodes[nodes] = numpy.arange(len(nodes))
    data_file.open()
    supports = io.load_data(params, 'supports')
    low_channels_thr = params.getint('detection', 'low_channels_thr')
    median_channels = numpy.median(numpy.sum(supports, 1))
    fixed_amplitudes = params.getboolean('clustering', 'fixed_amplitudes')

    # get electrode positions here
    _, positions = get_nodes_and_positions(params)
    # get parameters 'real_time','spatial_sparse'
    real_time = params.getboolean('fitting','real_time')
    spatial_sparse = params.getboolean('fitting','spatial_sparse')
    valid_multiple = params.getboolean('fitting','valid_multiple')
    duration = params.getint('fitting', 'duration')
    print("----------------------------------------")
    print("real time implementation :", real_time)
    print("spatial sparsity considered :", spatial_sparse)
    print("valid once and match:", valid_multiple)
    print("two components : ", two_components)
    print("duration: ", duration)
    print("----------------------------------------")
    # save electrode positions
    positions = numpy.array(positions)
    numpy.save('./spykingcircus_output/sorter_output/electrode_positions', positions)


    weird_thresh = params.get('detection', 'weird_thresh')
    if weird_thresh != '':
        ignore_artefacts = True
        weird_thresh = io.load_data(params, 'weird-thresholds')
    else:
        ignore_artefacts = False

    if not fixed_amplitudes:
        nb_amp_bins = params.getint('clustering', 'nb_amp_bins')
        splits = np.linspace(0, params.data_file.duration, nb_amp_bins)
        interpolated_times = np.zeros(len(splits) - 1, dtype=numpy.float32)
        for count in range(0, len(splits) - 1):
            interpolated_times[count] = (splits[count] + splits[count + 1])/2
        interpolated_times = numpy.concatenate(([0], interpolated_times, [params.data_file.duration]))
        nb_amp_times = len(splits) + 1

    mse_error = params.getboolean('fitting', 'mse_error')
    if mse_error:
        stds = io.load_data(params, 'stds')
        stds_norm = numpy.linalg.norm(stds)

    #################################################################

    # We should modify the template to enable spatially sparse template matching
    # N_tm: indicates the number of templates
    # x: indicates the #of electrodes * time_width (this value is sparsified)

    # always densify...
    # sparse_threshold = 0

    chunk_size = duration

    if SHARED_MEMORY:
        templates, mpi_memory_1 = io.load_data_memshared(params, 'templates', normalize=templates_normalization, transpose=True, sparse_threshold=sparse_threshold)
        N_tm, x = templates.shape
        is_sparse = not isinstance(templates, numpy.ndarray)
    else:
        templates = io.load_data(params, 'templates')
        x, N_tm = templates.shape
        if N_tm > 0:
            sparsity = templates.nnz / (x * N_tm)
            is_sparse = sparsity < sparse_threshold
        else:
            is_sparse = True
        if not is_sparse:
            if comm.rank == 0:
                print_and_log(['Templates sparsity is low (%g): densified to speedup the algorithm' %sparsity], 'debug', logger)
            templates = templates.toarray()


    temp_2_shift = 2 * template_shift
    temp_3_shift = 3 * template_shift
    # n_tm: the number of actual templates, single template can be decomposed into 2 type : E,V
    n_tm = N_tm // 2
    n_scalar = n_e * n_t

    temp_window = numpy.arange(-template_shift, template_shift + 1)
    size_window = n_e * (2 * template_shift + 1)

    #print("temp_window:", temp_window)

    if not amp_auto:
        amp_limits = numpy.zeros((n_tm, 2))
        amp_limits[:, 0] = tmp_limits[0]
        amp_limits[:, 1] = tmp_limits[1]
    else:
        amp_limits = io.load_data(params, 'limits')

    norm_templates = io.load_data(params, 'norm-templates')

    sub_norm_templates = n_scalar * norm_templates[:n_tm]
    if not templates_normalization:
        norm_templates_2 = (norm_templates ** 2.0) * n_scalar
        sub_norm_templates_2 = norm_templates_2[:n_tm]

    if not SHARED_MEMORY:
        # Normalize templates (if necessary).
        if templates_normalization:
            if is_sparse:
                for idx in range(templates.shape[1]):
                    myslice = numpy.arange(templates.indptr[idx], templates.indptr[idx+1])
                    templates.data[myslice] /= norm_templates[idx]
            else:
                for idx in range(templates.shape[1]):
                    templates[:, idx] /= norm_templates[idx]
        # Transpose templates.
        templates = templates.T

    maxoverlap = io.load_data(params, 'maxoverlap')/n_scalar
    similar = np.where(maxoverlap > 0.5)

    idx = similar[0] < similar[1]
    similar = similar[0][idx], similar[1][idx]
    nb_mixtures = len(similar[0])

    waveform_neg = numpy.empty(0)
    matched_thresholds_neg = None
    waveform_pos = numpy.empty(0)
    matched_thresholds_pos = None
    if matched_filter:
        if sign_peaks in ['negative', 'both']:
            waveform_neg = io.load_data(params, 'waveform')[::-1]
            waveform_neg /= (numpy.abs(numpy.sum(waveform_neg)) * len(waveform_neg))
            matched_thresholds_neg = io.load_data(params, 'matched-thresholds')
        if sign_peaks in ['positive', 'both']:
            waveform_pos = io.load_data(params, 'waveform-pos')[::-1]
            waveform_pos /= (numpy.abs(numpy.sum(waveform_pos)) * len(waveform_pos))
            matched_thresholds_pos = io.load_data(params, 'matched-thresholds-pos')

    if ignore_dead_times:
        if SHARED_MEMORY:
            all_dead_times, mpi_memory_3 = get_dead_times(params)
        else:
            all_dead_times = get_dead_times(params)
    else:
        all_dead_times = None  # default assignment (for PyCharm code inspection)

    thresholds = io.get_accurate_thresholds(params, ratio_thresh)

    # save thresholds
    np.save(file_out_suff + '.thresholds.npy', thresholds)

    neighbors = {}
    if collect_all:
        is_sparse = not isinstance(templates, numpy.ndarray)
        for i in range(0, n_tm):
            if is_sparse:
                tmp = templates[i, :].toarray().reshape(n_e, n_t)
            else:
                tmp = templates[i].reshape(n_e, n_t)
            if templates_normalization:
                tmp = tmp * norm_templates[i]
            neighbors[i] = numpy.where(numpy.sum(tmp, axis=1) != 0.0)[0]

    info_string = ''

    if comm.rank == 0:
        info_string = "using %d CPUs" % comm.size

    comm.Barrier()

    c_overlap = io.get_overlaps(params, nb_cpu=nb_cpu, nb_gpu=nb_gpu, use_gpu=use_gpu)
    over_shape = c_overlap.get('over_shape')[:]
    n_over = int(numpy.sqrt(over_shape[0]))
    s_over = over_shape[1]
    s_center = s_over // 2
    # # If the number of overlaps is different from templates, we need to recompute them.
    if n_over != N_tm:
        if comm.rank == 0:
            print_and_log(['Templates have been modified, recomputing the overlaps...'], 'default', logger)
        c_overlap = io.get_overlaps(params, erase=True, nb_cpu=nb_cpu, nb_gpu=nb_gpu, use_gpu=use_gpu)
        over_shape = c_overlap.get('over_shape')[:]
        n_over = int(numpy.sqrt(over_shape[0]))
        s_over = over_shape[1]

    # Comment:
    # Retrieve the precalculated overlaps
    if SHARED_MEMORY:
        c_overs, mpi_memory_2 = io.load_data_memshared(params, 'overlaps')
    else:
        c_overs = io.load_data(params, 'overlaps')

    comm.Barrier()

    if n_tm == 0:
        if comm.rank == 0:
            print_and_log(["No templates present. Redo clustering?"], 'default', logger)

        sys.exit(0)

    if comm.rank == 0:
        print_and_log(["Here comes the SpyKING CIRCUS %s and %d templates..." % (info_string, n_tm)], 'default', logger)
        purge(file_out_suff, '.data')

    if do_spatial_whitening:
        spatial_whitening = io.load_data(params, 'spatial_whitening')
    else:
        spatial_whitening = None  # default assignment (for PyCharm code inspection)
    if do_temporal_whitening:
        temporal_whitening = io.load_data(params, 'temporal_whitening')
    else:
        temporal_whitening = None  # default assignment (for PyCharm code inspection)

    nb_chunks, last_chunk_len = data_file.analyze(chunk_size)
    processed_chunks = int(min(nb_chunks, max_chunk))

    comm.Barrier()
    spiketimes_file = open(file_out_suff + '.spiketimes-%d.data' % comm.rank, 'wb')
    comm.Barrier()
    amplitudes_file = open(file_out_suff + '.amplitudes-%d.data' % comm.rank, 'wb')
    comm.Barrier()
    templates_file = open(file_out_suff + '.templates-%d.data' % comm.rank, 'wb')
    comm.Barrier()

    if ignore_artefacts:
        comm.Barrier()
        arte_spiketimes_file = open(file_out_suff + '.times-%d.sata' % comm.rank, 'wb')
        comm.Barrier()
        arte_electrodes_file = open(file_out_suff + '.elec-%d.sata' % comm.rank, 'wb')
        comm.Barrier()
        arte_amplitudes_file = open(file_out_suff + '.amp-%d.sata' % comm.rank, 'wb')
        comm.Barrier()

    if mse_error:
        mse_file = open(file_out_suff + '.mses-%d.data' % comm.rank, 'wb')
        comm.Barrier()

    if collect_all:
        garbage_times_file = open(file_out_suff + '.gspiketimes-%d.data' % comm.rank, 'wb')
        comm.Barrier()
        garbage_temp_file = open(file_out_suff + '.gtemplates-%d.data' % comm.rank, 'wb')
        comm.Barrier()
    else:
        garbage_times_file = None  # default assignment (for PyCharm code inspection)
        garbage_temp_file = None  # default assignment (for PyCharm code inspection)

    if debug:
        # Open debug files.
        chunk_nbs_debug_file = open(file_out_suff + '.chunk_nbs_debug_%d.data' % comm.rank, mode='wb')
        comm.Barrier()
        iteration_nbs_debug_file = open(file_out_suff + '.iteration_nbs_debug_%d.data' % comm.rank, mode='wb')
        comm.Barrier()
        peak_nbs_debug_file = open(file_out_suff + '.peak_nbs_debug_%d.data' % comm.rank, mode='wb')
        comm.Barrier()
        peak_local_time_steps_debug_file = open(
            file_out_suff + '.peak_local_time_steps_debug_%d.data' % comm.rank, mode='wb'
        )
        comm.Barrier()
        peak_time_steps_debug_file = open(file_out_suff + '.peak_time_steps_debug_%d.data' % comm.rank, mode='wb')
        comm.Barrier()
        peak_scalar_products_debug_file = open(
            file_out_suff + '.peak_scalar_products_debug_%d.data' % comm.rank, mode='wb'
        )
        comm.Barrier()
        peak_solved_flags_debug_file = open(file_out_suff + '.peak_solved_flags_debug_%d.data' % comm.rank, mode='wb')
        comm.Barrier()
        template_nbs_debug_file = open(file_out_suff + '.template_nbs_debug_%d.data' % comm.rank, mode='wb')
        comm.Barrier()
        success_flags_debug_file = open(file_out_suff + '.success_flags_debug_%d.data' % comm.rank, mode='wb')
        comm.Barrier()
    else:
        chunk_nbs_debug_file = None
        iteration_nbs_debug_file = None
        peak_nbs_debug_file = None
        peak_local_time_steps_debug_file = None
        peak_time_steps_debug_file = None
        peak_scalar_products_debug_file = None
        peak_solved_flags_debug_file = None
        template_nbs_debug_file = None
        success_flags_debug_file = None

    last_chunk_size = 0
    slice_indices = numpy.zeros(0, dtype=numpy.int32)

    # Comment:
    # Existing parallel the computations by splitting the computations across different timesteps
    # FIXME (RealTime)
    # We may need to change the code to parallel the computations across different data
    to_explore = list(range(comm.rank, processed_chunks, comm.size))

    # templates_spatial (N_tm x (dictionary {key:elec value:waveform}))
    templates_spatial = []
    if spatial_sparse:
        for template_idx in range(N_tm):
            dict_template = {}
            # make dict_template
            for elec in range(n_e):
                # iterate through electrodes
                elec_template = templates[template_idx, n_t*elec:n_t*(elec+1)]

                # check if template is empty or not
                all_zeros = not numpy.array(elec_template).any()
                if not all_zeros:
                    dict_template.update({elec:elec_template}) # save only nonzero template
            templates_spatial.append(dict_template)


    # check density
    template_density_list = []
    total_valid_elec_num = 0
    for template_idx in range(n_tm):
        valid_elec_num = 0
        for elec in range(n_e):
            elec_template = templates[template_idx, n_t*elec:n_t*(elec+1)]
            all_zeros = not numpy.array(elec_template).any()
            if not all_zeros:
                valid_elec_num += 1
                total_valid_elec_num += 1
        template_density = valid_elec_num / n_e
        template_density_list.append(template_density)
    avg_density = np.mean(np.array(template_density_list))

    if comm.rank == 0:
        # save summary analysis file
        # 1. total number of templates
        total_num_templates = n_tm
        # 2. total number of electrodes
        total_num_electrodes = n_e
        # 3. average # of electrodes per template
        avg_num_elec_per_template = avg_density * n_e
        # 4. average # of templates per electrode
        avg_num_temp_per_electrode = total_valid_elec_num / n_e
        analysis_file = open('./spykingcircus_output/summary_analysis_file.csv', 'w')

        analysis_file.write(f'total number of templates, {total_num_templates}\n')
        analysis_file.write(f'total number of electrodes, {total_num_electrodes}\n')
        analysis_file.write(f'average number of electrodes per template, {avg_num_elec_per_template}\n')
        analysis_file.write(f'average number of templates per electrode, {avg_num_temp_per_electrode}\n')
        analysis_file.close()

    if comm.rank == 0:
        to_explore = get_tqdm_progressbar(params, to_explore)

    # default true
    if fixed_amplitudes:
        min_scalar_products = amp_limits[:,0][:, numpy.newaxis]
        max_scalar_products = amp_limits[:,1][:, numpy.newaxis]

        if templates_normalization:
            min_sps = min_scalar_products * sub_norm_templates[:, numpy.newaxis]
            max_sps = max_scalar_products * sub_norm_templates[:, numpy.newaxis]
        else:
            min_sps = min_scalar_products * sub_norm_templates_2[:, numpy.newaxis]
            max_sps = max_scalar_products * sub_norm_templates_2[:, numpy.newaxis]

    # Comment : loop through the local chunks
    sum_miss_peak_times = 0
    sum_detected_peak_times = 0

    # collect peak times per electrodes
    elec_peak_times = {}
    for i in range(n_e):
        elec_peak_times[i] = []

    custom_result = {
            'sorting': [[] for _ in range(n_tm)]
            }

    if comm.rank == 0:
        total_chunk,_ = data_file.get_data(0, duration, (0,temp_3_shift), nodes)
        print("total chunk shape", total_chunk.shape)
        if do_spatial_whitening:
            total_chunk = numpy.dot(total_chunk, spatial_whitening)
        numpy.save('./spykingcircus_output/sorter_output/whitened_recording', total_chunk)

    # chunk size == simulation duration
    # if not is_first, terminate simulation

    for gcount, gidx in enumerate(to_explore):
        # print "Node", comm.rank, "is analyzing chunk", gidx, "/", nb_chunks, " ..."
        # # We need to deal with the borders by taking chunks of size [0, chunck_size + template_shift].

        is_first = data_file.is_first_chunk(gidx, nb_chunks)
        is_last = data_file.is_last_chunk(gidx, nb_chunks)

        # Comment:
        # put the padding for different chunk
        if not (is_first and is_last):
            if is_last:
                padding = (-temp_3_shift, 0)
            elif is_first:
                padding = (0, temp_3_shift)
            else:
                padding = (-temp_3_shift, temp_3_shift)
        else:
            padding = (0, 0)

        if not is_first:
            print("terminate simulation")
            break

        result = {
            'spiketimes': [],
            'amplitudes': [],
            'templates': [],
        }

        if mse_error:
            mse_fit = {
            'spiketimes': [],
            'amplitudes': [],
            'templates': [],
        }

        result_debug = {
            'chunk_nbs': [],
            'iteration_nbs': [],
            'peak_nbs': [],
            'peak_local_time_steps': [],
            'peak_time_steps': [],
            'peak_scalar_products': [],
            'peak_solved_flags': [],
            'template_nbs': [],
            'success_flags': [],
        }

        local_chunk, t_offset = data_file.get_data(gidx, chunk_size, padding, nodes=nodes)

        len_chunk = len(local_chunk)
        if is_last:
            my_chunk_size = last_chunk_size
        else:
            my_chunk_size = chunk_size

        if do_spatial_whitening:
            local_chunk = numpy.dot(local_chunk, spatial_whitening)
                # save local_chunk
            print("local_chunk shape:", local_chunk.shape)
        if do_temporal_whitening:
            local_chunk = scipy.ndimage.filters.convolve1d(local_chunk, temporal_whitening, axis=0, mode='constant')


        # FALSE BY DEFAULT
        all_found_spikes = {}
        if collect_all:
            for i in range(n_e):
                all_found_spikes[i] = []

        local_peaktimes = [numpy.empty(0, dtype=numpy.uint32)]

        # done temporal, spatial filtering

        if ignore_artefacts:
            artefacts_peaktimes = [numpy.zeros(0, dtype=numpy.uint32)]
            artefacts_elecs = [numpy.zeros(0, dtype=numpy.uint32)]
            artefacts_amps = [numpy.zeros(0, dtype=numpy.float32)]

        if matched_filter:
            if sign_peaks in ['positive', 'both']:
                filter_chunk = scipy.ndimage.filters.convolve1d(local_chunk, waveform_pos, axis=0, mode='constant')
                for i in range(n_e):
                    peaktimes = scipy.signal.find_peaks(filter_chunk[:, i], height=matched_thresholds_pos[i])[0]

                    if ignore_artefacts:
                        artetimes = scipy.signal.find_peaks(numpy.abs(filter_chunk[:, i]), height=weird_thresh[i])[0]
                        to_keep = numpy.logical_not(numpy.in1d(peaktimes, artetimes))
                        peaktimes = peaktimes[to_keep]
                        artefacts_peaktimes.append(artetimes)
                        artefacts_elecs.append(i*numpy.ones(len(artetimes), dtype='uint32'))
                        artefacts_amps.append(local_chunk[artetimes, i])

                    local_peaktimes.append(peaktimes)
                    if collect_all:
                        all_found_spikes[i] += peaktimes.tolist()
            if sign_peaks in ['negative', 'both']:
                filter_chunk = scipy.ndimage.filters.convolve1d(local_chunk, waveform_neg, axis=0, mode='constant')
                for i in range(n_e):
                    peaktimes = scipy.signal.find_peaks(filter_chunk[:, i], height=matched_thresholds_neg[i])[0]

                    if ignore_artefacts:
                        artetimes = scipy.signal.find_peaks(numpy.abs(filter_chunk[:, i]), height=weird_thresh[i])[0]
                        to_keep = numpy.logical_not(numpy.in1d(peaktimes, artetimes))
                        peaktimes = peaktimes[to_keep]
                        artefacts_peaktimes.append(artetimes)
                        artefacts_elecs.append(i*numpy.ones(len(artetimes), dtype='uint32'))
                        artefacts_amps.append(local_chunk[artetimes, i])

                    local_peaktimes.append(peaktimes)
                    if collect_all:
                        all_found_spikes[i] += peaktimes.tolist()
            local_peaktimes = numpy.concatenate(local_peaktimes)

            if ignore_artefacts:
                artefacts_peaktimes = numpy.concatenate(artefacts_peaktimes)
                artefacts_elecs = numpy.concatenate(artefacts_elecs)
                artefacts_amps = numpy.concatenate(artefacts_amps)
        else:
            # BY DEFAULT
            # find peaktimes per electrodes
            for i in range(n_e):
                if sign_peaks == 'negative':
                    peaktimes = scipy.signal.find_peaks(-local_chunk[:, i], height=thresholds[i])[0]
                elif sign_peaks == 'positive':
                    peaktimes = scipy.signal.find_peaks(local_chunk[:, i], height=thresholds[i])[0]
                elif sign_peaks == 'both':
                    peaktimes = scipy.signal.find_peaks(numpy.abs(local_chunk[:, i]), height=thresholds[i])[0]
                else:
                    raise ValueError("Unexpected value %s" % sign_peaks)

                real_peaktimes = peaktimes + t_offset + padding[0]

                if ignore_artefacts:
                    artetimes = scipy.signal.find_peaks(numpy.abs(local_chunk[:, i]), height=weird_thresh[i])[0]
                    to_keep = numpy.logical_not(numpy.in1d(peaktimes, artetimes))
                    peaktimes = peaktimes[to_keep]
                    artefacts_peaktimes.append(artetimes)
                    artefacts_elecs.append(i*numpy.ones(len(artetimes), dtype='uint32'))
                    artefacts_amps.append(local_chunk[artetimes, i])

                local_peaktimes.append(peaktimes)
                if collect_all:
                    all_found_spikes[i] += peaktimes.tolist()
            local_peaktimes = numpy.concatenate(local_peaktimes)

            if ignore_artefacts:
                artefacts_peaktimes = numpy.concatenate(artefacts_peaktimes)
                artefacts_elecs = numpy.concatenate(artefacts_elecs)
                artefacts_amps = numpy.concatenate(artefacts_amps)

        # ignore what electrode spiked
        # save only the spike time
        local_peaktimes = numpy.unique(local_peaktimes)
        g_offset = t_offset + padding[0]

        if ignore_dead_times:
            dead_indices = numpy.searchsorted(all_dead_times, [t_offset, t_offset + my_chunk_size])
            if dead_indices[0] != dead_indices[1]:
                is_included = numpy.in1d(local_peaktimes + g_offset, all_dead_times[dead_indices[0]:dead_indices[1]])
                local_peaktimes = local_peaktimes[~is_included]

                if ignore_artefacts:
                    is_included = numpy.in1d(artefacts_peaktimes + g_offset, all_dead_times[dead_indices[0]:dead_indices[1]])
                    artefacts_peaktimes = artefacts_peaktimes[~is_included]
                    artefacts_elecs = artefacts_elecs[~is_included]
                    artefacts_amps = artefacts_amps[~is_included]

                local_peaktimes = numpy.sort(local_peaktimes)
        else:
            dead_indices = None  # default assignment (for PyCharm code inspection)

        # print "Removing the useless borders..."
        local_borders = (template_shift, len_chunk - template_shift)
        idx = (local_peaktimes >= local_borders[0]) & (local_peaktimes < local_borders[1])
        local_peaktimes = numpy.compress(idx, local_peaktimes)

        if ignore_artefacts:
            artefacts_peaktimes = artefacts_peaktimes + g_offset
            idx = (artefacts_peaktimes >= t_offset) & (artefacts_peaktimes < t_offset + my_chunk_size)
            artefacts_peaktimes = numpy.compress(idx, artefacts_peaktimes)
            artefacts_elecs = numpy.compress(idx, artefacts_elecs)
            artefacts_amps = numpy.compress(idx, artefacts_amps)

        if collect_all:
            for i in range(n_e):
                all_found_spikes[i] = numpy.array(all_found_spikes[i], dtype=numpy.uint32)

                if ignore_dead_times:
                    if dead_indices[0] != dead_indices[1]:
                        is_included = numpy.in1d(
                            all_found_spikes[i] + g_offset, all_dead_times[dead_indices[0]:dead_indices[1]]
                        )
                        all_found_spikes[i] = all_found_spikes[i][~is_included]
                        all_found_spikes[i] = numpy.sort(all_found_spikes[i])

                idx = (all_found_spikes[i] >= local_borders[0]) & (all_found_spikes[i] < local_borders[1])
                all_found_spikes[i] = numpy.compress(idx, all_found_spikes[i])

        nb_local_peak_times = len(local_peaktimes)

        if nb_local_peak_times > 0:
            # print "Computing the b (should full_gpu by putting all chunks on GPU if possible?)..."

            if collect_all or mse_error:
                c_local_chunk = local_chunk.copy()
            else:
                c_local_chunk = None  # default assignment (for PyCharm code inspection)

            # Currently, the code extracts only the recordings near the peaks
            # We should retrieve the whole recordings when processing the data in real time
            # Ex) The code should
            # 1) Perform spatial and temporal filtering
            # 2) Detect peaks
            # 3) We should retrieve the recordings near the peak
            # 4) And, perform dot product with the template (This procedure should consider the Spatial sparsity)
            # 5) Then, sort the spike (check if valid)
            # 6) After that, subtract the template data from the current sub_mat (due to the overlap)

            sub_mat = local_chunk[local_peaktimes[:, None] + temp_window]
            # sub_mat => (#of peaks, #timesteps for a spike, #of electrodes)
            sub_mat = sub_mat.transpose(2, 1, 0).reshape(size_window, nb_local_peak_times)

            #del local_chunk

            # all the distances
            # for the given peak times and the templates
            b = templates.dot(sub_mat)

            local_restriction = (t_offset, t_offset + my_chunk_size)
            #all_spikes = local_peaktimes + g_offset

            if collect_all:
                c_all_times = numpy.zeros((len_chunk, n_e), dtype=numpy.bool)
                c_min_times = numpy.maximum(numpy.arange(len_chunk) - template_shift, 0)
                c_max_times = numpy.minimum(numpy.arange(len_chunk) + template_shift + 1, len_chunk)
                for i in range(n_e):
                    c_all_times[all_found_spikes[i], i] = True
            else:
                c_all_times = None  # default assignment (for PyCharm code inspection)
                c_min_times = None  # default assignment (for PyCharm code inspection)
                c_max_times = None  # default assignment (for PyCharm code inspection)

            iteration_nb = 0

            # Comment:
            # Split the templates into two parts
            # 1) 0 ~ n_tm -1    => template 1 (primary matching)
            # 2) n_tm ~ N_tm -1 => template 2 (secondary matching)
            # We may need to divide the templates according to the peak electrode
            data = b[:n_tm, :]

            curr_idx = 0
            nb_miss_peak_times = 0
            nb_hit_peak_times = 0

            if not fixed_amplitudes:
                amp_index = numpy.searchsorted(splits, local_restriction[0], 'right')
                scaling = 1/(splits[amp_index] - splits[amp_index - 1])
                min_scalar_products = amp_limits[:, amp_index, 0] + (amp_limits[:, amp_index, 0] - amp_limits[:, amp_index+1, 0])*scaling
                max_scalar_products = amp_limits[:, amp_index, 1] + (amp_limits[:, amp_index, 1] - amp_limits[:, amp_index+1, 0])*scaling

                min_scalar_products = min_scalar_products[:, numpy.newaxis]
                max_scalar_products = max_scalar_products[:, numpy.newaxis]

                if templates_normalization:
                    min_sps = min_scalar_products * sub_norm_templates[:, numpy.newaxis]
                    max_sps = max_scalar_products * sub_norm_templates[:, numpy.newaxis]
                else:
                    min_sps = min_scalar_products * sub_norm_templates_2[:, numpy.newaxis]
                    max_sps = max_scalar_products * sub_norm_templates_2[:, numpy.newaxis]

            # for assignment error
            fifo_local_chunk = local_chunk.copy()

            while True:
                # Comment:
                # points with the valid template matching range
                # extract the valid indices for the template
                # valid_indices => CSR-like format with x, y index each
                # data : dot product value with primary template
                # data.shape = (n_t x n_spiketimes)
                is_valid = 0
                valid_indices = 0
                # real time
                if curr_idx < local_borders[0]:
                    curr_idx += 1
                    continue
                if curr_idx >= local_borders[1]: # if out of chunk index
                    break

                # perform peak detection
                # assume local chunk is padded
                spiked = False
                curr_dat = - fifo_local_chunk[curr_idx, :]
                peaked_elec_idx_list = []
                # maybe multiple electrode spike?
                for i in range(n_e):
                    if curr_dat[i] > thresholds[i]:
                        prev_dat_scalar = - fifo_local_chunk[curr_idx-1, i]
                        next_dat_scalar = - fifo_local_chunk[curr_idx+1, i]
                        if curr_dat[i] > prev_dat_scalar and curr_dat[i] > next_dat_scalar:
                            spiked = True
                            if g_offset + curr_idx < duration - template_shift:
                                peaked_elec_idx_list.append(i)
                                elec_peak_times[i].append(g_offset + curr_idx)

                if spiked == False:
                    curr_idx += 1
                    continue

                # NOTE spatial
                # get SPATIAL primary templates
                primary_templates_spatial = templates_spatial[:n_tm]
                primary_templates = templates[:n_tm]

                # find best template match for curr_spike_waveform == sub_mat[:,curr_idx]
                best_similarity = 0
                best_temp_idx = 0
                curr_spike_waveform = np.array(fifo_local_chunk[curr_idx-template_shift:curr_idx+template_shift+1, :]).T.ravel()

                #curr_spike_waveform[abs(curr_spike_waveform) < 1.25] = 0
                # Can we use 2x ??
                resolution = 2.25
                curr_spike_waveform = curr_spike_waveform / resolution
                curr_spike_waveform = np.round(curr_spike_waveform, 0)

                bit_prec = 5
                range_data = 2 ** (bit_prec - 1)
                curr_spike_waveform[curr_spike_waveform > range_data] = range_data
                curr_spike_waveform[curr_spike_waveform < -range_data] = -range_data

                curr_spike_waveform = curr_spike_waveform * resolution

                # to support multiple match
                best_temp_idx_list = []

                # NOTE toggle valid once
                none_is_valid = True # if at least one is found to be valid, this will be False

                # NOTE real time
                valid_number = 0
                if real_time:
                    if not valid_multiple:
                        # per-template threshold for validation
                        is_valid = (best_similarity > min_sps[best_temp_idx])*(best_similarity < max_sps[best_temp_idx])
                        if is_valid:
                            valid_number = 1 # this is fixed
                            none_is_valid = False
                    else:
                        # NOTE assume only one VALID INDEX
                        # FIX : check only for the connected template with peaked electrode
                        for idx, dict_template in enumerate(primary_templates_spatial):
                            # check if template_dict have key == peaked_elec_idx
                            is_template = False
                            for peaked_elec_idx in peaked_elec_idx_list:
                                if peaked_elec_idx in dict_template:
                                    is_template = True
                                    break
                            if is_template:
                                similarity = 0
                                for elec, template_waveform in dict_template.items():
                                    similarity += np.dot(curr_spike_waveform[n_t*elec:n_t*(elec+1)], template_waveform)
                                is_valid = (similarity > min_sps[idx]) * (similarity < max_sps[idx])
                                if is_valid:
                                    best_similarity = similarity
                                    best_temp_idx = idx
                                    # to support multiple match
                                    best_temp_idx_list.append(idx)
                                    valid_number += 1
                                    none_is_valid = False

                    if (valid_number > 1):
                        nb_miss_peak_times += 1

                    if (valid_number == 1):
                        nb_hit_peak_times += 1


                    if none_is_valid:
                        curr_idx += 1
                        continue # skip this value

                # NOTE real time
                if real_time:
                    # name translation to be compatible with the rest of the codes
                    #best_template_index = best_temp_idx
                    #best_template2_index = best_temp_idx + n_tm
                    best_template_index_list = best_temp_idx_list
                    peak_index = curr_idx
                    peak_scalar_product = best_similarity

                # true by default
                best_amp_list = []
                best_amp2_list = []
                best_amp_n = 0
                best_amp2_n = 0
                if templates_normalization:
                    # n_scalar : n_e * n_t
                    if real_time:
                        if spatial_sparse:
                            for best_template_index in best_template_index_list:
                                best_amp = 0
                                for elec, template_waveform in templates_spatial[best_template_index].items():
                                    # spatially sparse inner product -> this value is already saved in curr_similarity_buffer
                                    electrode_waveform = curr_spike_waveform[n_t*elec:n_t*(elec+1)]
                                    best_amp += np.dot(electrode_waveform, template_waveform) / n_scalar
                                best_amp_list.append(best_amp)
                                best_amp2_list.append(0)
                                # FOR ACCURACY CHECK
                                best_amp_n = best_amp / norm_templates[best_template_index]
                                best_amp2_n = 0

                # This procedure should be modified in the RealTime template matching (Refer to the 6-step procedure presented above)
                # subtract best match template from the spike waveform
                # subtract template from FIFO_FILTER, i.e., local_chunk

                # subtract overlap for ALL valid template_indices
                for i, best_template_index in enumerate(best_template_index_list):
                    dict_template_best = templates_spatial[best_template_index]
                    for elec, template_waveform in dict_template_best.items():
                        fifo_local_chunk[curr_idx - template_shift: curr_idx + template_shift + 1, elec] = fifo_local_chunk[curr_idx - template_shift: curr_idx + template_shift + 1, elec] - best_amp_list[i] * template_waveform

                # increment curr_idx, now we are done with this timestep
                curr_idx += 1

                # Add matching to the result.
                t_spike = peak_index + g_offset
                #t_spike = all_spikes[peak_index]

                if (t_spike >= local_restriction[0]) and (t_spike < local_restriction[1]):
                    for best_template_index in best_template_index_list:
                        custom_result['sorting'][best_template_index].append(t_spike)
                        result['spiketimes'] += [t_spike]
                        result['amplitudes'] += [(best_amp_n, best_amp2_n)]
                        result['templates'] += [best_template_index]
                elif mse_error:
                    mse_fit['spiketimes'] += [t_spike]
                    #mse_fit['amplitudes'] += [(best_amp_n, best_amp2_n)]
                    #mse_fit['templates'] += [best_template_index]

                # Save debug data.
                if debug:
                    result_debug['chunk_nbs'] += [gidx]
                    result_debug['iteration_nbs'] += [iteration_nb]
                    result_debug['peak_nbs'] += [peak_index]
                    result_debug['peak_local_time_steps'] += [local_peaktimes[peak_index]]
                    result_debug['peak_time_steps'] += [t_spike]
                    result_debug['peak_scalar_products'] += [peak_scalar_product]
                    #result_debug['peak_solved_flags'] += [b[best_template_index, peak_index]]
                    #result_debug['template_nbs'] += [best_template_index]
                    result_debug['success_flags'] += [True]

                iteration_nb += 1

            sum_miss_peak_times += nb_miss_peak_times
            sum_detected_peak_times += nb_miss_peak_times + nb_hit_peak_times

            spikes_to_write = numpy.array(result['spiketimes'], dtype=numpy.uint32)
            amplitudes_to_write = numpy.array(result['amplitudes'], dtype=numpy.float32)
            templates_to_write = numpy.array(result['templates'], dtype=numpy.uint32)

            spiketimes_file.write(spikes_to_write.tostring())
            amplitudes_file.write(amplitudes_to_write.tostring())
            templates_file.write(templates_to_write.tostring())

            if ignore_artefacts:
                arte_spiketimes_file.write(artefacts_peaktimes.astype(numpy.uint32).tostring())
                arte_electrodes_file.write(artefacts_elecs.tostring())
                arte_amplitudes_file.write(artefacts_amps.tostring())

            if mse_error:
                curve = numpy.zeros((len_chunk, n_e), dtype=numpy.float32)
                for spike, temp_id, amplitude in zip(result['spiketimes'], result['templates'], result['amplitudes']):
                    spike = spike - t_offset - padding[0]
                    if is_sparse:
                        tmp1 = templates[temp_id].toarray().reshape(n_e, n_t)
                        tmp2 = templates[temp_id + n_tm].toarray().reshape(n_e, n_t)
                    else:
                        tmp1 = templates[temp_id].reshape(n_e, n_t)
                        tmp2 = templates[temp_id + n_tm].reshape(n_e, n_t)

                    curve[spike - template_shift:spike + template_shift + 1, :] += (amplitude[0] * tmp1 + amplitude[1] * tmp2).T

                for spike, temp_id, amplitude in zip(mse_fit['spiketimes'], mse_fit['templates'], mse_fit['amplitudes']):
                    spike = spike - t_offset + padding[0]
                    if is_sparse:
                        tmp1 = templates[temp_id].toarray().reshape(n_e, n_t)
                        tmp2 = templates[temp_id + n_tm].toarray().reshape(n_e, n_t)
                    else:
                        tmp1 = templates[temp_id].reshape(n_e, n_t)
                        tmp2 = templates[temp_id + n_tm].reshape(n_e, n_t)
                    try:
                        curve[int(spike) - template_shift:int(spike) + template_shift + 1, :] += (amplitude[0] * tmp1 + amplitude[1] * tmp2).T
                    except Exception:
                        pass
                mse = numpy.linalg.norm((curve - c_local_chunk)[-padding[0]:-padding[1]])
                nb_points = len(curve) - (padding[1] - padding[0])
                mse_ratio = mse/(numpy.sqrt(nb_points)*stds_norm)
                mse_to_write = numpy.array([g_offset, mse_ratio], dtype=numpy.float32)
                mse_file.write(mse_to_write.tostring())

            if collect_all:

                for temp, spike in zip(templates_to_write, spikes_to_write - g_offset):
                    c_all_times[c_min_times[spike]:c_max_times[spike], neighbors[temp]] = False

                gspikes = numpy.where(numpy.sum(c_all_times, 1) > 0)[0]
                c_all_times = numpy.take(c_all_times, gspikes, axis=0)
                c_local_chunk = numpy.take(c_local_chunk, gspikes, axis=0) * c_all_times

                if sign_peaks == 'negative':
                    bestlecs = numpy.argmin(c_local_chunk, 1)
                    if matched_filter:
                        threshs = -matched_thresholds_neg[bestlecs]
                    else:
                        threshs = -thresholds[bestlecs]
                    idx = numpy.where(numpy.min(c_local_chunk, 1) < threshs)[0]
                elif sign_peaks == 'positive':
                    bestlecs = numpy.argmax(c_local_chunk, 1)
                    if matched_filter:
                        threshs = matched_thresholds_pos[bestlecs]
                    else:
                        threshs = thresholds[bestlecs]
                    idx = numpy.where(numpy.max(c_local_chunk, 1) > threshs)[0]
                elif sign_peaks == 'both':
                    c_local_chunk = numpy.abs(c_local_chunk)
                    bestlecs = numpy.argmax(c_local_chunk, 1)
                    if matched_filter:
                        threshs = numpy.minimum(matched_thresholds_neg[bestlecs], matched_thresholds_pos[bestlecs])
                    else:
                        threshs = thresholds[bestlecs]
                    idx = numpy.where(numpy.max(c_local_chunk, 1) > threshs)[0]
                else:
                    raise ValueError("Unexpected value %s" % sign_peaks)

                gspikes = numpy.take(gspikes, idx)
                bestlecs = numpy.take(bestlecs, idx)
                gspikes_to_write = numpy.array(gspikes + g_offset, dtype=numpy.uint32)
                gtemplates_to_write = numpy.array(bestlecs, dtype=numpy.uint32)

                garbage_times_file.write(gspikes_to_write.tostring())
                garbage_temp_file.write(gtemplates_to_write.tostring())

            if debug:
                # Write debug data to debug files.
                for field_label, field_dtype, field_file in [
                    ('chunk_nbs', numpy.uint32, chunk_nbs_debug_file),
                    ('iteration_nbs', numpy.uint32, iteration_nbs_debug_file),
                    ('peak_nbs', numpy.uint32, peak_nbs_debug_file),
                    ('peak_local_time_steps', numpy.uint32, peak_local_time_steps_debug_file),
                    ('peak_time_steps', numpy.uint32, peak_time_steps_debug_file),
                    ('peak_scalar_products', numpy.float32, peak_scalar_products_debug_file),
                    ('peak_solved_flags', numpy.float32, peak_solved_flags_debug_file),
                    ('template_nbs', numpy.uint32, template_nbs_debug_file),
                    ('success_flags', numpy.bool, success_flags_debug_file),
                ]:
                    field_to_write = numpy.array(result_debug[field_label], dtype=field_dtype)
                    field_file.write(field_to_write.tostring())

    # end of local chunk iteration
    # sum up the stats

    print("--------------------------")
    print(sum_miss_peak_times, "missed out of", sum_detected_peak_times)
    print("--------------------------")

    # save elec_peak_times as txt file
    for i in range(n_e):
        elec_peak_times[i] = np.unique(np.array(elec_peak_times[i]))
    if comm.rank == 0:
        debug_file = open(f'./multicore_spike_out{comm.rank}.dat', 'w')
        for i in range(n_e + n_tm):
            #for time in elec_peak_times[i]:
            #debug_file.write('\n')
            if i < n_e:
                debug_file.write(str(i) + ": " + str(list(elec_peak_times[i])) + '\n')
            else:
                idx = i - n_e
                debug_file.write(str(i) + ": " + str(list(custom_result['sorting'][idx])) + '\n')
        debug_file.close()

    sys.stderr.flush()

    spiketimes_file.flush()
    os.fsync(spiketimes_file.fileno())
    spiketimes_file.close()

    amplitudes_file.flush()
    os.fsync(amplitudes_file.fileno())
    amplitudes_file.close()

    templates_file.flush()
    os.fsync(templates_file.fileno())
    templates_file.close()

    #stats_file.flush()
    #os.fsync(stats_file.fileno())
    #stats_file.close()

    if collect_all:

        garbage_temp_file.flush()
        os.fsync(garbage_temp_file.fileno())
        garbage_temp_file.close()

        garbage_times_file.flush()
        os.fsync(garbage_times_file.fileno())
        garbage_times_file.close()

    if mse_error:
        mse_file.flush()
        os.fsync(mse_file.fileno())
        mse_file.close()

    if ignore_artefacts:
        arte_spiketimes_file.flush()
        os.fsync(arte_spiketimes_file.fileno())
        arte_spiketimes_file.close()

        arte_electrodes_file.flush()
        os.fsync(arte_electrodes_file.fileno())
        arte_electrodes_file.close()

        arte_amplitudes_file.flush()
        os.fsync(arte_amplitudes_file.fileno())
        arte_amplitudes_file.close()

    if debug:
        # Close debug files.
        for field_file in [
            chunk_nbs_debug_file,
            iteration_nbs_debug_file,
            peak_nbs_debug_file,
            peak_local_time_steps_debug_file,
            peak_time_steps_debug_file,
            peak_scalar_products_debug_file,
            peak_solved_flags_debug_file,
            template_nbs_debug_file,
            success_flags_debug_file,
        ]:
            field_file.flush()
            os.fsync(field_file.fileno())
            field_file.close()

    comm.Barrier()

    if SHARED_MEMORY:
        for memory in mpi_memory_1 + mpi_memory_2:
            memory.Free()
        if ignore_dead_times:
            mpi_memory_3.Free()

    if comm.rank == 0:
        io.collect_data(comm.size, params, erase=True)

        if ignore_artefacts:
            io.collect_artefacts(comm.size, params, erase=True)

    data_file.close()

