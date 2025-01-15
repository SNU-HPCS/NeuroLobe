cimport cython
cimport numpy as np
import numpy as np
import sys, os
import copy
from math import log2, ceil

import GlobalVars as GV
from collections import deque

import Task
from Placement import combine_entry

def coord_to_ind(x, y):
    return int(y * GV.sim_params['max_core_x'] + x)

def ind_to_coord(ind):
    return int(ind % GV.sim_params['max_core_x']), int(ind / GV.sim_params['max_core_x'])

def closest_ind(target, core_list):
    ind_list = [ind_to_coord(core) for core in core_list]
    min_distance = float('inf')  # Initialize with a large value
    closest = None

    for ind in ind_list:
        distance = np.linalg.norm(np.array(target) - np.array(ind))
        if distance < min_distance:
            min_distance = distance
            closest = ind

    return closest

cpdef init_spike_sync_topology(list route_forward, list route_mem, list cores, int external):
    cdef object queue

    cdef list sync_role = [None for _ in range(GV.sim_params['used_core_num'])]

    sync_info = [{} for _ in range(GV.sim_params['used_core_num'])]
    sync_dat = [{'parent': -1, 'children' : [], 'num_children' : 0} for _ in range(GV.sim_params['used_core_num'])]

    centre_x = GV.sim_params['max_core_x'] // 2
    centre_y = GV.sim_params['max_core_y'] // 2

    internal_cores = [core for core in cores if core < GV.sim_params['max_core_x']* GV.sim_params['max_core_y']]

    root_x, root_y  = closest_ind((centre_x, centre_y), internal_cores)
    root_ind = coord_to_ind(root_x, root_y)
    sync_role[root_ind] = 'root'

    external_cores = [ind_to_coord(core) for core in cores if core >= GV.sim_params['max_core_x']* GV.sim_params['max_core_y']]
    external_parent_id = []
    if external:
        for x, y in external_cores:
            p_x = x
            p_y = y - 1
            while True:
                if coord_to_ind(p_x, p_y) in internal_cores:
                    external_parent_id.append((p_x, p_y))
                    break
                if p_y > 0:
                    p_y = p_y - 1
                else:
                    p_x, p_y = closest_ind((centre_x, centre_y), internal_cores)
                    external_parent_id.append((p_x, p_y))
                    break

    queue = deque()
    queue.append( ((root_x, root_y), (root_x, root_y)) ) # current coord, actual parent coord

    while queue:
        (x, y), (p_x, p_y) = queue.popleft()
        core_ind = coord_to_ind(p_x, p_y)

        if external:
            if (x, y) in external_cores:
                sync_role[external_ind] = 'non_root'
                continue

        if x > 0:
            left_ind = coord_to_ind(x-1, y)
            if sync_role[left_ind] == None:
                if left_ind in cores:
                    sync_role[left_ind] = 'non_root'
                    #####################################################################
                    sync_dat[left_ind]['parent'] = core_ind
                    sync_dat[core_ind]['num_children'] += 1
                    sync_dat[core_ind]['children'].append(left_ind)
                    queue.append( ((x-1, y), (x-1, y)) )
                else:
                    sync_role[left_ind] = 'path'
                    queue.append( ((x-1, y), (p_x, p_y)) )
        if x < GV.sim_params['max_core_x'] - 1:
            right_ind = coord_to_ind(x+1, y)
            if sync_role[right_ind] == None:
                if right_ind in cores:
                    sync_role[right_ind] = 'non_root'
                    #####################################################################
                    sync_dat[right_ind]['parent'] = core_ind
                    sync_dat[core_ind]['num_children'] += 1
                    sync_dat[core_ind]['children'].append(right_ind)
                    queue.append( ((x+1, y), (x+1, y)) )
                else:
                    sync_role[right_ind] = 'path'
                    queue.append( ((x+1, y), (p_x, p_y)) )
        if y > 0:
            bot_ind = coord_to_ind(x, y-1)
            if sync_role[bot_ind] == None:
                if bot_ind in cores:
                    sync_role[bot_ind] = 'non_root'
                    #####################################################################
                    sync_dat[bot_ind]['parent'] = core_ind
                    sync_dat[core_ind]['num_children'] += 1
                    sync_dat[core_ind]['children'].append(bot_ind)
                    queue.append( ((x, y-1), (x, y-1)) )
                else:
                    sync_role[bot_ind] = 'path'
                    queue.append( ((x, y-1), (p_x, p_y)) )
        if y < GV.sim_params['max_core_y'] - 1:
            top_ind = coord_to_ind(x, y+1)
            if sync_role[top_ind] == None:
                if top_ind in cores:
                    sync_role[top_ind] = 'non_root'
                    #####################################################################
                    sync_dat[top_ind]['parent'] = core_ind
                    sync_dat[core_ind]['num_children'] += 1
                    sync_dat[core_ind]['children'].append(top_ind)
                    queue.append( ((x, y+1), (x, y+1)) )
                else:
                    sync_role[top_ind] = 'path'
                    queue.append( ((x, y+1), (p_x, p_y)) )

        if external:
            if (x, y) in external_parent_id:
                assert(coord_to_ind(x, y) in cores)
                ext_x, ext_y = external_cores[external_parent_id.index((x, y))]
                external_ind = coord_to_ind(ext_x, ext_y)
                assert(sync_role[external_ind] != 'root')
                sync_role[external_ind] = 'non_root'
                # we should set the parent for the external module
                sync_dat[external_ind]['parent'] = core_ind
                sync_dat[core_ind]['num_children'] += 1
                sync_dat[core_ind]['children'].append(external_ind)
                queue.append( ((ext_x, ext_y), (ext_x, ext_y)) )

        if sync_dat[core_ind]['num_children'] == 0 and core_ind != root_ind:
            if core_ind in internal_cores:
                sync_role[core_ind] = 'non_root'
            else:
                sync_role[core_ind] = 'path'

    # print(sync_dat)

    # Update the routing table according to the parent and child
    # This may change when we utilize different cores for different functions
    for core_ind in range(GV.sim_params['used_core_num']):
        ###
        sync_info[core_ind]['sync_role'] = sync_role[core_ind]
        sync_info[core_ind]['num_children'] = sync_dat[core_ind]['num_children']
        # Append the parent
        route_forward[core_ind]['sync'] = [0, 1]
        route_mem[core_ind]['sync'] = [{'dst core': sync_dat[core_ind]['parent'], 'pid': -1}]

        # We append self to the route memory
        if core_ind == root_ind:
            route_forward[core_ind]['commit'] = [0, sync_dat[core_ind]['num_children'] + 1]
            route_mem[core_ind]['commit'] = []
            for child in sync_dat[core_ind]['children']:
                route_mem[core_ind]['commit'].append({'dst core': child, 'pid': -1})
            route_mem[core_ind]['commit'].append({'dst core': root_ind, 'pid': -1})
        else:
            route_forward[core_ind]['commit'] = [0, sync_dat[core_ind]['num_children']]
            route_mem[core_ind]['commit'] = []
            for child in sync_dat[core_ind]['children']:
                route_mem[core_ind]['commit'].append({'dst core': child, 'pid': -1})

    return sync_info
    
# Map neurons to cores
cpdef mapping(list gid_to_core, list tid_to_core, list lid_to_gid, mapping_filename, int gtask_id, conn_list, p_unit_name, p_unit_table):
    cdef int core_num
    cdef int neu
    cdef int neu_index
    cdef int core_index
    cdef int core

    core_num = GV.sim_params['used_core_num']

    num_unit_per_core = [[0 for _ in range(GV.sim_params['num_unit_types'][gtask_id])] for _ in range(core_num)]

    node_list = np.load(mapping_filename, allow_pickle=True, encoding='bytes')['node_list'].item()

    # Translate per-task state core_id into global core_id
    temp_node_list = node_list.copy()
    node_list = {}

    print("\nTask", gtask_id, ":",  GV.sim_params['task_to_core'][gtask_id])
    for core_id in temp_node_list.keys():
        if core_id in GV.sim_params['task_to_core'][gtask_id]:
            node_list[core_id] = temp_node_list[core_id]

    # Calculate the number of units mapped to each core
    for core in GV.sim_params['task_to_core'][gtask_id]:
        for lid in range(len(node_list[core])):
            gid = node_list[core][lid]
            tid, type_id = GV.sim_params['gid_to_tid'][gid]
            gid_to_core[gid] = (core, lid)
            tid_to_core[type_id][tid] = (core, num_unit_per_core[core][type_id])
            num_unit_per_core[core][type_id] += 1
    
    # Add a partial accumulation unit
    if p_unit_table:
        unit_to_unit = [[] for _ in range(GV.sim_params['num_unit'][gtask_id][GV.sim_params['p_unit'][gtask_id]['src_type']])]
        for idx in range(len(conn_list)):
            src, dst, _, _ = conn_list[idx]
            unit_to_unit[src].append(dst)
    
        type_id = GV.sim_params['num_unit_types'][gtask_id]-1  # partial templates / pseudo neurons
        tid_to_core.append([])
        gid = GV.sim_params['num_total_unit'][gtask_id]
        tid = GV.sim_params['total_'+p_unit_name+'_num']

        for core in GV.sim_params['task_to_core'][gtask_id]:
                # Add offset to the lid
                lid = 0
                for types in range(type_id-1):
                    lid += num_unit_per_core[core][types]
                # lid = num_unit_per_core[core][0] + num_unit_per_core[core][1]
                for src in range(len(unit_to_unit)):
                    if src in node_list[core]:
                        for dst in unit_to_unit[src]:
                            if p_unit_table[core][dst] == -1:
                                GV.sim_params['gid_to_tid'].append((tid, type_id))
                                gid_to_core.append((core, lid))
                                tid_to_core[type_id].append((core, num_unit_per_core[core][type_id]))
                                num_unit_per_core[core][type_id] += 1
                                p_unit_table[core][dst] = tid

                                lid += 1
                                gid += 1
                                tid +=1

        GV.sim_params['total_'+p_unit_name+'_num'] = tid
        GV.sim_params['num_unit'][gtask_id].append(GV.sim_params['total_'+p_unit_name+'_num'])
        GV.sim_params['num_total_unit'][gtask_id] += GV.sim_params['total_'+p_unit_name+'_num']

    GV.sim_params['num_unit_per_core'][gtask_id] = num_unit_per_core

    # update lid_to_gid for template matching task
    lid_to_gid = [[[] for _ in range(GV.sim_params['num_unit_types'][gtask_id])] for _ in range(core_num)]

    # map state unit to global neuron (for debugging purpose)
    for core in range(core_num):
        for type_id in range(len(num_unit_per_core[core])):
            num_unit = num_unit_per_core[core][type_id]
            lid_to_gid[core][type_id] = [0 for _ in range(num_unit)]

    for gid in range(GV.sim_params['num_total_unit'][gtask_id]):
        tid, type_id = GV.sim_params['gid_to_tid'][gid]
        core_index, lid = tid_to_core[type_id][tid]
        if core_index >= 0:
            lid_to_gid[core_index][type_id][lid] = gid

    GV.tid_to_core[gtask_id] = tid_to_core
    GV.gid_to_core[gtask_id] = gid_to_core
    GV.lid_to_gid[gtask_id] = lid_to_gid

cpdef load_external_stimulus(external_filename):
    return np.load(external_filename, allow_pickle=True, mmap_mode='r')

cpdef init_state_mem(initial_state_mem, #
                     name,              # the name of the state
                     type_id,           # either type_id or (src_type_id / dst_type_id)
                     initial_state,     # Initial entry values
                     share,             # If the entries are shared
                     gtask_id):
    core_num = GV.sim_params['used_core_num']
    # The data is for the metadata of the source states 
    if type(type_id) is tuple:
        num_unit_per_core = [len(GV.pid_to_gid[gtask_id][core_id][type_id[0]][type_id[1]]) \
                             for core_id in range(core_num)]
        # We do not allow non-shared entry in such a case
        # (The number of entries are determined only after mapping)
        assert(share == True)
    # The data if for the state states
    else:
        num_unit_per_core = [GV.sim_params['num_unit_per_core'][gtask_id][core_id][type_id] \
                             for core_id in range(core_num)]
    if share:
        temp_data = [[copy.deepcopy(initial_state) for _ in range(num_unit_per_core[core_id])] \
                                                   for core_id in range(core_num)]
    else:
        # Initialize the temporary data
        temp_data = [[None for _ in range(num_unit_per_core[core_id])] \
                           for core_id in range(core_num)]
        for tid in range(GV.sim_params['num_unit'][gtask_id][type_id]):
            core_id, lid = GV.tid_to_core[gtask_id][type_id][tid]
            temp_data[core_id][lid] = initial_state[tid]
    for core_id in range(core_num):
        initial_state_mem[core_id][name] = temp_data[core_id]


cpdef init_hist_mem(initial_hist_mem, #
                    max_delay,
                    name,              # the name of the state
                    type_id,           # either type_id or (src_type_id / dst_type_id)
                    initial_state,     # Initial entry values
                    gtask_id,
                    share = True,
                    precision = 32):

    precision = 2 ** ceil(log2(precision))
    core_num = GV.sim_params['used_core_num']

    # The data is for the metadata of the source states 
    num_unit_per_core = [GV.sim_params['num_unit_per_core'][gtask_id][core_id][type_id] \
                         for core_id in range(core_num)]
    
    temp_data = [[[] for lid in range(num_unit_per_core[core_id])] \
                     for core_id in range(core_num)]

    for tid in range(GV.sim_params['num_unit'][gtask_id][type_id]):
        core_id, lid = GV.tid_to_core[gtask_id][type_id][tid]
        offset = 0
        for pos in range(max_delay + 1):
            if offset + precision > GV.MEM_WIDTH['hist_mem'] or offset == 0:
                new_entry = {}
                new_entry[0] = initial_state if share else initial_state[tid]
                temp_data[core_id][lid].append(new_entry)
                offset = precision
            else:
                temp_data[core_id][lid][-1][offset] = initial_state if share else initial_state[tid]
                offset += precision

    for core_id in range(core_num):
        initial_hist_mem[core_id][name] = (temp_data[core_id], precision)

#cpdef init_stack_mem(initial_stack_mem, #
#                     name,              # the name of the state
#                     type_id,           # either type_id or (src_type_id / dst_type_id)
#                     gtask_id):
#
#    core_num = GV.sim_params['used_core_num']
#
#    # The data is for the metadata of the source states 
#    if type_id == None:
#        num_unit_per_core = [1 for core_id in range(core_num)]
#    elif type(type_id) is tuple:
#        num_unit_per_core = [len(GV.pid_to_gid[gtask_id][core_id][type_id[0]][type_id[1]]) \
#                             for core_id in range(core_num)]
#    else:
#        num_unit_per_core = [GV.sim_params['num_unit_per_core'][gtask_id][core_id][type_id] \
#                             for core_id in range(core_num)]
#
#    temp_data = [[None for _ in range(num_unit_per_core[core_id])] \
#                                    for core_id in range(core_num)]
#
#    for core_id in range(core_num):
#        initial_stack_mem[core_id][name] = temp_data[core_id]

cpdef init_ack_left_mem(initial_ack_left_mem, name, gtask_id):

    core_num = GV.sim_params['used_core_num']
    # The data is for the metadata of the source states 
    entry_per_core = [[0 for _ in range(1)] \
                       for core_id in range(core_num)]

    for core_id in range(core_num):
        initial_ack_left_mem[core_id][name] = entry_per_core[core_id]

cpdef init_ack_num_mem(initial_ack_num_mem, name, gtask_id):

    core_num = GV.sim_params['used_core_num']
    # The data is for the metadata of the source states 
    entry_per_core = [[0 for _ in range(core_num)] \
                       for core_id in range(core_num)]

    for core_id in range(core_num):
        initial_ack_num_mem[core_id][name] = entry_per_core[core_id]

# Check if there exists an internal fragmentation (if so, align the bit precision)
cpdef align_entry(entry):
    # Parse the entry info
    outer_entry_info = []
    inner_entry_info = []
    entry_length = 0
    for dat in entry:
        name = dat[0]
        loop = 'inner' if isinstance(dat[1], list) else 'outer'
        data = dat[1] if isinstance(dat[1], list) else [dat[1]]
        prec = dat[2]
        new_entry = {}
        new_entry['name'] = name
        new_entry['prec'] = prec
        new_entry['num'] = len(data)
        # append to the sptial and inner 
        if loop == 'outer':
            outer_entry_info.append(new_entry)
        else:
            if entry_length: assert(entry_length == len(data))
            else:            entry_length = len(data)
            inner_entry_info.append(new_entry)
    
    # Perform heuristic algorithm for the inner loop first
    outer_entry_info, outer_offset_info, outer_loop_offset, outer_width_info = \
        combine_entry(outer_entry_info)
    inner_entry_info, inner_offset_info, inner_loop_offset, inner_width_info = \
        combine_entry(inner_entry_info)

    align_info = {'outer' : outer_entry_info, 'inner' : inner_entry_info}
    offset_info = {**outer_offset_info, **inner_offset_info}
    loop_info = {'outer' : outer_loop_offset, 'inner' : inner_loop_offset}

    width_info = {**outer_width_info, **inner_width_info}

    return {'align' : align_info, 'offset' : offset_info, 'loop' : loop_info, 'width' : width_info}

# allocate a new entry
cpdef allocate_entry(entry, mem, offset, align_info):
    outer_entry_data = []
    inner_entry_data = []

    inner_loop_length = 0
    for dat in entry:
        loop = 'inner' if isinstance(dat[1], list) else 'outer'
        data = dat[1] if isinstance(dat[1], list) else [dat[1]]
        # Check if the data requires an inner iteration
        if loop == 'outer':
            outer_entry_data.append(data)
        else:
            inner_entry_data.append(data)
            inner_loop_length = len(data)

    #
    outer_entry_info = align_info['outer']
    inner_entry_info = align_info['inner']
    
    # reshapse the outer loop-related data
    current_offset = offset['outer']
    for entry_info in outer_entry_info:
        data = outer_entry_data[entry_info['original']]
        prec = entry_info['prec']

        if offset['outer'] + prec > GV.MEM_WIDTH['corr_mem'] or offset['outer'] == 0:
            new_entry = {}
            new_entry[0] = data[0]
            # append a new entry
            mem['outer'].append(new_entry)
            #assert(offset['outer'] % GV.MEM_WIDTH == 0)
            offset['outer'] = prec
            current_offset = prec
        else:
            mem['outer'][-1][current_offset] = data[0]
            offset['outer'] += prec
            current_offset += prec

    current_offset = offset['inner']
    # reshapse the inner loop-related data
    for ind in range(inner_loop_length):
        for entry_info in inner_entry_info:
            data_list = inner_entry_data[entry_info['original']]
            prec = entry_info['prec']

            if offset['inner'] + prec > GV.MEM_WIDTH['corr_mem'] or offset['inner'] == 0:
                new_entry = {}
                new_entry[0] = data_list[ind]
                # append a new entry
                mem['inner'].append(new_entry)
                #assert(offset['inner'] % GV.MEM_WIDTH == 0)
                offset['inner'] = prec
                current_offset = prec
            else:
                mem['inner'][-1][current_offset] = data_list[ind]
                offset['inner'] += prec
                current_offset += prec

    return mem, offset

cpdef ceil_div(a, b):
    return -(a // -b)

cpdef init_ack_stack_mem(initial_ack_stack_mem, name, gtask_id):

    core_num = GV.sim_params['used_core_num']
    # The data is for the metadata of the source states 
    entry_per_core = [[0 for _ in range(core_num)] \
                       for core_id in range(core_num)]

    for core_id in range(core_num):
        initial_ack_stack_mem[core_id][name] = entry_per_core[core_id]


# load required connections
cpdef init_corr_mem (conn_filename, mapping_filename, cores, gtask_id):
    cdef int core_num
    cdef int core
    cdef int num_neu
    cdef int x
    cdef int arr_ind = 0

    cdef list gid_to_core
    cdef list lid_to_gid
    cdef list neu_states_dat

    cdef list corr_mem

    cdef list pid_to_gid
    cdef list gid_to_pid

    cdef list route_forward
    cdef list route_mem
    cdef list core_mem_ind
    
    # added variables

    # +1 to enable external connection
    core_num = GV.sim_params['used_core_num']
    num_types = GV.sim_params['num_unit_types'][gtask_id]
    num_total_unit = GV.sim_params['num_total_unit'][gtask_id] # total number of 'units' -> including 'templates' and 'electrodes'

    gid_to_core = [(-1, -1) for _ in range(num_total_unit)]
    tid_to_core = [[(-1, -1) for _ in range(GV.sim_params['num_unit'][gtask_id][type_id])] \
                             for type_id in range(GV.sim_params['num_unit_types'][gtask_id])]

    lid_to_gid = [[[] for _ in range(GV.sim_params['num_unit_types'][gtask_id])] \
                      for _ in range(core_num)]


    initial_conn_list = np.load(conn_filename, allow_pickle=True)

    task_type = Task.get_task(gtask_id).type
    
    p_unit_dict = GV.sim_params['p_unit'][gtask_id]
    use_partial = GV.sim_params['use_partial'][gtask_id]
    # print("USE PARTIAL", use_partial)
    if len(p_unit_dict) != 0 and use_partial:
        p_unit_name = str(p_unit_dict['unit_type'])
        src_type = int(p_unit_dict['src_type'])
        dst_type = int(p_unit_dict['dst_type'])

        GV.sim_params['total_'+p_unit_name+'_num'] = 0
        GV.sim_params['num_unit_types'][gtask_id] += 1
        GV.sim_params['unit_type_name'][gtask_id].append(p_unit_name)
        p_unit_type = GV.sim_params['num_unit_types'][gtask_id]-1

        target_conn_list = initial_conn_list[src_type][dst_type]

        p_unit_table = [[-1 for _ in range(GV.sim_params['num_unit'][gtask_id][dst_type])] for _ in range(core_num)]
        mapping(gid_to_core, tid_to_core, lid_to_gid, mapping_filename, gtask_id, target_conn_list, p_unit_name, p_unit_table)

        num_types = GV.sim_params['num_unit_types'][gtask_id]
        num_total_unit = GV.sim_params['num_total_unit'][gtask_id] # total number of 'units' (neu / bci_neu / bci / pseudo_neu )
        
        total_list = [[[] for _ in range(num_types)] for _ in range(num_types)]
        for i, connection in enumerate(target_conn_list):
            src = connection[0]
            dst = connection[1]
            src_core = gid_to_core[src][0]
            p_unit = p_unit_table[src_core][dst]
            if p_unit != -1:
                total_list[src_type][p_unit_type].append((src, p_unit, connection[2], connection[3])) # changed total_list index because we now have new type 'bci'
                if p_unit not in [entry[0] for entry in total_list[p_unit_type][dst_type]]:
                    total_list[p_unit_type][dst_type].append((p_unit, dst, False, None))

        total_list[src_type][p_unit_type] = sorted(total_list[src_type][p_unit_type])
        total_list[p_unit_type][dst_type] = sorted(total_list[p_unit_type][dst_type])
        for i in range(num_types-1):
            if i == p_unit_type:
                continue
            for j in range(num_types-1):
                if i == src_type and (j == p_unit_type or j == dst_type):
                    continue
                total_list[i][j] = initial_conn_list[i][j]
        initial_conn_list = total_list
    else:
        print("Mapping without partial unit")
        mapping(gid_to_core, tid_to_core, lid_to_gid, mapping_filename, gtask_id, None, None, None)


    ### We initialize all the correlation-related memory in advance
    # 1) corr_mem => the data stored in the correlation memory
    corr_mem = [[[[] for _ in range(num_types)] \
                     for _ in range(num_types)] \
                     for core in range(core_num)]
    
    # 2) corr_forward => the metadata to indirect the memory address
    corr_forward = [[[[] for _ in range(num_types)] \
                         for _ in range(num_types)] \
                         for _ in range(core_num)]

    # pid: The pid indicates the id of the source @ the destination core
    pid_to_gid = [[[[] for _ in range(num_types)] \
                       for _ in range(num_types)] \
                       for _ in range(core_num)]

    gid_to_pid = [[[{} for _ in range(num_types)] \
                       for _ in range(num_types)] \
                       for _ in range(core_num)]

    # route_forward: indirection table for the source
    # -> for a source neuron, an indirection table provides start & end address of route_mem memory
    route_forward = [[[[0 for _ in range(GV.sim_params['num_unit_per_core'][gtask_id][core][src_type_id] + 1)] \
                          for dst_type_id in range(num_types)] \
                          for src_type_id in range(num_types)] \
                          for core in range(core_num)]
    # returns the list of destination cores
    route_mem = [[[[] for _ in range(num_types)] \
                      for _ in range(num_types)] \
                      for core in range(core_num)]

    print("\nEstablishing connection ...")

    # Iterate multiple times
    for src_type_id in range(num_types):
        for dst_type_id in range(num_types):
            conn_list = initial_conn_list[src_type_id][dst_type_id]
            num_src_unit = GV.sim_params['num_unit'][gtask_id][src_type_id]
            num_dst_unit = GV.sim_params['num_unit'][gtask_id][dst_type_id]
            if not conn_list: continue
            
            # set the size of an entry
            src_name = GV.sim_params['unit_type_name'][gtask_id][src_type_id]
            dst_name = GV.sim_params['unit_type_name'][gtask_id][dst_type_id]
            conn_name = str(src_name) + '_to_' + str(dst_name)

            # Initialize metadata
            route_mem_ind = [0 for _ in range(core_num)]
            corr_mem_ind = [0 for _ in range(core_num)]
            corr_mem_ind_prev = [0 for _ in range(core_num)]

            max_src = -1
            arr_ind = 0
            for src_id in range(num_src_unit):
                # Retrieve the source core and 
                # the its lid in the source core
                (src_core_id, src_lid) = tid_to_core[src_type_id][src_id]

                # Use the data to generate the routing table
                # route_mem_ind => indicates the current size allocated to the route_mem
                route_forward[src_core_id][src_type_id][dst_type_id][src_lid] = \
                    route_mem_ind[src_core_id]

                # Tracks the destination core connected to the source neuron
                dst_core_set = []
                # make an array of dictionary
                src_to_addr = [{} for _ in range(core_num)]

                # corr_mem_ind => indicates the current size allocated to the corr_mem
                for core in range(core_num):
                    corr_mem_ind_prev[core] = corr_mem_ind[core]

                # Partially allocate the outer and inner memory data
                mem = [{'outer': [], 'inner': []} for _ in range(core_num)]
                # Set different memory offset for inner and outer
                mem_offset = [{'outer':0, 'inner':0} for _ in range(core_num)]
                num_iter = [0 for _ in range(core_num)]
                corr_mem_offset = [0 for _ in range(core_num)]

                # Identify the destination addresses for a given source
                while not arr_ind == len(conn_list):
                    conn = conn_list[arr_ind]
                    src = int(conn[0])
                    dst = int(conn[1])
                    has_entry = bool(conn[2])
                    entry = conn[3]

                    if max_src <= src: max_src = src
                    else: 
                        print(max_src, src)
                        assert(0)
                    if not src == src_id: break

                    # Retrieve the info of the destination 
                    (dst_core_id, dst_lid) = tid_to_core[dst_type_id][dst]
                    
                    if has_entry:
                        entry.insert(0, ('lid', dst_lid, GV.LID_PRECISION))
                    else:
                        entry = [('lid', dst_lid, GV.LID_PRECISION)]

                    # increment the number of iterations
                    num_iter[dst_core_id] += 1

                    # Replace the entry for the outer and inner loop
                    if not conn_name in GV.loop_info[gtask_id]:
                        GV.loop_info[gtask_id][conn_name]= align_entry(entry)

                    # Utilize the loop offset when allocating the memory
                    mem[dst_core_id], mem_offset[dst_core_id] = \
                        allocate_entry(entry, mem[dst_core_id], mem_offset[dst_core_id], \
                                       GV.loop_info[gtask_id][conn_name]['align'])
                    
                    # Set the pid for the connected destination core
                    if src not in pid_to_gid[dst_core_id][src_type_id][dst_type_id]:
                        gid_to_pid[dst_core_id][src_type_id][dst_type_id][src] = \
                            len(pid_to_gid[dst_core_id][src_type_id][dst_type_id])
                        pid_to_gid[dst_core_id][src_type_id][dst_type_id].append(src)

                    # Track the destination cores connected to the source
                    if not dst_core_id in dst_core_set:
                        dst_core_set.append(dst_core_id)

                        # inverse table (input: src core index => output: memory addr)
                        src_to_addr[src_core_id][dst_core_id] = route_mem_ind[src_core_id]
                        
                        assert(len(route_mem[src_core_id][src_type_id][dst_type_id]) \
                                == route_mem_ind[src_core_id])

                        # Increment the route memory address
                        route_mem[src_core_id][src_type_id][dst_type_id].append(None)
                        route_mem_ind[src_core_id] += 1
                    arr_ind += 1
                # different for the inner and outer loop and for each dtype
                # Append the two memory data
                for dst_core_id in range(core_num):
                    # Set the offset information (between the outer and inner)
                    corr_mem_offset[dst_core_id] = len(mem[dst_core_id]['outer'])
                    # Combine the inner and outer loop data
                    mem[dst_core_id] = mem[dst_core_id]['outer'] + mem[dst_core_id]['inner']
                    corr_mem_ind[dst_core_id] = corr_mem_ind[dst_core_id] + len(mem[dst_core_id])
                    corr_mem[dst_core_id][src_type_id][dst_type_id] += mem[dst_core_id]

                # Iterate over the connected destination cores connected to the source
                for dst_core_id in dst_core_set:
                    # start addr / dst addr
                    route_mem_addr = src_to_addr[src_core_id][dst_core_id]
                    # append the pid 
                    route_mem[src_core_id][src_type_id][dst_type_id][route_mem_addr] = \
                            {'dst core': dst_core_id, 'pid': len(pid_to_gid[dst_core_id][src_type_id][dst_type_id]) - 1}
                    corr_forward[dst_core_id][src_type_id][dst_type_id].append((corr_mem_ind_prev[dst_core_id], corr_mem_offset[dst_core_id], num_iter[dst_core_id]))


            for dst_core_id in range(core_num):
                if not len(corr_forward[dst_core_id][src_type_id][dst_type_id]) == 0: 
                    corr_forward[dst_core_id][src_type_id][dst_type_id].append((corr_mem_ind[dst_core_id], 0, 0))
            for src_core_id in range(core_num):
                route_forward[src_core_id][src_type_id][dst_type_id]\
                             [GV.sim_params['num_unit_per_core'][gtask_id][src_core_id][src_type_id]] = \
                             route_mem_ind[src_core_id]

    GV.pid_to_gid[gtask_id] = pid_to_gid

    initial_corr_forward = [{} for _ in range(GV.sim_params['used_core_num'])]
    initial_corr_mem = [{} for _ in range(GV.sim_params['used_core_num'])]
    initial_route_forward = [{} for _ in range(GV.sim_params['used_core_num'])]
    initial_route_mem = [{} for _ in range(GV.sim_params['used_core_num'])]

    for src_type_id in range(num_types):
        for dst_type_id in range(num_types):
            conn_list = initial_conn_list[src_type_id][dst_type_id]
            if conn_list:
                src_name = GV.sim_params['unit_type_name'][gtask_id][src_type_id]
                dst_name = GV.sim_params['unit_type_name'][gtask_id][dst_type_id]
                conn_name = str(src_name) + '_to_' + str(dst_name)
                #
                initial_corr_forward = [{**initial_corr_forward[core],
                                        **{conn_name: corr_forward[core][src_type_id][dst_type_id]}} \
                                        for core in range(GV.sim_params['used_core_num'])]

                initial_corr_mem = [{**initial_corr_mem[core],
                                    **{conn_name: corr_mem[core][src_type_id][dst_type_id]}} \
                                    for core in range(GV.sim_params['used_core_num'])]

                initial_route_forward = [{**initial_route_forward[core],
                                         **{conn_name: route_forward[core][src_type_id][dst_type_id]}} \
                                         for core in range(GV.sim_params['used_core_num'])]

                initial_route_mem = [{**initial_route_mem[core],
                                     **{conn_name: route_mem[core][src_type_id][dst_type_id]}} \
                                     for core in range(GV.sim_params['used_core_num'])]


    # Add connection for used externals and cores
    # print(cores)
    internal_cores = [core for core in cores if core < GV.sim_params['max_core_x']* GV.sim_params['max_core_y']]
    external_cores = [core for core in cores if core >= GV.sim_params['max_core_x']* GV.sim_params['max_core_y']]

    # broadcast signal to notify the start of a bci task
    if not external_cores:
        # single-core, no external
        external_cores.append(0)
    for ext_core_id in external_cores:
        initial_route_forward[ext_core_id]['init'] = [0, len(internal_cores)]
        initial_route_mem[ext_core_id]['init'] = []
        for int_core_id in internal_cores:
            initial_route_mem[ext_core_id]['init'].append({'dst core': int_core_id, 'pid': -1})

    return initial_corr_forward, initial_corr_mem, \
           initial_route_forward, initial_route_mem

cpdef init_other_mem(state_filename, task_type, gtask_id):

    # Retrive the proper metadata
    core_num = GV.sim_params['used_core_num']
    state_dict = []
    state_dict = np.load(state_filename, allow_pickle=True).item()

    # bit precision for hist mem given from benchmark api
    # default value is 16
    if 'init_prec' in state_dict.keys():
        bit_prec_dict = state_dict['init_prec']
    else:
        print("Using default bit precision for hist mem")
        bit_prec_dict = {}

    initial_state_mem = [{} for _ in range(core_num)]
    initial_hist_mem = [{} for _ in range(core_num)]
    initial_ack_mem = [{} for _ in range(core_num)]
    #initial_stack_mem = [{} for _ in range(core_num)]
    initial_ack_left_mem = [{} for _ in range(core_num)]
    initial_ack_num_mem = [{} for _ in range(core_num)]
    initial_ack_stack_mem = [{} for _ in range(core_num)]
        
    init_ack_left_mem(initial_ack_left_mem, 0, gtask_id)

    # The cascaded sync here (depth 1)
    init_ack_left_mem(initial_ack_left_mem, 1, gtask_id)
    init_ack_num_mem(initial_ack_num_mem, 1, gtask_id)
    init_ack_stack_mem(initial_ack_stack_mem, 1, gtask_id)

    # The cascaded sync here (depth 2)
    init_ack_left_mem(initial_ack_left_mem, 2, gtask_id)
    init_ack_num_mem(initial_ack_num_mem, 2, gtask_id)
    init_ack_stack_mem(initial_ack_stack_mem, 2, gtask_id)

    # The cascaded sync here (depth 3)
    init_ack_left_mem(initial_ack_left_mem, 3, gtask_id)
    init_ack_num_mem(initial_ack_num_mem, 3, gtask_id)
    init_ack_stack_mem(initial_ack_stack_mem, 3, gtask_id)

    # The cascaded sync here (depth 4)
    init_ack_left_mem(initial_ack_left_mem, 4, gtask_id)
    init_ack_num_mem(initial_ack_num_mem, 4, gtask_id)
    init_ack_stack_mem(initial_ack_stack_mem, 4, gtask_id)

    # The cascaded sync here (depth 5)
    init_ack_left_mem(initial_ack_left_mem, 5, gtask_id)
    init_ack_num_mem(initial_ack_num_mem, 5, gtask_id)
    init_ack_stack_mem(initial_ack_stack_mem, 5, gtask_id)

    if task_type == 'snn':

        refr_list = [state_dict['neuron_state'][gid]['refr'] for gid in range(len(state_dict['neuron_state']))]
        I_t_list = [state_dict['neuron_state'][gid]['I_t'] for gid in range(len(state_dict['neuron_state']))]
        v_t_list = [state_dict['neuron_state'][gid]['v_t'] for gid in range(len(state_dict['neuron_state']))]
        decay_v_list = [state_dict['neuron_state'][gid]['decay_v'] for gid in range(len(state_dict['neuron_state']))]
        g_t_list = [state_dict['neuron_state'][gid]['g_t'] for gid in range(len(state_dict['neuron_state']))]
        decay_g_list = [state_dict['neuron_state'][gid]['decay_g'] for gid in range(len(state_dict['neuron_state']))]
        threshold_list = [state_dict['neuron_state'][gid]['threshold'] for gid in range(len(state_dict['neuron_state']))]

        init_state_mem(initial_state_mem, 'neuron states refr', 0, refr_list, False, gtask_id)
        init_state_mem(initial_state_mem, 'neuron states I_t', 0, I_t_list, False, gtask_id)
        init_state_mem(initial_state_mem, 'neuron states v_t', 0, v_t_list, False, gtask_id)
        init_state_mem(initial_state_mem, 'neuron states decay_v', 0, decay_v_list, False, gtask_id)
        init_state_mem(initial_state_mem, 'neuron states g_t', 0, g_t_list, False, gtask_id)
        init_state_mem(initial_state_mem, 'neuron states decay_g', 0, decay_g_list, False, gtask_id)
        init_state_mem(initial_state_mem, 'neuron states threshold', 0, threshold_list, False, gtask_id)

        if GV.sim_params['use_partial'][gtask_id]:
            init_hist_mem(initial_hist_mem, 0, 'weight accum', 0, 0, gtask_id)

        initial_hist_entry = [0 for _ in state_dict['neuron_state']]
        if GV.sim_params['use_partial'][gtask_id]:
            init_hist_mem(initial_hist_mem, 256, 'spike weight accum', 3, 0, gtask_id, True, bit_prec_dict.get('spike weight accum', 16))
            init_hist_mem(initial_hist_mem, 256, 'ext weight accum', 0, initial_hist_entry, gtask_id, False, bit_prec_dict.get('ext weight accum', 16))
        else:
            #init_hist_mem(initial_hist_mem, 256, 'weight accum', 0, 0, gtask_id)
            init_hist_mem(initial_hist_mem, 256, 'weight accum', 0, initial_hist_entry, gtask_id, False, bit_prec_dict.get('weight accum', 16))

    elif task_type == 'ann':
        refr_list = [state_dict['neuron_state'][gid]['refr'] for gid in range(len(state_dict['neuron_state']))]
        I_t_list = [state_dict['neuron_state'][gid]['I_t'] for gid in range(len(state_dict['neuron_state']))]
        # h_t_list = [state_dict['neuron_state'][gid]['h_t'] for gid in range(len(state_dict['neuron_state']))]

        #init_hist_mem(initial_hist_mem, 0, 'neuron states refr', 0, refr_list, gtask_id, False)
        #init_hist_mem(initial_hist_mem, 0, 'neuron states I_t', 0, I_t_list, gtask_id, False)
        init_state_mem(initial_state_mem, 'neuron states refr', 0, refr_list, False, gtask_id)
        init_state_mem(initial_state_mem, 'neuron states I_t', 0, I_t_list, False, gtask_id)
        # init_hist_mem(initial_hist_mem, 0, 'neuron states h_t', 0, h_t_list, gtask_id, False)
        
        initial_hist_entry = [0 for _ in state_dict['neuron_state']]
        if GV.sim_params['use_partial'][gtask_id]:
            init_hist_mem(initial_hist_mem, 0, 'weight accum', 0, 0, gtask_id)
            init_hist_mem(initial_hist_mem, 1, 'ext weight accum', 0, initial_hist_entry, gtask_id, False, bit_prec_dict.get('ext weight accum', 16))
            init_hist_mem(initial_hist_mem, 1, 'spike weight accum', 3, 0, gtask_id, True, bit_prec_dict.get('spike weight accum', 16))
        else:
            init_hist_mem(initial_hist_mem, 1, 'weight accum', 0, initial_hist_entry, gtask_id, False, bit_prec_dict.get('weight accum', 16))

        init_hist_mem(initial_hist_mem, state_dict['bin_width'] - 1, 'bci neuron history', 1, 0, gtask_id, True, bit_prec_dict.get('bci neuron history', 16))
        init_hist_mem(initial_hist_mem, 0, 'bci', 1, 0, gtask_id, True, bit_prec_dict.get('bci', 16))


    elif task_type == 'ss':
        sps_data = np.array(state_dict['sps'])
        #init_hist_mem(initial_hist_mem, 0, 'sps min', 1, sps_data[:,0,0], gtask_id, False)
        #init_hist_mem(initial_hist_mem, 0, 'sps max', 1, sps_data[:,1,0], gtask_id, False)
        #init_hist_mem(initial_hist_mem, 0, 'thresholds', 0, -state_dict['thresholds'], gtask_id, False) # save negative threshold value
        #init_hist_mem(initial_hist_mem, 0, 'partial_distance', 1, 0, gtask_id)
        #init_hist_mem(initial_hist_mem, 0, 'template_occupied', 1, False, gtask_id)

        init_state_mem(initial_state_mem, 'sps min', 1, sps_data[:,0,0], False, gtask_id)
        init_state_mem(initial_state_mem, 'sps max', 1, sps_data[:,1,0], False, gtask_id)
        init_state_mem(initial_state_mem, 'partial_distance', 1, 0, True, gtask_id)
        init_state_mem(initial_state_mem, 'template_occupied', 1, 0, True, gtask_id)
        init_state_mem(initial_state_mem, 'thresholds', 0, -state_dict['thresholds'], False, gtask_id)
        
        #init_stack_mem(initial_stack_mem, 'template', 1, gtask_id)

        init_hist_mem(initial_hist_mem, 200, 'bci', 0, 0, gtask_id, True, bit_prec_dict.get('bci', 16))

    elif task_type == 'tm':
        #init_hist_mem(initial_hist_mem, 0, 'neuron i_n', 0, 0, gtask_id)
        init_state_mem(initial_state_mem, 'neuron i_n', 0, 0, True, gtask_id)
        temp_consts = np.array(state_dict['temp_consts'])
        init_state_mem(initial_state_mem, 'template constants C1', 1, temp_consts[:,0], False, gtask_id)
        init_state_mem(initial_state_mem, 'template constants C2', 1, temp_consts[:,1], False, gtask_id)
        init_state_mem(initial_state_mem, 'template constants C3', 1, temp_consts[:,2], False, gtask_id)
        
        #init_hist_mem(initial_hist_mem, 0, 'template sums S1', 1, 0, gtask_id)
        #init_hist_mem(initial_hist_mem, 0, 'template sums S2', 1, 0, gtask_id)
        #init_hist_mem(initial_hist_mem, 0, 'template sums S3', 1, 0, gtask_id)
        
        init_state_mem(initial_state_mem, 'template sums S1', 1, 0, True, gtask_id)
        init_state_mem(initial_state_mem, 'template sums S2', 1, 0, True, gtask_id)
        init_state_mem(initial_state_mem, 'template sums S3', 1, 0, True, gtask_id)

        if GV.sim_params['use_partial'][gtask_id]:
            #init_hist_mem(initial_hist_mem, 0, 'template psums P2', 3, 0, gtask_id)
            #init_hist_mem(initial_hist_mem, 0, 'template psums P3', 3, 0, gtask_id)
            init_state_mem(initial_state_mem, 'template psums P2', 3, 0, True, gtask_id)
            init_state_mem(initial_state_mem, 'template psums P3', 3, 0, True, gtask_id)


        temp_width = GV.sim_params['consts'][gtask_id]['n_t']
        init_hist_mem(initial_hist_mem, temp_width-1, 'R2', 1, 0, gtask_id, True, bit_prec_dict.get('R2', 16))
        init_hist_mem(initial_hist_mem, temp_width-1, 'R3', 1, 0, gtask_id, True, bit_prec_dict.get('R3', 16))
        if GV.sim_params['use_partial'][gtask_id]:
            init_hist_mem(initial_hist_mem, temp_width-1, 'P1', 3, 0, gtask_id, True, bit_prec_dict.get('P1', 16))
        else:
            init_hist_mem(initial_hist_mem, temp_width-1, 'P1', 1, 0, gtask_id, True, bit_prec_dict.get('P1', 16))

    elif task_type == 'pc':
        window = GV.sim_params['consts'][gtask_id]['window']
        corr_period = GV.sim_params['consts'][gtask_id]['corr_period']
        # corr_period = 100000
        
        init_hist_mem(initial_hist_mem, window, 'history', 0, 0, gtask_id, True, bit_prec_dict.get('history', 16))
        init_state_mem(initial_state_mem, 'neuron spikes', 0, 0, True, gtask_id)
    else:
        assert(0)

    return initial_state_mem, initial_hist_mem, \
           initial_ack_left_mem, initial_ack_num_mem, initial_ack_stack_mem
 
cpdef init_task_mem(core_list):
    task_to_core = []
    for item in eval(core_list):
        task_to_core.append(item)
        # add external core id
        if GV.sim_params['external']:
            for external_core_idx in range(GV.sim_params['max_core_x']):
                target_core = list(range(external_core_idx, GV.sim_params['used_core_num'], GV.sim_params['max_core_x']))
                for core_id in target_core:
                    if core_id in item:
                        task_to_core[-1].append(GV.sim_params['used_core_num'] + external_core_idx)
                        if (external_core_idx, GV.sim_params["max_core_y"]) not in GV.external_id:
                            GV.external_id.append(external_core_idx)
                        break
    if GV.sim_params['external']:
        GV.sim_params['used_core_num'] += GV.sim_params['max_core_x']
    else:
        GV.external_id.append(0)
    GV.sim_params['task_to_core'] = task_to_core

    # Set the list of tasks for a given core
    core_to_task = [[] for _ in range(GV.sim_params['used_core_num'])]
    for core_id in range(GV.sim_params['used_core_num']):
        for gtask_id in range(GV.TOTAL_TASKS):
            if core_id in task_to_core[gtask_id]:
                core_to_task[core_id].append(gtask_id)

    GV.sim_params['core_to_task'] = core_to_task

    # convert the state-to-global / global-to-state task id translation
    GV.ltask_to_gtask = [[None for _ in range(GV.MAX_LOCAL_TASK)] for _ in range(GV.sim_params['used_core_num'])]
    GV.gtask_to_ltask = [[None for _ in range(GV.sim_params['used_core_num'])] for _ in range(GV.TOTAL_TASKS)]

    GV.NUM_SCHEDULED_TASKS = [0 for _ in range(GV.sim_params['used_core_num'])]
    GV.NUM_COMPLETED_TASKS = [[] for _ in range(GV.sim_params['used_core_num'])]
    for core_id in range(GV.sim_params['used_core_num']):
        GV.NUM_SCHEDULED_TASKS[core_id] += len(core_to_task[core_id])
        for ltask_id in range(len(core_to_task[core_id])):
            gtask_id = core_to_task[core_id][ltask_id]
            GV.ltask_to_gtask[core_id][ltask_id] = gtask_id
            GV.gtask_to_ltask[gtask_id][core_id] = ltask_id
    

    # Make a task_to_task translation
    initial_task_translation = [[[None for _ in range(GV.MAX_LOCAL_TASK)] \
                                       for _ in range(GV.sim_params['used_core_num'])] \
                                       for _ in range(GV.sim_params['used_core_num'])]

    for src_core_id in range(GV.sim_params['used_core_num']):
        for dst_core_id in range(GV.sim_params['used_core_num']):
            for ltask_id in range(GV.MAX_LOCAL_TASK):
                gtask_id = GV.ltask_to_gtask[src_core_id][ltask_id]
                if gtask_id != None:
                    dst_ltask_id = GV.gtask_to_ltask[gtask_id][dst_core_id]
                    initial_task_translation[src_core_id][dst_core_id][ltask_id] = dst_ltask_id

    return initial_task_translation

cpdef init_network (network_filename, gtask_id):
    network_dict = np.load(network_filename, allow_pickle=True).item()

    assert((network_dict['simtime'] >= GV.sim_params['workload_timestep'][gtask_id]) and
            'The defined simulation time for a network is shorter than the target simulation time.\nPlease regenerate the benchmark or run the simulation for shorter period')
    GV.sim_params['unit_type_name'][gtask_id] = network_dict['unit_types']
    GV.sim_params['num_unit'][gtask_id] = network_dict['num_unit']
    GV.sim_params['num_unit_types'][gtask_id] = len(GV.sim_params['num_unit'][gtask_id])
    GV.sim_params['num_total_unit'][gtask_id] = network_dict['num_internal'] + \
                                               network_dict['num_external']

    GV.sim_params['gid_to_tid'] = []
    for type_id in range(len(GV.sim_params['num_unit'][gtask_id])):
        num_unit = GV.sim_params['num_unit'][gtask_id][type_id]
        GV.sim_params['gid_to_tid'] += [(i, type_id) for i in range(num_unit)]

    GV.sim_params['consts'][gtask_id] = network_dict['consts'] if 'consts' in network_dict else None
