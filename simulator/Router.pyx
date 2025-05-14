import random
import GlobalVars as GV
from collections import deque
from heapq import heappop, heappush
from Event import Event

import Profiler
import sys
import Task
import copy
import numpy as np
import time

from Energy import op_count

def coord_to_ind(x, y):
    return int(y * GV.sim_params['max_core_x'] + x)

def ind_to_coord(ind):
    return int(ind % GV.sim_params['max_core_x']), int(ind / GV.sim_params['max_core_x'])
    
cdef class TaskScheduler:
    def __init__(self, ind, router):
        # latency budget of each task
        self.ind = ind
        self.router = router
        self.latency_budget = [0 for _ in range(GV.NUM_SCHEDULED_TASKS[self.ind])]
        self.sorted_task = [i for i in range(GV.NUM_SCHEDULED_TASKS[self.ind])]

    # after bci_send
    cpdef task_init(self, ltask_id, budget):
        self.latency_budget[ltask_id] = budget
        self.sorted_task = np.argsort(self.latency_budget).tolist()

    cpdef decrement_budget(self):
        # Decrement the budget by 1 (every 1 ms)
        if GV.cyc % GV.sim_params['1ms_cyc'] == 0:
            for task_id in range(GV.NUM_SCHEDULED_TASKS[self.ind]):
                self.latency_budget[task_id] = max(self.latency_budget[task_id]-1, 0)

    cpdef schedule(self, valid):
        # We should schedule bci_send_type (when not pending)
        # iterate over the valid list

        # MODE 0: get random
        if GV.SCHEDULER == 0: 
            target = [i for i, x in enumerate(valid) if x == True]
            ltask_id = random.choice(target)
        # MODE 1: Schedule task with minimum latency budget left
        elif GV.SCHEDULER == 1:
            min_budget = sys.maxsize
            ltask_id = GV.NUM_SCHEDULED_TASKS[self.ind]
            # Get the entry with the minimum value
            for tid in range(GV.NUM_SCHEDULED_TASKS[self.ind]):
                if valid[tid]:
                    if min_budget > self.latency_budget[tid]:
                        min_budget = self.latency_budget[tid]
                        ltask_id = tid
        # MODE 2: Schedule task with the first-came event (FCFS)
        elif GV.SCHEDULER == 2:
            first_event_cyc = sys.maxsize
            ltask_id = GV.NUM_SCHEDULED_TASKS[self.ind]
            buf = self.router.in_event_buf.get_buf()
            for tid in range(GV.NUM_SCHEDULED_TASKS[self.ind]):
                if len(buf[tid]):
                    assert(valid[tid])
                    _, _, cyc = buf[tid][0]
                    if cyc < first_event_cyc:
                        first_event_cyc = cyc
                        ltask_id = tid
        # MODE 3: Schedule task with most events in in_event_buf
        elif GV.SCHEDULER == 3:
            max_event_num = 0
            ltask_id = GV.NUM_SCHEDULED_TASKS[self.ind]
            buf = self.router.in_event_buf.get_buf()
            for tid in range(GV.NUM_SCHEDULED_TASKS[self.ind]):
                if len(buf[tid]) > max_event_num:
                    assert(valid[tid])
                    max_event_num = len(buf[tid])
                    ltask_id = tid
        # MODE 4: Statically schedule task with lowest latency budget
        elif GV.SCHEDULER == 4:
            max_event_num = 0
            ltask_id = GV.NUM_SCHEDULED_TASKS[self.ind]
            for tid in self.sorted_task:
                if valid[tid]:
                    ltask_id = tid
                    break
        # MODE 5: HIGHEST PRIORITY
        elif GV.SCHEDULER == 5:
            tid = GV.gtask_to_ltask[GV.sim_params['highest_priority']][self.ind]
            if tid and valid[tid]:
                ltask_id = tid
            else:
                target = [i for i, x in enumerate(valid) if x == True]
                ltask_id = random.choice(target)
        else:
            assert(0)
        return ltask_id

cdef class PriorityBuffer:
    def __init__(self, ind, task_scheduler):
        self.ind = ind
        self.buf = [deque() for ltask_id in range(GV.NUM_SCHEDULED_TASKS[self.ind])]
        # check if a valid entry exist
        self.valid = [False for _ in range(GV.NUM_SCHEDULED_TASKS[self.ind])]
        self.task_scheduler = task_scheduler

    cpdef empty(self):
        for ltask_id in range(GV.NUM_SCHEDULED_TASKS[self.ind]):
            if self.valid[ltask_id]: return False
        return True

    cpdef append(self, ltask_id, event_type, event):
        op_count["in_event_buf_push"] += 1
        self.buf[ltask_id].append((event_type, event, GV.cyc))
        self.valid[ltask_id] = True

        # The bci init operation sets the latency budget
        if event_type == 'bci_init': self.task_scheduler.task_init(ltask_id, event['data'])

    # return event with the highest priority
    cpdef popleft(self):
        op_count["in_event_buf_pop"] += 1
        ltask_id = self.task_scheduler.schedule(self.valid)
        # valid becomes false if an event is popped
        num_events = len(self.buf[ltask_id])
        assert(num_events > 0)
        if num_events == 1: self.valid[ltask_id] = False
        return self.buf[ltask_id].popleft(), ltask_id

    cpdef get_buf(self):
        return self.buf

cdef class Synchronizer:
    def __init__(self, int ind, Router router, list initial_sync_info):
        self.ind = ind
        self.router = router
        self.core = router.core
        self.mem = router.mem

        # Keeps the information related to the synchronization
        self.sync_table = {}
        for ltask_id in range(GV.NUM_SCHEDULED_TASKS[self.ind]):
            gtask_id = GV.ltask_to_gtask[self.ind][ltask_id]
            self.sync_table[ltask_id] = {}

            # 1) The role of the synchronization
            self.sync_table[ltask_id]['sync_role'] = initial_sync_info[gtask_id]['sync_role']

            # 2) The total number of children & the ready count
            self.sync_table[ltask_id]['num_children'] = initial_sync_info[gtask_id]['num_children']
            self.sync_table[ltask_id]['child_ready_count'] = 0

            # 3) Sync_event_done checks if the computation for the given synchronization is done
            self.sync_table[ltask_id]['sync_event_done'] = None

            # 4) Current commit phase
            self.sync_table[ltask_id]['commit_phase'] = 0
            # 5) The total number of phases
            self.sync_table[ltask_id]['num_phase'] = Task.get_task(gtask_id).num_phase

            # 6) Per packet => whether the packet triggers generating an entry
            self.sync_table[ltask_id]['event_type'] = Task.get_task(gtask_id).event_type
            
        # Keeps the target task to synchronize
        self.sync_target = -1

    # If the event is done
    # check if the event is the target of synchronization
    cpdef event_done(self, event_type, pck_type, event_data, ltask_id):
        cyc = 1
        # this is packet encoder/decoder
        mode = self.sync_table[ltask_id]['event_type'][event_type][pck_type] # pck_type is start or end
        depth = self.sync_table[ltask_id]['event_type'][event_type]['depth']
        prev = self.sync_table[ltask_id]['event_type'][event_type]['prev']
        op_count['event_table'] += 1
        
        # Set the sync done
        if mode == 'sync':
            cyc += 1
            self.sync_table[ltask_id]['child_ready_count'] += 1
            self.sync_target = ltask_id
            op_count['sync_table_update'] += 1
        elif mode == 'done':
            if self.sync_table[ltask_id]['sync_event_done'] == None:
                self.sync_table[ltask_id]['sync_event_done'] = True
                self.sync_target = ltask_id
                op_count['sync_table_update'] += 1
            else:
                self.sync_table[ltask_id]['sync_event_done'] = True
                self.sync_target = ltask_id
                op_count['sync_table_update'] += 1
        elif mode == 'commit':
            if self.sync_table[ltask_id]['sync_event_done'] != None:

                self.sync_table[ltask_id]['commit_phase'] += 1
                self.sync_table[ltask_id]['commit_phase'] %= self.sync_table[ltask_id]['num_phase']
                op_count['sync_table_update'] += 1
                if self.sync_table[ltask_id]['sync_role'] == 'non_root':
                    data = {'mode' : 'synchronize', 'type' : 'commit', 'task_id' : ltask_id, 'data': None, 'lid': 0, 'event_type' : None}
                    self.router.out_event_buf.insert(0, Event({'cyc': GV.cyc+cyc, 'data': data}))

        elif mode == 'gen_ack':
            # increment ack left (which will be decremented after event done)
            if not self.core.has_external:
                return 0
            cyc += 1
            ack_left = self.mem.read_ack_left(depth + 1, ltask_id, 0)
            if ack_left == 0:
                data = {'mode' : 'decrement',
                        'type' : prev,
                        'depth' : depth + 1,
                        'task_id' : ltask_id,
                        'data': None}
                self.router.out_event_buf.insert(0, Event({'cyc': GV.cyc+cyc, 'data': data}))
            else:
                cyc += 1
                # read the ack num
                src_x = event_data['src_x']
                src_y = event_data['src_y']
                core_id = coord_to_ind(src_x, src_y)
                ack_num = self.mem.read_ack_num(depth + 1, ltask_id, core_id)
                # if zero => it is the first and should be pushed to the stack
                # then, increment the ack num
                if ack_num == 0:
                    cyc += 1
                    self.mem.push_ack_stack_mem(depth + 1, ltask_id, core_id)
                cyc += 1
                self.mem.write_ack_num(depth + 1, ltask_id, core_id, ack_num + 1)
        elif mode == 'dec_ack':
            if not self.core.has_external:
                return 0
            send_valid, ack_left, decrement_cyc = self.decrement_ack(depth, event_data['data'], ltask_id)
            cyc += decrement_cyc
            if send_valid:
                data = {'mode' : 'decrement',
                        'type' : prev,
                        'depth' : depth,
                        'task_id' : ltask_id,
                        'data': None}
                self.router.out_event_buf.insert(0, Event({'cyc': GV.cyc+cyc, 'data': data}))
        elif mode == 'send_ack':
            if not self.core.has_external:
                return 0
            data = {'mode' : 'ack',
                    'type' : prev,
                    'task_id' : ltask_id,
                    'dst_x' : event_data['src_x'], 'dst_y' : event_data['src_y'],
                    'data' : [None]}
            self.router.out_event_buf.insert(0, Event({'cyc': GV.cyc+cyc, 'data': data}))
        elif mode == None:
            pass
        else:
            print(mode)
            assert(0)

        return cyc

    # Increment the ack entry
    cpdef increment_ack(self, event_type, ltask_id):
        depth = self.sync_table[ltask_id]['event_type'][event_type]['depth']
        op_count['event_table'] += 1
        ack_left = self.mem.read_ack_left(depth + 1, ltask_id, 0)
        self.mem.write_ack_left(depth + 1, ltask_id, 0, ack_left + 1)


    cpdef decrement_ack(self, depth, ack_num, ltask_id):
        decrement_cyc = 0

        # If the ack is the root => we should decrement the ack
        if depth >= 0:
            decrement_cyc += 1
            ack_left = self.mem.read_ack_left(depth, ltask_id, 0)
            decrement_cyc += 1
            ack_left -= ack_num
            if ack_left < 0:
                print(depth)
                assert(0)
            self.mem.write_ack_left(depth, ltask_id, 0, ack_left)
            if ack_left == 0: self.sync_target = ltask_id

        # Send acknowledgement packet only upon a predefined condition
        decrement_cyc += 1
        if depth > 0 and ack_left == 0:
            return True, ack_left, decrement_cyc
        else:
            return False, ack_left, decrement_cyc
        

    # Defined for custom events that should be synchronized
    cpdef synchronize(self):
        sync_cyc = 0
        ltask_id = self.sync_target
        self.sync_target = -1

        sync_cyc += 1
        op_count['sync_table_read'] += 1
        if self.sync_table[ltask_id]['child_ready_count'] == \
           self.sync_table[ltask_id]['num_children'] and \
           self.sync_table[ltask_id]['sync_event_done']:

            if self.core.has_external:
                sync_cyc += 1
                ack_left = self.mem.read_ack_left(0, ltask_id, 0)
            else:
                ack_left = 0

            if ack_left == 0:
                op_count['sync_table_update'] += 1
                self.sync_table[ltask_id]['child_ready_count'] = 0
                self.sync_table[ltask_id]['sync_event_done'] = False

                # The router sends the commit packet to the childrent + self if root
                sync_cyc += 1
                if self.core.has_external:
                    packet_type = 'commit' if self.sync_table[ltask_id]['sync_role'] == 'root' else 'sync'
                    data = {'mode' : 'synchronize', 'type' : packet_type, 'task_id' : ltask_id, 'data': [None], 'lid': 0, 'event_type' : None}
                    self.router.out_event_buf.insert(0, Event({'cyc': GV.cyc+sync_cyc, 'data': data}))
                else:
                    assert(self.ind == 0)
                    # Triggers an event after the commit
                    packet = {'mode' : 'synchronize', 'pck_type' : 'commit', 'task_id': ltask_id, 'src_pid' : None, 'src_x': self.router.x, 'src_y': self.router.y, 'data' : [None], 'event_type' : None}
                    self.router.recv_buf.append(packet)
                    op_count['recv_buf_push'] += 1

                # There are for debugging and simulation purpose
                # Check if the task is completed or not
                gtask_id = GV.ltask_to_gtask[self.ind][ltask_id]
                if GV.leading_timestep[gtask_id] > GV.sim_params['workload_timestep'][gtask_id]:
                    if gtask_id not in GV.NUM_COMPLETED_TASKS[self.ind]:
                        GV.NUM_COMPLETED_TASKS[self.ind].append(gtask_id)
                # IF root, profile the end of a timestep (@root)
                if self.sync_table[ltask_id]['sync_role'] == 'root':
                    if self.sync_table[ltask_id]['commit_phase'] == \
                       self.sync_table[ltask_id]['num_phase'] - 1:
                        GV.ts_profile['done'] = True
                        GV.ts_profile['gtask_id'] = gtask_id

        return sync_cyc

cdef class Router:

    def __init__(self, int ind, Core.Core core, list initial_sync_info, list initial_task_translation, int has_external):
        # Set position

        # The receive buffer is double buffered
        # to enable accurate multicore processing
        self.recv_buf = deque()
        self.send_buf = []
        self.mem = core.mem
        self.core = core
        self.ind = ind
        if has_external:
            self.x, self.y = ind_to_coord(self.ind)
            self.sw = GV.NoC.sw[self.x][self.y]
            self.sw.router = self
        else:
            self.x, self.y = ind_to_coord(self.ind)
        
        self.synchronizer = Synchronizer(ind, self, initial_sync_info)
            
        # Event Store
        self.task_scheduler = TaskScheduler(self.ind, self)
        self.in_event_buf = PriorityBuffer(self.ind, self.task_scheduler)
        self.out_event_buf = []

        # Event to Packet Generation
        self.generator_available_cyc = 0

        # Should determine the packet to event
        self.packet_to_event = [None for ltask_id in range(GV.NUM_SCHEDULED_TASKS[self.ind])]
        for ltask_id in range(GV.NUM_SCHEDULED_TASKS[self.ind]):
            gtask_id = GV.ltask_to_gtask[self.ind][ltask_id]
            self.packet_to_event[ltask_id] = Task.get_task(gtask_id).packet_to_event

        self.packet_to_route = [None for ltask_id in range(GV.NUM_SCHEDULED_TASKS[self.ind])]
        for ltask_id in range(GV.NUM_SCHEDULED_TASKS[self.ind]):
            gtask_id = GV.ltask_to_gtask[self.ind][ltask_id]
            self.packet_to_route[ltask_id] = Task.get_task(gtask_id).packet_to_type

        self.task_translation = initial_task_translation

    # Generate Packet for a Given Out Event
    cpdef packet_generation(self):
        cdef int src_lid
        cdef int src_pid

        cdef int addr
        
        cdef int dst_core

        cdef int dst_x
        cdef int dst_y
        cdef dict event
        
        # Check if the spike generator is available
        if not GV.cyc >= self.generator_available_cyc:
            GV.valid_advance = True
            return

        # If so, we first try synchronization for the target tasks
        if self.synchronizer.sync_target >= 0:
            if self.core.has_external:
                GV.valid_advance = True
                self.generator_available_cyc = GV.cyc + self.synchronizer.synchronize()
                return
            else:
                # Sync only if all event buffers are empty
                if not self.out_event_buf and not self.recv_buf and self.in_event_buf.empty():
                    GV.valid_advance = True
                    self.generator_available_cyc = GV.cyc + self.synchronizer.synchronize()
                    return

        # Check if an out event exists
        if not self.out_event_buf: return

        # Check if an out event is yet to be processed
        if self.out_event_buf[0]['cyc'] > GV.cyc:
            GV.valid_advance = True
            return

        GV.valid_advance = True
        out_event = self.out_event_buf.pop(0)

        # 
        data        = out_event['data']
        mode        = data['mode']
        ltask_id    = data['task_id']
        pck_type    = data['type'] if 'type' in data else None
        event_type  = data['event_type'] if 'event_type' in data else None
        self.generator_available_cyc = GV.cyc

        # Process Event from the out-event queue
        # Check the list of events that are finished 
        if mode == 'event':
            event_data  = data['event_data']
            cyc = self.synchronizer.event_done(event_type, pck_type, event_data, ltask_id)
            self.generator_available_cyc += cyc
            return

        gtask_id = GV.ltask_to_gtask[self.ind][ltask_id]
        # This will be a general packet
        if not type(data['data']) is list: data['data'] = [data['data']]

        packet = {'mode' : mode,
                  'pck_type' : pck_type,
                  'src_pid' : None,
                  'dst_x' : None, 'dst_y' : None,
                  'src_x': self.x, 'src_y': self.y,
                  'data' : data['data']}

        if mode == 'probe':
            assert(self.core.has_external)
            self.generator_available_cyc += 1
           
            # get external's core x,y
            dst_x = self.ind % GV.sim_params['max_core_x']
            dst_y = GV.sim_params['max_core_y']
            packet['dst_x'] = dst_x
            packet['dst_y'] = dst_y
            packet['task_id'] = gtask_id 

            self.send_buf.append({'cyc': self.generator_available_cyc, 'packet': packet})
            op_count['send_buf_push'] += 1

            return

        if mode == 'decrement':
            assert(self.core.has_external)
            depth = data['depth']
            while True:
                self.generator_available_cyc += 1
                # Pop the acknowledgement stack
                empty, dst_core_id = self.mem.pop_ack_stack_mem(depth, ltask_id)
                if empty: break

                self.generator_available_cyc += 1
                dst_x, dst_y = ind_to_coord(dst_core_id)

                # Read ack num & Reset ack num
                ack_num = self.mem.read_ack_num(depth, ltask_id, dst_core_id)
                self.mem.write_ack_num(depth, ltask_id, dst_core_id, 0)

                # Send the ack packet
                packet = copy.deepcopy(packet)
                # For ACK packets
                packet['mode'] = 'ack'
                packet['dst_x'] = dst_x
                packet['dst_y'] = dst_y
                packet['data'] = [ack_num]
                packet['task_id'] = self.task_translation[dst_core_id][ltask_id]
                self.generator_available_cyc += 1
                self.send_buf.append({'cyc': self.generator_available_cyc, 'packet': packet})
                op_count['send_buf_push'] += 1
            return

        if mode == 'ack':
            assert(self.core.has_external)
            if not pck_type in Task.get_task(gtask_id).packet:
                print(pck_type)
            assert(pck_type in Task.get_task(gtask_id).packet)
            packet = copy.deepcopy(packet)
            # For ACK packets
            packet['dst_x'] = data['dst_x']
            packet['dst_y'] = data['dst_y']
            dst_core_id = coord_to_ind(data['dst_x'], data['dst_y'])
            packet['task_id'] = self.task_translation[dst_core_id][ltask_id]
            packet['data'] = [1]
            self.generator_available_cyc += 1 # mark that packet generation is available at this cycle
            self.send_buf.append({'cyc': self.generator_available_cyc, 'packet': packet})
            op_count['send_buf_push'] += 1
            return

        if mode == 'synchronize':
            assert(self.core.has_external)
            self.generator_available_cyc += 1
            route_name = pck_type
            routing_addr = data['lid']
            start_addr, end_addr = self.mem.read_route_forward(route_name, ltask_id, routing_addr)

                
            for addr in range(start_addr, end_addr):
                # address of the destination core
                route_dat = self.mem.read_route(route_name, ltask_id, addr)
                dst_core_id = route_dat['dst core']
                dst_x, dst_y = ind_to_coord(dst_core_id)

                packet = copy.deepcopy(packet)
                packet['dst_x'] = dst_x
                packet['dst_y'] = dst_y
                packet['src_pid'] = data['lid']
                packet['task_id'] = self.task_translation[dst_core_id][ltask_id]

                self.generator_available_cyc += 1
                self.send_buf.append({'cyc': self.generator_available_cyc, 'packet': packet})
                op_count['send_buf_push'] += 1
            return

        if mode == 'packet':
            if not pck_type in Task.get_task(gtask_id).packet:
                print(pck_type)
            assert(pck_type in Task.get_task(gtask_id).packet)

            self.generator_available_cyc += 1
            route_name = self.get_packet_to_route(pck_type, ltask_id)
            # Through correlation memory
            if route_name != None:
                routing_addr = data['lid']
                self.generator_available_cyc += 1
                start_addr, end_addr = self.mem.read_route_forward(route_name, ltask_id, routing_addr)
                for addr in range(start_addr, end_addr):
                    route_dat = self.mem.read_route(route_name, ltask_id, addr)
                    dst_core_id = route_dat['dst core']
                    src_pid = route_dat['pid']
                    dst_x, dst_y = ind_to_coord(dst_core_id)

                    #########################################################
                    packet = copy.deepcopy(packet)

                    packet['dst_x'] = dst_x
                    packet['dst_y'] = dst_y
                    packet['src_pid'] = src_pid
                    packet['task_id'] = self.task_translation[dst_core_id][ltask_id]

                    if self.core.has_external:
                        self.generator_available_cyc += 2
                        self.synchronizer.increment_ack(event_type, ltask_id)
                        self.generator_available_cyc += 1
                        self.send_buf.append({'cyc': self.generator_available_cyc, 'packet': packet})
                        op_count['send_buf_push'] += 1
                    else:
                        self.recv_buf.append(packet)
                        op_count['recv_buf_push'] += 1
            # Through ack-like method
            else:
                dst_x = data['lid']['dst_x']
                dst_y = data['lid']['dst_y']
                lid = data['lid']['lid']
                packet['dst_x'] = dst_x
                packet['dst_y'] = dst_y
                packet['src_pid'] = lid
                dst_core_id = coord_to_ind(data['dst_x'], data['dst_y'])
                packet['task_id'] = self.task_translation[dst_core_id][ltask_id]
                if self.core.has_external:
                    self.generator_available_cyc += 2
                    self.synchronizer.increment_ack(event_type, ltask_id)
                    self.generator_available_cyc += 1
                    self.send_buf.append({'cyc': self.generator_available_cyc, 'packet': packet})
                    op_count['send_buf_push'] += 1
                else:
                    self.recv_buf.append(packet)
                    op_count['recv_buf_push'] += 1
            return

        assert(0 and data)

    # Define what to do after the packet reception
    # We should convert the packet to the event
    cpdef packet_reception(self):
        if self.recv_buf:
            # print('packet reception')
            GV.valid_advance = True
            recv_packet = self.recv_buf.popleft()
            # 
            mode = recv_packet['mode']

            pck_type = recv_packet['pck_type']
            ltask_id = recv_packet['task_id']

            src_pid = recv_packet['src_pid']
            src_x = recv_packet['src_x']
            src_y = recv_packet['src_y']
            data = recv_packet['data']

            # Determine whether to receive the data as a scalar or vector
            if len(data) == 1: data = data[0]

            # We send the event_data in a generalized format
            event_data = {'src_pid' : src_pid, 'data' : data, 'src_x' : src_x, 'src_y' : src_y}
            
            # Access the memory to retrieve the next event
            if (self.core.external or not self.core.has_external) and mode == 'probe': # external can receive probe packet
                event_type = None # probe packet doesn't require event 
            else:
                event_type = self.get_packet_to_event(mode, pck_type, ltask_id)

            if event_type != None:
                # if external & bci_send => send to the pending queue
                if 'bci_send' in event_type and (self.core.external or not self.core.has_external):
                    # increment the workload cycle and send the bci event to the valid queue
                    gtask_id = GV.ltask_to_gtask[self.ind][ltask_id]
                    timestep = GV.per_core_timestep[gtask_id][self.ind]
                    GV.workload_cyc[gtask_id] = (timestep + 1) * int(GV.sim_params['dt'][gtask_id] / GV.sim_params['cyc_period'])
                    GV.target_cyc[gtask_id] = GV.workload_cyc[gtask_id] + int(GV.sim_params['latency'][gtask_id] / GV.sim_params['cyc_period'])
                    self.core.external_module.pending_queue.append((ltask_id, gtask_id))
                    self.core.external_module.pending_cyc.append(GV.workload_cyc[gtask_id])
                else:
                    if event_type == None: assert(0)
                    self.in_event_buf.append(ltask_id, event_type, event_data)

    cpdef get_packet_to_route(self, pck_type, ltask_id):
        if pck_type == 'init':
            return 'init'
        else:
            src_type, dst_type = self.packet_to_route[ltask_id][pck_type]
            op_count['pck_to_route_table'] += 1
            if src_type == None: return None
            gtask_id = GV.ltask_to_gtask[self.ind][ltask_id]
            route_name = "{}_to_{}".format(src_type, dst_type)
            return route_name

    cpdef get_packet_to_event(self, mode, pck_type, ltask_id):
        # this is packet encoder/decoder
        event_type = self.packet_to_event[ltask_id][pck_type][mode]
        op_count['pck_to_event_table'] += 1
        # this is phase controller - pre/post proc. list
        if type(event_type) is list:
            phase = self.synchronizer.sync_table[ltask_id]['commit_phase']
            op_count['sync_table_read'] += 1
            event_type = event_type[phase]
        return event_type
