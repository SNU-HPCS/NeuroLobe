cimport cython
import numpy as np
import os
cimport numpy as np
import Task
import Profiler
from heapq import heappop, heappush
from Event import Event

from ISA import ISAController
from EventProcessor import PC

import time
import GlobalVars as GV
from collections import deque
import copy
import sys
import EventProcessor
from Inst_Parser import parse_inst_file
import heapq

from Energy import op_count

cdef class Register:
    def __init__(self):
        # Event Data Region
        # self.event_size = 10
        self.event_size = GV.reg_event_size
        # Free Region
        # self.free_region = 30
        self.free_region = GV.reg_free_region
        # Timestep Region
        self.timestep_region = GV.MAX_LOCAL_TASK

        self.general_region_size = self.event_size + self.free_region
        # GV.reg_general_region_size = self.general_region_size

        self.size = self.event_size + \
                    self.free_region + \
                    self.timestep_region
        self.reg = [None for _ in range(self.size)]

    cpdef read(self, addr):
        # used when register file is accessed from ISA
        translated_addr = addr
        assert(translated_addr < self.size)
        return self.reg[translated_addr]

    cpdef write(self, addr, dat):
        # used when register file is accessed from ISA
        translated_addr = addr
        assert(translated_addr < self.size)
        self.reg[translated_addr] = dat

    # cpdef print_dat(self, addr, task_id):
    #     translated_addr = self.translation(addr, task_id)


# Supports three-level loop controller for correlation memory
# in the worst case =>
# 1) unit iteration
# 2~3) corr outer / inner
cdef class LoopCtrl:
    def __init__(self, reg, mem):
        # current depth of the loop
        self.reg = reg
        self.depth = -1
        MAX_DEPTH = 3

        # num_iter: the total number of iterations to be executed
        # counter: the current number of iterates
        # timestep_addr: the address of the register file to store the counter
        # (sumed up with timestep / forward & reverse order)
        self.num_iter = [0 for _ in range(MAX_DEPTH)]
        self.counter = [0 for _ in range(MAX_DEPTH)]
        self.timestep_addr = [0 for _ in range(MAX_DEPTH)]
        self.forward_order = [0 for _ in range(MAX_DEPTH)]

        # inter_addr_offset: the memory address to access
        # intra_addr_offset: the offset within the address
        self.inter_addr_offset = [0 for _ in range(MAX_DEPTH)]
        self.intra_addr_offset = [0 for _ in range(MAX_DEPTH)]

        # The instruction return address
        # (if the condition is not met (start_addr) / met (end_addr))
        self.inst_loop_addr = [0 for _ in range(MAX_DEPTH)]
        self.inst_end_addr = [0 for _ in range(MAX_DEPTH)]

        # the number of offset increment for each iteration
        self.loop_offset = [0 for _ in range(MAX_DEPTH)]

    cpdef insert(self, func_type, outer_offset, inner_offset,
                 num_iter, timestep_addr, inst_loop_addr, inst_end_addr,
                 loop_offset, forward_order):

        op_count["activate_loop_ctrl"] += 1
        self.depth += 1
        # Set the number of iterations and counter address
        self.num_iter[self.depth] = num_iter
        self.timestep_addr[self.depth] = timestep_addr
        self.counter[self.depth] = 0
        #
        self.forward_order[self.depth] = forward_order

        # Arbitrary bit precision
        # the iteration memory offset is set to zero
        # We keep the previous loop offset for inner
        if func_type == 'outer':
            self.inter_addr_offset[self.depth] = outer_offset
            self.intra_addr_offset[self.depth] = 0
            # Preset the offset address in advance
            self.inter_addr_offset[self.depth + 1] = outer_offset + inner_offset
            self.intra_addr_offset[self.depth + 1] = 0
        # For inner, we use the predefined addresses

        # Set the instruction addresses
        self.inst_loop_addr[self.depth] = inst_loop_addr
        self.inst_end_addr[self.depth] = inst_end_addr

        self.loop_offset[self.depth] = loop_offset

        # if the condition met
        if num_iter == 0:
            addr = self.inst_end_addr[self.depth]
            self.depth -= 1
            return addr
        # if the condition unmet
        else:
            return self.inst_loop_addr[self.depth]

    # at the end
    cpdef iterate(self):
        op_count["activate_loop_ctrl"] += 1
        num_iter = self.num_iter[self.depth]
        timestep_addr = self.timestep_addr[self.depth]
        loop_offset = self.loop_offset[self.depth]

        forward_order = self.forward_order[self.depth]

        inter_addr_offset = self.inter_addr_offset[self.depth]
        intra_addr_offset = self.intra_addr_offset[self.depth]

        # increment the counter data by 1
        counter = self.counter[self.depth]
        counter += 1
        self.counter[self.depth] = counter

        # increment or decrement the timestep data
        timestep = self.reg.read(timestep_addr)
        if forward_order: timestep += 1
        else:             timestep -= 1
        self.reg.write(timestep_addr, timestep)
        op_count["reg_op"] += 1

        # increment the address offset
        incremented_offset = intra_addr_offset + loop_offset
        self.inter_addr_offset[self.depth] += incremented_offset // GV.MEM_WIDTH['corr_mem']
        self.intra_addr_offset[self.depth] = incremented_offset % GV.MEM_WIDTH['corr_mem']

        # if the condition met
        if counter == num_iter:
            addr = self.inst_end_addr[self.depth]
            self.depth -= 1
            return addr
        # if the condition unmet
        else:
            return self.inst_loop_addr[self.depth]

    cpdef get_addr(self, offset, width):
        # Check if the data is within the given line
        incremented_offset_total = offset + self.intra_addr_offset[self.depth]
        incremented_addr = incremented_offset_total // GV.MEM_WIDTH['corr_mem']
        incremented_offset = incremented_offset_total % GV.MEM_WIDTH['corr_mem']
        return self.inter_addr_offset[self.depth] + incremented_addr, incremented_offset

# A custom datastructure that stores the events that are processed
cdef class ProcessingEvent:
    def __init__(self, core):
        self.buf = {}
        self.core = core
        self.ind = core.ind
        self.busy_type = None
        for ltask_id in range(GV.NUM_SCHEDULED_TASKS[self.ind]):
            gtask_id = GV.ltask_to_gtask[self.ind][ltask_id]
            for event_type in Task.get_task(gtask_id).event:
                self.buf[event_type] = {'data' : None, 'cyc' : {}, 'processing' : False, 'task_id' : -1}

    cpdef busy(self):
        return self.busy_type

    cpdef schedule_event(self, event_type, event_data, int ltask_id):
        self.busy_type = event_type
        self.buf[event_type]['processing'] = True
        self.buf[event_type]['data'] = event_data
        self.buf[event_type]['cyc'] = None
        self.buf[event_type]['task_id'] = ltask_id

        # Write event data to the register file
        self.core.reg.write(0, event_data['src_pid'])
        self.core.reg.write(1, event_data['data'])
        self.core.reg.write(2, event_data['src_x'])
        self.core.reg.write(3, event_data['src_y'])

    cpdef event_done(self, event_type):
        self.busy_type = None
        self.buf[event_type]['processing'] = False
        self.buf[event_type]['task_id'] = -1

    cpdef get_data(self, event_type):
        data = self.buf[event_type]['data']
        cyc = self.buf[event_type]['cyc']
        ltask_id = self.buf[event_type]['task_id']
        return data, cyc, ltask_id

    cpdef set_end_state(self, event_type, cyc):
        self.buf[event_type]['cyc'] = cyc

    cpdef print_buf(self):
        for ltask_id in range(GV.NUM_SCHEDULED_TASKS[self.ind]):
            gtask_id = GV.ltask_to_gtask[self.ind][ltask_id]
            for event_type in Task.tasks[gtask_id]['event']:
                print(event_type, self.buf[event_type]['processing'])
                assert(0)

cdef class Core:
    def __init__(self, int ind, 
                 list state_mem,
                 list state_mem_offset,
                 list hist_mem_num_entries,
                 list hist_metadata,
                 list hist_pos,
                 list hist_mem_offset,
                 list hist_mem,
                 #list stack_mem_num_entries,
                 #list stack_mem_offset,
                 #list stack_mem,
                 #list stack_ptr,
                 list corr_mem_num_entries,
                 list corr_mem_offset,
                 list corr_mem,
                 list corr_forward_num_entries,
                 list corr_forward_offset,
                 list corr_forward,
                 list route_forward_num_entries,
                 list route_forward_offset,
                 list route_forward,
                 list route_mem_num_entries,
                 list route_mem_offset,
                 list route_mem,
                 list ack_left_mem_num_entries,
                 list ack_left_mem_offset,
                 list ack_left_mem,
                 list ack_num_mem_num_entries,
                 list ack_num_mem_offset,
                 list ack_num_mem,
                 list ack_stack_mem_num_entries,
                 list ack_stack_mem_offset,
                 list ack_stack_mem,
                 list ack_stack_ptr,
                 list initial_sync_info,
                 list initial_task_translation,
                 list external_input,
                 list table_key_list,
                 list base_list,
                 list bound_list,
                 list inst_list,
                 int has_external,
                 int external):
        
        self.ind = ind 
        self.has_external = has_external
        self.external = external
        assert(has_external or not external) # non-external mode should not have external core

        if self.external or not self.has_external:
            external_id = self.ind % GV.sim_params['max_core_x']
            self.external_module = ExternalModule(self, external_input, external_id)
        else:
            self.external_module = None

        # now memory layout is predetermined by the compiler
        self.mem = Memory(self,
                    state_mem,
                    state_mem_offset,
                    hist_mem_num_entries,
                    hist_metadata,
                    hist_pos,
                    hist_mem_offset,
                    hist_mem,
                    #stack_mem_num_entries,
                    #stack_mem_offset,
                    #stack_mem,
                    #stack_ptr,
                    corr_mem_num_entries,
                    corr_mem_offset,
                    corr_mem,
                    corr_forward_num_entries,
                    corr_forward_offset,
                    corr_forward,
                    route_forward_num_entries,
                    route_forward_offset,
                    route_forward,
                    route_mem_num_entries,
                    route_mem_offset,
                    route_mem,
                    ack_left_mem_num_entries,
                    ack_left_mem_offset,
                    ack_left_mem,
                    ack_num_mem_num_entries,
                    ack_num_mem_offset,
                    ack_num_mem,
                    ack_stack_mem_num_entries,
                    ack_stack_mem_offset,
                    ack_stack_mem,
                    ack_stack_ptr)
        self.router = Router(ind, self, initial_sync_info, initial_task_translation, has_external)
        self.isa_ctrl = ISAController(self)
        self.processingEvent = ProcessingEvent(self)
        self.debug_module = DebugModule(self)

        # Initialize the register file
        self.reg = Register()

        # Initialize the loop ctrl
        self.loop_ctrl = LoopCtrl(self.reg, self.mem)
        
        # Set the timestep per task
        for ltask_id in range(GV.NUM_SCHEDULED_TASKS[self.ind]):
            self.reg.write(GV.reg_general_region_size + ltask_id, -1 + GV.initial_timestep[ltask_id])

        # Initialize the PC controller
        self.pc = PC(self, table_key_list, base_list, bound_list, inst_list)

    # called at each tick
    cpdef core_advance(self):
        # For each cycle
        # print(self.processingEvent.busy())
        if self.processingEvent.busy():
            GV.valid_advance = True
            # if self.external or not self.has_external:
            # if self.external or \
            #     not self.has_external and 'bci_send' in self.processingEvent.busy():
            #     self.process_external()
            # else:
            #     self.process_events()
            if self.external:
                self.process_external()
            elif self.has_external:
                self.process_events()
            else:
                # single core, no external
                if 'bci_send' in self.processingEvent.busy():
                    self.process_events()
                    self.process_external()
                else:
                    self.process_events()
        else:
            if not self.router.in_event_buf.empty():
                GV.valid_advance = True
                self.schedule_events()

        # The external buffer is waiting for the bci signal
        if self.external or not self.has_external:
            self.external_module.pending_bci_send()
        self.router.task_scheduler.decrement_budget()
        self.router.packet_generation()
        self.router.packet_reception()

    # convert a spike to synapse events (iterate over the connections)
    cpdef schedule_events(self):
        # Return if busy
        if self.processingEvent.busy(): return

        (event_type, event_data, _), ltask_id = self.router.in_event_buf.popleft()

        gtask_id = GV.ltask_to_gtask[self.ind][ltask_id]
        timestep = GV.per_core_timestep[gtask_id][self.ind]
        if 'bci_send' in event_type:
            GV.leading_timestep[gtask_id] = timestep + 1
            GV.per_core_timestep[gtask_id][self.ind] = timestep + 1

        # We do two things when scheduling an event
        # 1) We notify the router to perform sync-related operations after scheduling an event
        # 2) We actually schedule the event (so that it can be processed in the next cycle)
        event = {'mode' : 'event',
                 'type' : 'start',
                 'event_type' : event_type,
                 'event_data' : event_data,
                 'task_id' : ltask_id}
        self.router.out_event_buf.append(Event({'cyc': GV.cyc, 'data': event}))
        self.processingEvent.schedule_event(event_type, event_data, ltask_id)

        Profiler.calc_event_number(gtask_id, event_type, timestep, self.ind)
        op_count['out_event_buf_push'] += 1
        op_count['event_to_base_bound_table'] += 1

    # Process 
    cpdef process_events(self):
        event_type = self.processingEvent.busy()
        # Get the metadata related to the event
        event_data, cyc, ltask_id = self.processingEvent.get_data(event_type)

        # Profile per-task and per-event cycle
        gtask_id = GV.ltask_to_gtask[self.ind][ltask_id]
        timestep = GV.per_core_timestep[gtask_id][self.ind]
        Profiler.calc_event_cyc(gtask_id, event_type, timestep, self.ind)
        
        self.isa_ctrl.set_task(ltask_id)
        self.isa_ctrl.set_event(event_type)
        
        if cyc == None:
            output_state = {'cyc' : GV.cyc, 'reg' : self.reg, 'pc' : 0}
            # debug = True
            debug = False
            # print("EVENT {}".format(event_type))
            output_state = self.pc.process_event(output_state, ltask_id, event_type, debug)
            cyc = output_state['cyc']
            self.processingEvent.set_end_state(event_type, cyc)

        assert(GV.cyc <= cyc)
        if GV.cyc == cyc:
            # Notify the router that the event is done
            self.processingEvent.event_done(event_type)
            event = {'mode' : 'event',
                     'type' : 'end',
                     'event_type' : event_type,
                     'event_data' : event_data,
                     'task_id' : ltask_id}
            self.router.out_event_buf.append(Event({'cyc': GV.cyc, 'data': event}))
            op_count['out_event_buf_push'] += 1

    # Process 
    cpdef process_external(self):
        event_type = self.processingEvent.busy()
        # Get the metadata related to the event
        event_data, cyc, ltask_id = self.processingEvent.get_data(event_type)

        # Profile per-task and per-event cycle
        gtask_id = GV.ltask_to_gtask[self.ind][ltask_id]
        timestep = GV.per_core_timestep[gtask_id][self.ind]
        Profiler.calc_event_cyc(gtask_id, event_type, timestep, self.ind)
        
        output_state = {'cyc' : GV.cyc, 'reg' : self.reg, 'pc' : 0}
        # debug = True
        debug = False
        if debug: print("event_type", event_type)

        # Has a dedicated operation for the bci_send operation in the external
        if 'bci_send' in event_type:
                event, external_dat = self.external_module.external_data(gtask_id)
                if event == 'init':
                    # Send the target budget
                    budget = (GV.target_cyc[gtask_id] - GV.cyc) // GV.sim_params['1ms_cyc']
                    packet = {'mode' : 'packet',
                              'task_id' : ltask_id,
                              'dst_x' : self.reg.read(2),
                              'dst_y' : self.reg.read(3),
                              'lid' : 0,
                              'data' : budget,
                              'event_type' : event_type,
                              'type' : 'init'}
                    self.router.out_event_buf.append(Event({'cyc': GV.cyc + 1, 'data': packet}))
                    op_count["out_event_buf_push"] += 1
                    return
                elif event == 'event':
                    electrode_id, recording_dat = external_dat
                    packet = {'mode' : 'packet',
                              'task_id' : ltask_id,
                              'dst_x' : self.reg.read(2),
                              'dst_y' : self.reg.read(3),
                              'lid' : electrode_id,
                              'data' : recording_dat,
                              'event_type' : event_type,
                              'type' : 'bci'}
                    self.router.out_event_buf.append(Event({'cyc': GV.cyc + 1, 'data': packet}))
                    op_count["out_event_buf_push"] += 1
                    return
                elif event == 'no event':
                    self.processingEvent.event_done(event_type)
                    event = {'mode' : 'event',
                             'type' : 'end',
                             'event_type' : event_type,
                             'event_data' : event_data,
                             'task_id' : ltask_id}
                    self.router.out_event_buf.append(Event({'cyc': GV.cyc + 1, 'data': event}))
                    op_count['out_event_buf_push'] += 1
                    return
                else:
                    assert(0)
        else:
            # Notify the router that the event is done
            self.processingEvent.event_done(event_type)
            event = {'mode' : 'event',
                     'type' : 'end',
                     'event_type' : event_type,
                     'event_data' : event_data,
                     'task_id' : ltask_id}
            self.router.out_event_buf.append(Event({'cyc': GV.cyc, 'data': event}))
            op_count['out_event_buf_push'] += 1
