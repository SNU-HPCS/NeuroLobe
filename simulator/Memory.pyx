cimport cython
import numpy as np
cimport numpy as np
from copy import deepcopy
import Task
# from math import log2, ceil

import GlobalVars as GV
import copy
import sys

from Energy import op_count

cdef class Memory:

    def __init__(self, Core.Core core,
                   list state_mem,
                   list state_mem_offset,
                   list hist_mem_num_entries,
                   list hist_metadata,
                   list hist_pos,
                   list hist_mem_offset,
                   list hist_mem,
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
                   list ack_stack_ptr):

        self.core = core
        self.ind = core.ind

        self.hist_task_translator = [{} for _ in range(GV.MAX_LOCAL_TASK)]
        index = 0
        for ltask_id in range(len(hist_metadata)):
            for key in hist_metadata[ltask_id]['unit_offset'].keys():
                self.hist_task_translator[ltask_id][key] = index
                index += 1

        self.state_mem = state_mem
        self.state_mem_offset = state_mem_offset

        self.hist_mem_num_entries = hist_mem_num_entries
        self.hist_metadata = hist_metadata
        self.hist_mem_offset = hist_mem_offset
        self.hist_mem = hist_mem

        # Set the tasks
        self.hist_pos = hist_pos
        # reuse buffer for hist_mem
        self.hist_mem_cache_size = 3
        self.hist_mem_cache_addr =  [0 for _ in range(self.hist_mem_cache_size)]
        self.hist_mem_cache_dirty = [False for _ in range(self.hist_mem_cache_size)]
        self.hist_mem_cache_dat = [{} for _ in range(self.hist_mem_cache_size)]
        self.hist_mem_cache_queue =  []
        if len(self.hist_mem):
            self.hist_mem_cache_dat[0] = self.hist_mem[0]
            self.hist_mem_cache_queue.append(0)

        self.corr_mem_num_entries = corr_mem_num_entries
        self.corr_mem_offset = corr_mem_offset
        self.corr_mem = corr_mem
        # reuse buffer for corr_mem
        self.corr_mem_cache_size = 3
        self.corr_mem_cache_addr =  [0 for _ in range(self.corr_mem_cache_size)]
        self.corr_mem_cache_dirty = [False for _ in range(self.corr_mem_cache_size)]
        self.corr_mem_cache_dat = [0 for _ in range(self.corr_mem_cache_size)]
        self.corr_mem_cache_queue =  []
        if len(self.corr_mem):
            self.corr_mem_cache_dat[0] = self.corr_mem[0]
            self.corr_mem_cache_queue.append(0)

        self.corr_forward_num_entries = corr_forward_num_entries
        self.corr_forward_offset = corr_forward_offset
        self.corr_forward = corr_forward

        self.route_forward_num_entries = route_forward_num_entries
        self.route_forward_offset = route_forward_offset
        self.route_forward = route_forward

        self.route_mem_num_entries = route_mem_num_entries
        self.route_mem_offset = route_mem_offset
        self.route_mem = route_mem

        self.ack_left_mem_num_entries = ack_left_mem_num_entries
        self.ack_left_mem_offset = ack_left_mem_offset
        self.ack_left_mem = ack_left_mem

        self.ack_num_mem_num_entries = ack_num_mem_num_entries
        self.ack_num_mem_offset = ack_num_mem_offset
        self.ack_num_mem = ack_num_mem

        self.ack_stack_mem_num_entries = ack_stack_mem_num_entries
        self.ack_stack_mem_offset = ack_stack_mem_offset
        self.ack_stack_mem = ack_stack_mem
        self.ack_stack_ptr = ack_stack_ptr

    cpdef increment_pos(self, ltask_id):
        # iterate over the position tracking register 
        # and increment the value (related to the ltask_id)
        for pos in self.hist_pos:
            if ltask_id == pos['task']:
                data_type = pos['data']
                length = self.hist_metadata[ltask_id]['length'][data_type]
                pos['val'] = pos['val'] + 1
                if pos['val'] >= length:
                    pos['val'] -= length

    cpdef read_state_mem(self, data_type, mem_offset, task_id, virtual_addr):
        physical_addr = virtual_addr + mem_offset
        stall = 0
        # Directly access the write buffer
        op_count['state_mem_read'] += 1
        return copy.deepcopy(self.state_mem[physical_addr]), stall

    cpdef write_state_mem(self, data_type, mem_offset, task_id, virtual_addr, write_dat):
        physical_addr = virtual_addr + mem_offset
        stall = 0 
        self.state_mem[physical_addr] = write_dat
        op_count['state_mem_write'] += 1
        return stall

    cpdef accum_state_mem(self, func_type, data_type, mem_offset, ltask_id, virtual_addr, write_dat, alu_dat = None):
        physical_addr = virtual_addr + mem_offset
        stall = 0
        op_count['state_mem_read'] += 1
        op_count['state_mem_write'] += 1
        op_count['near_cache_alu'] += 1
        # We support three different operation modes
        if func_type == 'add':
            self.state_mem[physical_addr] += write_dat
            return None, stall
        elif func_type == 'sub':
            self.state_mem[physical_addr] -= write_dat
            return None, stall
        elif func_type == 'mac':
            self.state_mem[physical_addr] += write_dat * alu_dat
            return None, stall
        elif func_type == 'pop':
            data = copy.deepcopy(self.state_mem[physical_addr])
            self.state_mem[physical_addr] = 0
            return data, stall
        else:
            assert(0)

    # This involves at most three cycles (w/o loop ctrl)
    cpdef get_hist_addr(self, data_type, ltask_id, virtual_addr, pos):
        unit_offset = self.hist_metadata[ltask_id]['unit_offset'][data_type]
        log2_precision = self.hist_metadata[ltask_id]['log2_precision'][data_type]
        length = self.hist_metadata[ltask_id]['length'][data_type]

        addr = unit_offset * virtual_addr

        pos += self.hist_pos[self.hist_task_translator[ltask_id][data_type]]['val']
        while True:
            if pos >= length:
                pos -= length
            elif pos < 0:
                pos += length
            else:
                break
        
        # All these functions can be replaced with bitwise operators
        entries_per_line = GV.MEM_WIDTH['hist_mem'] >> log2_precision
        assert(pos >= 0 and pos < length)
        addr += pos >> (5 - log2_precision)
        offset = (pos & (entries_per_line - 1)) << log2_precision

        assert(addr < self.hist_mem_num_entries[ltask_id][data_type])
        return addr, offset

    cpdef read_hist_mem(self, data_type, mem_offset, ltask_id, virtual_addr, pos, no_loop_ctrl = False):
        physical_addr, offset = self.get_hist_addr(data_type, ltask_id, virtual_addr, pos)
        physical_addr += mem_offset
        if no_loop_ctrl:
            op_count['inst_mem'] += 3
            stall = 3
        else: stall = 0
        # Directly access the write buffer
        if physical_addr in self.hist_mem_cache_addr:
            write_entry = self.hist_mem_cache_addr.index(physical_addr)
            # move the entry to the end of the queue
            self.hist_mem_cache_queue.remove(write_entry)
            self.hist_mem_cache_queue.append(write_entry)
            op_count['hist_mem_cache_access'] += 1
            return copy.deepcopy(self.hist_mem_cache_dat[write_entry][offset]), stall
        # on write buffer miss
        else:
            # allocate new entry to write buffer
            if len(self.hist_mem_cache_queue) < self.hist_mem_cache_size:
                write_entry = len(self.hist_mem_cache_queue)
                self.hist_mem_cache_queue.append(write_entry)
            else:
                # evict LRU
                write_entry = self.hist_mem_cache_queue[0]
                # move the entry to the end of the queue
                self.hist_mem_cache_queue.remove(write_entry)
                self.hist_mem_cache_queue.append(write_entry)

                if self.hist_mem_cache_dirty[write_entry]:
                    self.hist_mem[self.hist_mem_cache_addr[write_entry]] = self.hist_mem_cache_dat[write_entry].copy()
                    op_count['hist_mem_cache_access'] += 1
                    op_count['hist_mem_write'] += 1
                    stall = 1 # stall for simulataneous read/write
                else:
                    assert(self.hist_mem[self.hist_mem_cache_addr[write_entry]] == self.hist_mem_cache_dat[write_entry])
            self.hist_mem_cache_dat[write_entry] = self.hist_mem[physical_addr].copy()
            self.hist_mem_cache_addr[write_entry] = physical_addr
            self.hist_mem_cache_dirty[write_entry] = False
            op_count['hist_mem_read'] += 1
            op_count['hist_mem_cache_access'] += 1

            return copy.deepcopy(self.hist_mem_cache_dat[write_entry][offset]), stall

    cpdef write_hist_mem(self, data_type, mem_offset, ltask_id, virtual_addr, pos, write_dat, no_loop_ctrl = False):
        physical_addr, offset = self.get_hist_addr(data_type, ltask_id, virtual_addr, pos)
        physical_addr += mem_offset
        if no_loop_ctrl:
            op_count['inst_mem'] += 3
            stall = 3
        else: stall = 0

        # Directly access the write buffer
        if physical_addr in self.hist_mem_cache_addr:
            write_entry = self.hist_mem_cache_addr.index(physical_addr)
            # move the entry to the end of the queue
            self.hist_mem_cache_queue.remove(write_entry)
            self.hist_mem_cache_queue.append(write_entry)
            self.hist_mem_cache_dat[write_entry][offset] = write_dat
            self.hist_mem_cache_dirty[write_entry] = True
            op_count['hist_mem_cache_access'] += 1

        # on write buffer miss
        else:
            # allocate new entry to write buffer
            if len(self.hist_mem_cache_queue) < self.hist_mem_cache_size:
                write_entry = len(self.hist_mem_cache_queue)
                self.hist_mem_cache_queue.append(write_entry)
            else:
                # evict LRU
                write_entry = self.hist_mem_cache_queue[0]
                # move the entry to the end of the queue
                self.hist_mem_cache_queue.remove(write_entry)
                self.hist_mem_cache_queue.append(write_entry)

                if self.hist_mem_cache_dirty[write_entry]:
                    self.hist_mem[self.hist_mem_cache_addr[write_entry]] = self.hist_mem_cache_dat[write_entry].copy()
                    op_count['hist_mem_cache_access'] += 1
                    op_count['hist_mem_write'] += 1
                    stall = 1 # stall for simulataneous read/write
                else:
                    assert(self.hist_mem[self.hist_mem_cache_addr[write_entry]] == self.hist_mem_cache_dat[write_entry])
            self.hist_mem_cache_dat[write_entry] = self.hist_mem[physical_addr].copy()
            self.hist_mem_cache_addr[write_entry] = physical_addr
            self.hist_mem_cache_dat[write_entry][offset] = write_dat
            self.hist_mem_cache_dirty[write_entry] = True
            op_count['hist_mem_read'] += 1
            op_count['hist_mem_cache_access'] += 1
        return stall

    cpdef accum_hist_mem(self, func_type, data_type, mem_offset, ltask_id, virtual_addr, pos, write_dat, alu_dat = None):
        physical_addr, offset = self.get_hist_addr(data_type, ltask_id, virtual_addr, pos)
        physical_addr += mem_offset
        stall = 0

        # Directly access the write buffer
        if physical_addr in self.hist_mem_cache_addr:
            write_entry = self.hist_mem_cache_addr.index(physical_addr)
            # move the entry to the end of the queue
            self.hist_mem_cache_queue.remove(write_entry)
            self.hist_mem_cache_queue.append(write_entry)
            op_count['hist_mem_cache_access'] += 1

        # on write buffer miss
        else:
            # allocate new entry to write buffer
            if len(self.hist_mem_cache_queue) < self.hist_mem_cache_size:
                write_entry = len(self.hist_mem_cache_queue)
                self.hist_mem_cache_queue.append(write_entry)
            else:
                # evict LRU
                write_entry = self.hist_mem_cache_queue[0]
                # move the entry to the end of the queue
                self.hist_mem_cache_queue.remove(write_entry)
                self.hist_mem_cache_queue.append(write_entry)

                if self.hist_mem_cache_dirty[write_entry]:
                    self.hist_mem[self.hist_mem_cache_addr[write_entry]] = self.hist_mem_cache_dat[write_entry].copy()
                    op_count['hist_mem_cache_access'] += 1
                    op_count['hist_mem_write'] += 1
                    stall = 1 # stall for simulataneous read/write
                else:
                    assert(self.hist_mem[self.hist_mem_cache_addr[write_entry]] == self.hist_mem_cache_dat[write_entry])
            self.hist_mem_cache_dat[write_entry] = self.hist_mem[physical_addr].copy()
            self.hist_mem_cache_addr[write_entry] = physical_addr
            op_count['hist_mem_read'] += 1
            op_count['hist_mem_cache_access'] += 1

        op_count['near_cache_alu'] += 1
        # We support three different operation modes
        self.hist_mem_cache_dirty[write_entry] = True
        if func_type == 'add':
            self.hist_mem_cache_dat[write_entry][offset] += write_dat
            return None, stall
        elif func_type == 'sub':
            self.hist_mem_cache_dat[write_entry][offset] -= write_dat
            return None, stall
        elif func_type == 'mac':
            self.hist_mem_cache_dat[write_entry][offset] += write_dat * alu_dat
            return None, stall
        elif func_type == 'pop':
            data = copy.deepcopy(self.hist_mem_cache_dat[write_entry][offset])
            self.hist_mem_cache_dat[write_entry][offset] = 0
            return data, stall
        else:
            assert(0)

    cpdef read_corr_forward(self, data_type, mem_offset, ltask_id, virtual_addr):
        assert(0 <= virtual_addr < self.corr_forward_num_entries[ltask_id][data_type])
        physical_addr = virtual_addr + mem_offset
        #physical_addr_end = physical_addr_start + 1

        # decrement the start address by 1
        start, offset, num_iter = self.corr_forward[physical_addr]
        op_count['corr_forward_read'] += 1 # corr forward read access 2 values at once
        return start, offset, num_iter

    cpdef read_corr(self, data_type, ltask_id, virtual_addr, offset):
        physical_addr = virtual_addr
        stall = 0

        # if the data can be reused
        if physical_addr in self.corr_mem_cache_addr:
            reuse_entry = self.corr_mem_cache_addr.index(physical_addr)
            # move the entry to the end of the queue
            self.corr_mem_cache_queue.remove(reuse_entry)
            self.corr_mem_cache_queue.append(reuse_entry)
            op_count['corr_mem_cache_access'] += 1
            return copy.deepcopy(self.corr_mem_cache_dat[reuse_entry][offset]), stall
        # if the data reuse miss
        else:
            # allocate new entry to reuse_buffer
            if len(self.corr_mem_cache_queue) < self.corr_mem_cache_size:
                reuse_entry = len(self.corr_mem_cache_queue)
                self.corr_mem_cache_queue.append(reuse_entry)
            else:
                # evict LRU
                reuse_entry = self.corr_mem_cache_queue[0]
                # move the entry to the end of the queue
                self.corr_mem_cache_queue.remove(reuse_entry)
                self.corr_mem_cache_queue.append(reuse_entry)
                # if dirty, write the data to the memory
                if self.corr_mem_cache_dirty[reuse_entry]:
                    self.corr_mem[self.corr_mem_cache_addr[reuse_entry]] = self.corr_mem_cache_dat[reuse_entry]
                    op_count['corr_mem_cache_access'] += 1
                    op_count['corr_mem_write'] += 1
                    # stall = 1 # stall for simulataneous read/write
                else:
                    assert(self.corr_mem[self.corr_mem_cache_addr[reuse_entry]] == self.corr_mem_cache_dat[reuse_entry])
            self.corr_mem_cache_dat[reuse_entry] = self.corr_mem[physical_addr]
            self.corr_mem_cache_addr[reuse_entry] = physical_addr
            self.corr_mem_cache_dirty[reuse_entry] = False
            op_count['corr_mem_read'] += 1
            op_count['corr_mem_cache_access'] += 1
            return copy.deepcopy(self.corr_mem_cache_dat[reuse_entry][offset]), stall

    cpdef write_corr(self, data_type, ltask_id, virtual_addr, offset, write_dat):
        physical_addr = virtual_addr
        stall = 0

        # if the data can be reused
        if physical_addr in self.corr_mem_cache_addr:
            reuse_entry = self.corr_mem_cache_addr.index(physical_addr)
            # move the entry to the end of the queue
            self.corr_mem_cache_queue.remove(reuse_entry)
            self.corr_mem_cache_queue.append(reuse_entry)
            self.corr_mem_cache_dat[reuse_entry][offset] = write_dat
            self.corr_mem_cache_dirty[reuse_entry] = True
            op_count['corr_mem_cache_access'] += 1
        else:
            # allocate new entry to reuse_buf
            if len(self.corr_mem_cache_queue) < self.corr_mem_cache_size:
                reuse_entry = len(self.corr_mem_cache_queue)
                self.corr_mem_cache_queue.append(reuse_entry)
            else:
                # evict LRU
                reuse_entry = self.corr_mem_cache_queue[0]
                # move the entry to the end of the queue
                self.corr_mem_cache_queue.remove(reuse_entry)
                self.corr_mem_cache_queue.append(reuse_entry)
                if self.corr_mem_cache_dirty[reuse_entry]:
                    self.corr_mem[self.corr_mem_cache_addr[reuse_entry]] = self.corr_mem_cache_dat[reuse_entry]
                    op_count['corr_mem_cache_access'] += 1
                    op_count['corr_mem_write'] += 1
                    stall = 1 # stall for simulataneous read/write
                else:
                    assert(self.corr_mem[self.corr_mem_cache_addr[reuse_entry]] == self.corr_mem_cache_dat[reuse_entry])
            self.corr_mem_cache_dat[reuse_entry] = self.corr_mem[physical_addr]
            self.corr_mem_cache_addr[reuse_entry] = physical_addr
            self.corr_mem_cache_dat[reuse_entry][offset] = write_dat
            self.corr_mem_cache_dirty[reuse_entry] = True
            op_count['corr_mem_read'] += 1
            op_count['corr_mem_cache_access'] += 1
        return stall

    cpdef accum_corr_mem(self, func_type, data_type, ltask_id, virtual_addr, offset, write_dat, alu_dat = None):
        physical_addr = virtual_addr
        stall = 0

        if physical_addr in self.corr_mem_cache_addr:
            reuse_entry = self.corr_mem_cache_addr.index(physical_addr)
            # print("accum reuse", reuse_entry, physical_addr)
            # move the entry to the end of the queue
            self.corr_mem_cache_queue.remove(reuse_entry)
            self.corr_mem_cache_queue.append(reuse_entry)
            op_count['corr_mem_cache_access'] += 1
        else:
            # allocate new entry to reuse_buf
            if len(self.corr_mem_cache_queue) < self.corr_mem_cache_size:
                reuse_entry = len(self.corr_mem_cache_queue)
                self.corr_mem_cache_queue.append(reuse_entry)
            else:
                # search entry to evict
                reuse_entry = self.corr_mem_cache_queue[0]
                self.corr_mem_cache_queue.remove(reuse_entry)
                self.corr_mem_cache_queue.append(reuse_entry)

                # write back from the buffer
                if self.corr_mem_cache_dirty[reuse_entry]:
                    self.corr_mem[self.corr_mem_cache_addr[reuse_entry]] = self.corr_mem_cache_dat[reuse_entry]
                    op_count['corr_mem_cache_access'] += 1
                    op_count['corr_mem_write'] += 1
                    stall = 1 # stall for simulataneous read/write
                else:
                    assert(self.corr_mem[self.corr_mem_cache_addr[reuse_entry]] == self.corr_mem_cache_dat[reuse_entry])
            self.corr_mem_cache_addr[reuse_entry] = physical_addr
            self.corr_mem_cache_dat[reuse_entry] = self.corr_mem[physical_addr]
            op_count['corr_mem_read'] += 1
            op_count['corr_mem_cache_access'] += 1

        op_count['near_cache_alu'] += 1
        # We support three different operation modes
        self.corr_mem_cache_dirty[reuse_entry] = True
        if func_type == 'add':
            self.corr_mem_cache_dat[reuse_entry][offset] += write_dat
            return None, stall
        elif func_type == 'sub':
            self.corr_mem_cache_dat[reuse_entry][offset] -= write_dat
            return None, stall
        elif func_type == 'mac':
            self.corr_mem_cache_dat[reuse_entry][offset] += write_dat * alu_dat
            return None, stall
        elif func_type == 'pop':
            data = copy.deepcopy(self.corr_mem_cache_dat[reuse_entry][offset])
            self.corr_mem_cache_dat[reuse_entry][offset] = 0
            return data, stall
        else:
            assert(0)

    cpdef read_route_forward(self, data_type, ltask_id, virtual_addr):
        assert(0 <= virtual_addr < self.route_forward_num_entries[ltask_id][data_type])
        physical_addr_start = virtual_addr + self.route_forward_offset[ltask_id][data_type]
        physical_addr_end = physical_addr_start + 1
        start = self.route_forward[physical_addr_start]
        end = self.route_forward[physical_addr_end]
        op_count['route_mem_forward_read'] += 1 # access 2 values at once
        return start, end

    cpdef read_route(self, data_type, ltask_id, virtual_addr):
        physical_addr = virtual_addr 
        # We do not add offset for the memory w/ forward 
        # + self.route_forward_offset[ltask_id][data_type]
        data = copy.deepcopy(self.route_mem[physical_addr])
        op_count['route_mem_read'] += 1
        return data

    cpdef read_ack_left(self, data_type, ltask_id, virtual_addr):
        assert(0 <= virtual_addr < self.ack_left_mem_num_entries[ltask_id][data_type])
        physical_addr = virtual_addr + self.ack_left_mem_offset[ltask_id][data_type]
        op_count['ack_left_mem_read'] += 1
        return self.ack_left_mem[physical_addr]

    # write value to the entry indicated by ack_vaddr
    cpdef write_ack_left(self, data_type, ltask_id, virtual_addr, write_dat):
        assert(0 <= virtual_addr < self.ack_left_mem_num_entries[ltask_id][data_type])
        physical_addr = virtual_addr + self.ack_left_mem_offset[ltask_id][data_type]
        op_count['ack_left_mem_write'] += 1
        self.ack_left_mem[physical_addr] = write_dat

    cpdef read_ack_num(self, data_type, ltask_id, virtual_addr):
        assert(0 <= virtual_addr < self.ack_num_mem_num_entries[ltask_id][data_type])
        physical_addr = virtual_addr + self.ack_num_mem_offset[ltask_id][data_type]
        op_count['ack_num_mem_read'] += 1
        return self.ack_num_mem[physical_addr]

    # write value to the entry indicated by ack_vaddr
    cpdef write_ack_num(self, data_type, ltask_id, virtual_addr, write_dat):
        assert(0 <= virtual_addr < self.ack_num_mem_num_entries[ltask_id][data_type])
        physical_addr = virtual_addr + self.ack_num_mem_offset[ltask_id][data_type]
        op_count['ack_num_mem_write'] += 1
        self.ack_num_mem[physical_addr] = write_dat

    # generate entry for partial distance accumulation
    cpdef push_ack_stack_mem(self, data_type, ltask_id, init_write_dat):
        virtual_addr = self.ack_stack_ptr[ltask_id][data_type]
        self.ack_stack_ptr[ltask_id][data_type] += 1
        assert(0 <= virtual_addr < self.ack_stack_mem_num_entries[ltask_id][data_type])
        physical_addr = virtual_addr + self.ack_stack_mem_offset[ltask_id][data_type]
        op_count['ack_stack_mem_push'] += 1
        self.ack_stack_mem[physical_addr] = init_write_dat

    # pop the entry and decrement stack ptr
    cpdef pop_ack_stack_mem(self, data_type, ltask_id):
        if  self.ack_stack_ptr[ltask_id][data_type] == 0:
            is_empty = True
            data = 0
        else:
            is_empty = False
            self.ack_stack_ptr[ltask_id][data_type] -= 1
            virtual_addr = self.ack_stack_ptr[ltask_id][data_type]
            physical_addr = virtual_addr + self.ack_stack_mem_offset[ltask_id][data_type]
            data = self.ack_stack_mem[physical_addr]
            data = copy.deepcopy(data)
        return is_empty, data

    # save current memories to mem_state
    cpdef save_mem(self, mem_state):
        # Writeback all caches before saving the memory
        for entry in self.corr_mem_cache_queue:
            if self.corr_mem_cache_dirty[entry]:                    
                self.corr_mem[self.corr_mem_cache_addr[entry]] = self.corr_mem_cache_dat[entry]
        for entry in self.hist_mem_cache_queue:
            if self.hist_mem_cache_dirty[entry]:                    
                self.hist_mem[self.hist_mem_cache_addr[entry]] = self.hist_mem_cache_dat[entry].copy()

        mem_state['state_mem'] = self.state_mem.copy()
        mem_state['hist_pos'] = self.hist_pos.copy()
        mem_state['hist_mem'] = self.hist_mem.copy()
        mem_state['corr_mem'] = self.corr_mem.copy()
