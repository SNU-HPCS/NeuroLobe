import sys
import GlobalVars as GV
import numpy as np
from heapq import heappop, heappush
from Event import Event
# Iteration type
from Core import Core, Register, LoopCtrl
from Router import Router
from Router import Synchronizer
from Memory import Memory
from Debug import DebugModule

import sys

from Energy import op_count

class ISAController:
    def __init__(self, core: Core):
        self.core = core
        self.router = core.router
        self.ind = core.ind
        self.mem = core.mem
        self.has_external = core.has_external
        self.synchronizer = core.router.synchronizer

        #
        self.task_id = None
        self.event_type = None

        self.use_corr = False
        self.use_hist = False
        self.use_state = False

    def set_task(self, task_id):
        self.task_id = task_id

    def set_event(self, event_type):
        self.event_type = event_type

    # opcode        = 8 bit
    # func_type     = 3 bit
    # reg_addr1     = 5 bit
    # immediate     = 32 bit
    # jump_addr     = 8 bit
    # TOTAL         = 56 bit
    def if_imm_begin(self, input_state = None, func_type = None, reg_addr1 = None, immediate = None, jump_addr = None):
        cyc = input_state['cyc']
        reg = input_state['reg']
        pc = input_state['pc']

        # Suffers from the pipeline stall
        cyc += 2
        op_count["reg_op"] += 1 # register read
        input1 = reg.read(reg_addr1)

        op_count["alu_logical"] += 1
        if func_type == 'eq':
            cond = (input1 == immediate)
        elif func_type == 'neq':
            cond = (input1 != immediate)
        elif func_type == 'gt':
            cond = (input1 > immediate)
        elif func_type == 'ge':
            cond = (input1 >= immediate)
        elif func_type == 'lt':
            cond = (input1 < immediate)
        elif func_type == 'le':
            cond = (input1 <= immediate)
        else:
            assert(0)

        if cond:
            pc += 1
        else:
            pc = jump_addr

        return {'cyc' : cyc, 'reg' : reg, 'pc' : pc}

    # func_type = 3 bit
    # reg_addr1 = 5 bit
    # reg_addr2 = 5 bit
    # jump_addr = 8 bit (256 instructions per event)
    def if_reg_begin(self, input_state = None, func_type = None, reg_addr1 = None, reg_addr2 = None, jump_addr = None):
        cyc = input_state['cyc']
        reg = input_state['reg']
        pc = input_state['pc']

        # Suffers from the pipeline stall
        cyc += 2
        op_count["reg_op"] += 1 # register read
        input1 = reg.read(reg_addr1)
        input2 = reg.read(reg_addr2)

        op_count["alu_logical"] += 1
        if func_type == 'eq':
            cond = (input1 == input2)
        elif func_type == 'neq':
            cond = (input1 != input2)
        elif func_type == 'gt':
            cond = (input1 > input2)
        elif func_type == 'ge':
            cond = (input1 >= input2)
        elif func_type == 'lt':
            cond = (input1 < input2)
        elif func_type == 'le':
            cond = (input1 <= input2)
        else:
            assert(0)

        if cond:
            pc += 1
        else:
            pc = jump_addr

        return {'cyc' : cyc, 'reg' : reg, 'pc' : pc}

    # opcode        = 8 bit
    # func_type     = 1 bit
    # mem_offset    = 14 bit
    # reg_addr1     = 5 bit
    # immediate     = 16 bit
    # jump_addr     = 8 bit
    # loop_offset   = 10 bit (the memory bits for each iteration)
    # iter_order    = 1 bit
    # include ts    = 1 bit
    # TOTAL         = 64 bit
    def loop_corr(self, input_state = None, func_type = 'outer',
                  memory_name = None, mem_offset = None, reg_addr1 = None, immediate = 0,
                  jump_addr = None, loop_offset = None, forward_order = True, include_ts = True, pipelined = False):
        cyc = input_state['cyc']
        reg = input_state['reg']
        pc = input_state['pc']

        # 3 indicates pipeline bubble
        if pipelined and func_type == 'outer':
            cyc += 4
        else:
            cyc += 1

        # We should retrieve the counter start addr
        if include_ts and forward_order:
            init_timestep = - immediate + 1
        elif include_ts and not forward_order:
            init_timestep = 0
        elif not include_ts and forward_order:
            init_timestep = - immediate
        elif not include_ts and not forward_order:
            init_timestep = - 1
        else:
            assert(0)

        # the register address is set to 0
        if func_type == 'outer':
            pid = reg.read(0) # reg read
            outer_offset, inner_offset, num_iter = self.mem.read_corr_forward(memory_name, mem_offset, self.task_id, pid)

        else: # 'inner'
            outer_offset = None
            inner_offset = None
            num_iter = int(immediate)

        # we assume reading the connection memory once every outer loop
        # for lid and delay
        if func_type == 'outer':
            op_count["conn_mem_read"] += 1
        op_count["reg_op"] += 1 # reg write - performed within 1 cycle with reg read

        # Set the initial timestep register
        reg.write(reg_addr1, init_timestep)

        pc = self.core.loop_ctrl.insert(func_type, outer_offset, inner_offset, num_iter,
                                        reg_addr1, pc + 1, jump_addr, loop_offset, forward_order)

        return {'cyc' : cyc, 'reg' : reg, 'pc' : pc, 'pipelined' : pipelined}

    # opcode        = 8 bit
    # TOTAL         = 8 bit
    def corr_end(self, input_state = None, pipelined = False):
        cyc = input_state['cyc']
        reg = input_state['reg']
        pc = input_state['pc']

        if pipelined:
            op_count["pipeline"] += 1
        else:
            cyc += 1

        pc = self.core.loop_ctrl.iterate()

        return {'cyc' : cyc, 'reg' : reg, 'pc' : pc, 'pipelined' : pipelined}

    # v1: with imm
    # opcode        = 8 bit
    # func_type     = 1 bit
    # reg_addr1     = 5 bit
    # immediate     = 16 bit
    # jump_addr     = 8 bit
    # TOTAL         = 38 bit

    # v2: without imm
    # opcode        = 8 bit
    # func_type     = 1 bit
    # reg_addr1     = 5 bit
    # reg_addr2     = 5 bit
    # jump_addr     = 8 bit
    # TOTAL         = 27 bit
    def loop_unit(self, input_state = None, func_type = 'outer',
                  reg_addr1 = None, immediate = None,
                  reg_addr2 = None, jump_addr = None):
        cyc = input_state['cyc']
        reg = input_state['reg']
        pc = input_state['pc']

        cyc += 1

        outer_offset = 0
        inner_offset = 0
        if immediate != None:
            num_iter = immediate
        else:
            num_iter = reg.read(reg_addr2)

        loop_offset = GV.MEM_WIDTH['corr_mem']

        # Set the initial timestep register
        reg.write(reg_addr1, 0)
        # register read/write concurrently
        pc = self.core.loop_ctrl.insert(func_type, outer_offset, inner_offset, num_iter,
                                        reg_addr1, pc + 1, jump_addr, loop_offset, True)

        return {'cyc' : cyc, 'reg' : reg, 'pc' : pc}

    # opcode        = 8 bit
    # TOTAL         = 8 bit
    def unit_end(self, input_state = None):
        cyc = input_state['cyc']
        reg = input_state['reg']
        pc = input_state['pc']

        cyc += 1
        pc = self.core.loop_ctrl.iterate()

        return {'cyc' : cyc, 'reg' : reg, 'pc' : pc}

    ## opcode        = 8 bit
    ## mem_offset    = 10 bit
    ## reg_addr1     = 5 bit
    ## jump_addr     = 8 bit
    ## TOTAL         = 31 bit
    #def loop_stack(self, input_state = None, memory_name = None, mem_offset = None, reg_addr1 = None, jump_addr = None):
    #    cyc = input_state['cyc']
    #    reg = input_state['reg']
    #    pc = input_state['pc']

    #    # the stack memory does not incur pipeline stall
    #    cyc += 1
    #    is_empty = self.mem.check_empty(memory_name, mem_offset, self.task_id)
    #    if is_empty:
    #        pc = jump_addr
    #    else:
    #        data = self.mem.pop_stack_mem(memory_name, mem_offset, self.task_id)
    #        # pop stack mem -> read & write zero
    #        reg.write(reg_addr1, data)
    #        op_count["reg_op"] += 1 # reg write
    #        pc += 1
    #    return {'cyc' : cyc, 'reg' : reg, 'pc' : pc}

    ## opcode        = 8 bit
    ## jump_addr     = 8 bit
    ## TOTAL         = 16 bit
    #def stack_end(self, input_state = None, jump_addr = None):
    #    cyc = input_state['cyc']
    #    reg = input_state['reg']
    #    pc = input_state['pc']

    #    cyc += 1
    #    pc = jump_addr

    #    return {'cyc' : cyc, 'reg' : reg, 'pc' : pc}

    ## opcode        = 8 bit
    ## mem_offset    = 14 bit
    ## reg_addr1     = 5 bit
    ## TOTAL         = 27 bit
    #def stack_mem(self, input_state = None, memory_name = None, mem_offset = None, reg_addr1 = None):
    #    cyc = input_state['cyc']
    #    reg = input_state['reg']
    #    pc = input_state['pc']

    #    cyc += 1
    #    if reg_addr1 != None: data = reg.read(reg_addr1)
    #    else: assert(0)
    #    op_count["reg_op"] += 1 # reg read
    #    self.mem.push_stack_mem(memory_name, mem_offset, self.task_id, data)

    #    return {'cyc' : cyc, 'reg' : reg, 'pc' : pc + 1}

    # opcode        = 8 bit
    # func_type     = 2 bit
    # access_type   = 3 bit
    # mem_offset    = 16 bit
    # reg_addr1     = 5 bit
    # reg_addr2     = 5 bit
    # reg_addr3     = 5 bit
    # offset        = 10 bit
    # width         = 5 bit (max: 32)
    # TOTAL         = 64 bit
    def rmw_op(self, input_state = None, func_type = None, access_type = None,
                     src_mem = None, dst_mem = None, mem_offset = None,
                     reg_addr1 = None, reg_addr2 = None, reg_addr3 = None,
                     offset = None, width = None):

        cyc = input_state['cyc']
        reg = input_state['reg']
        pc = input_state['pc']
        stall = 0
        # four different types
        # Implement additional functions
        assert(access_type == 'ctoh' or access_type == 'htoc' or
               access_type == 'htoh' or access_type == 'stos' or
               access_type == 'chtor')

        cyc += 1
        use_corr = False
        use_hist = False
        use_state = False
        # if it involves accessing the correlation mem
        # retrive the address and offset
        if 'c' in access_type:
            use_corr = True
            addr, offset = self.core.loop_ctrl.get_addr(offset, width)
        if 'h' in access_type:
            use_hist = True
            lid = reg.read(reg_addr1)
            if reg_addr2:
                pos = reg.read(reg_addr2)
            else:
                pos = 0
        if 's' in access_type:
            use_state = True
            lid = reg.read(reg_addr1)

        # We support three different types of operations
        corr_stall = 0
        if access_type == 'ctoh':
            write_dat, corr_stall = self.mem.read_corr(src_mem, self.task_id, addr, offset)
        elif access_type == 'htoc':
            assert(self.mem.hist_mem_offset[self.task_id][src_mem] == mem_offset)
            write_dat, _ = self.mem.read_hist_mem(src_mem, mem_offset, self.task_id, lid, pos)
        elif access_type == 'chtor':
            corr_dat, corr_stall = self.mem.read_corr(src_mem, self.task_id, addr, offset)
            assert(self.mem.hist_mem_offset[self.task_id][dst_mem] == mem_offset)
            hist_dat, _ = self.mem.read_hist_mem(dst_mem, mem_offset, self.task_id, lid, pos)
        else:
            write_dat = reg.read(reg_addr3)

        if func_type == 'mac':
            alu_dat = reg.read(reg_addr3)
        else:
            alu_dat = None

        if access_type == 'htoc':
            data, corr_stall = self.mem.accum_corr_mem(func_type, dst_mem, self.task_id, addr, offset, write_dat, alu_dat)
        elif access_type == 'ctoh' or access_type == 'htoh':
            data, _ = self.mem.accum_hist_mem(func_type, dst_mem, mem_offset, self.task_id, lid, pos, write_dat, alu_dat)
            if func_type == 'pop':
                reg.write(reg_addr3, data) # performed within 1 cycle with reg read
        elif access_type == 'stos':
            data, _ = self.mem.accum_state_mem(func_type, dst_mem, mem_offset, self.task_id, lid, write_dat, alu_dat)
            if func_type == 'pop':
                reg.write(reg_addr3, data) # performed within 1 cycle with reg read
        elif access_type == 'chtor':
            assert(func_type == 'mac')
            reg.write(reg_addr3, corr_dat * hist_dat + alu_dat)
        stall += corr_stall

        return {'cyc' : cyc, 'reg' : reg, 'pc' : pc + 1, 'mem' : (use_corr, use_hist, use_state)}

    # opcode        = 8 bit
    # access_type   = 2 bit
    # mem_offset    = 16 bit
    # reg_addr1     = 5 bit
    # reg_addr2     = 5 bit
    # TOTAL         = 36 bit
    def state_mem(self, input_state = None,
                 memory_name = None, mem_offset = None, access_type = None,
                 reg_addr1 = None, reg_addr2 = None, pipelined = False):
        cyc = input_state['cyc']
        reg = input_state['reg']
        pc = input_state['pc']
        stall = 0

        lid = reg.read(reg_addr1)
        op_count["reg_op"] += 1 # reg read

        assert(self.mem.state_mem_offset[self.task_id][memory_name] == mem_offset)
        if access_type == 'read':
            cyc += 1
            data, stall = self.mem.read_state_mem(memory_name, mem_offset, self.task_id, lid)
            reg.write(reg_addr2, data)
        elif access_type == 'write':
            cyc += 1
            if reg_addr2 != None: data = reg.read(reg_addr2)
            else: assert(0)
            stall = self.mem.write_state_mem(memory_name, mem_offset, self.task_id, lid, data)
        else: assert(0)
        cyc += stall

        if pipelined:
            cyc = input_state['cyc']

        return {'cyc' : cyc, 'reg' : reg, 'pc' : pc + 1, 'mem' : (False, False, True), 'pipelined' : pipelined}

    # opcode        = 8 bit
    # access_type   = 2 bit
    # mem_offset    = 16 bit
    # reg_addr1     = 5 bit
    # reg_addr2     = 5 bit
    # reg_addr3     = 5 bit
    # TOTAL         = 41 bit
    def hist_mem(self, input_state = None,
                 memory_name = None, mem_offset = None, access_type = None,
                 reg_addr1 = None, reg_addr2 = None, reg_addr3 = None, pipelined = False):
        cyc = input_state['cyc']
        reg = input_state['reg']
        pc = input_state['pc']
        stall = 0

        lid = reg.read(reg_addr1)
        if reg_addr2:
            pos = reg.read(reg_addr2)
        else:
            pos = 0
        op_count["reg_op"] += 1 # reg read

        assert(self.mem.hist_mem_offset[self.task_id][memory_name] == mem_offset)
        if access_type == 'read':
            cyc += 1
            data, stall = self.mem.read_hist_mem(memory_name, mem_offset, self.task_id, lid, pos)
            reg.write(reg_addr3, data)
        elif access_type == 'write':
            cyc += 1
            if reg_addr3 != None: data = reg.read(reg_addr3)
            else: assert(0)
            stall = self.mem.write_hist_mem(memory_name, mem_offset, self.task_id, lid, pos, data)
        else: assert(0)
        cyc += stall

        if pipelined:
            cyc = input_state['cyc']

        return {'cyc' : cyc, 'reg' : reg, 'pc' : pc + 1, 'mem' : (False, True, False), 'pipelined' : pipelined}

    # opcode        = 8 bit
    # access_type   = 2 bit
    # reg_addr1     = 5 bit
    # offset        = 10 bit
    # width         = 5 bit (max: 32)
    # TOTAL         = 30 bit
    def corr_mem(self, input_state = None, memory_name = None, access_type = None, \
                       reg_addr1 = None, offset = None, width = None, pipelined = False):
        cyc = input_state['cyc']
        reg = input_state['reg']
        pc = input_state['pc']

        # retrieve the address
        addr, offset = self.core.loop_ctrl.get_addr(offset, width)
        op_count["reg_op"] += 1
        cyc += 1

        if access_type == 'read':
            corr_dat, stall = self.mem.read_corr(memory_name, self.task_id, addr, offset)
            reg.write(reg_addr1, corr_dat)
        elif access_type == 'write':
            if reg_addr1 != None: data = reg.read(reg_addr1)
            else: assert(0)
            stall = self.mem.write_corr(memory_name, self.task_id, addr, offset, data)
        else: assert(0)
        cyc += stall

        if pipelined:
            cyc = input_state['cyc']

        return {'cyc' : cyc, 'reg' : reg, 'pc' : pc + 1, 'mem' : (True, False, False), 'pipelined' : pipelined}

    # v1: with imm
    # opcode        = 8 bit
    # func_type     = 4 bit
    # reg_addr1     = 5 bit
    # reg_addr2     = 5 bit
    # immediate     = 32 bit
    # TOTAL         = 54 bit
    def alui_comp(self, input_state = None, func_type = None,
                  reg_addr1 = None, immediate = None, reg_addr2 = None):
        cyc = input_state['cyc']
        reg = input_state['reg']
        pc = input_state['pc']

        input1 = reg.read(reg_addr1) # states
        op_count["reg_op"] += 1 # register read

        if func_type == 'add':
            cyc += 1
            output = input1 + immediate
            op_count["alu_mac"] += 1
        elif func_type == 'sub':
            cyc += 1
            output = input1 - immediate
            op_count["alu_mac"] += 1
        elif func_type == 'subneg':
            cyc += 1
            output = immediate - input1
            #output = -(input1 - immediate)
            op_count["alu_mac"] += 1
        elif func_type == 'mul':
            cyc += 1
            output = input1 * immediate
            op_count["alu_mac"] += 1
        elif func_type == 'div':
            cyc += 12
            output = input1 / immediate
            op_count["alu_div"] += 1
            op_count["reg_op"] += 1 # reg write
        elif func_type == 'mod':
            cyc += 1
            output = input1 % immediate
            op_count["alu_mod"] += 1
        elif func_type == 'sqrt':
            cyc += 6
            output = input1 ** 0.5
            op_count["alu_sqrt"] += 1
            op_count["reg_op"] += 1 # reg write
        elif func_type == 'eq':
            cyc += 1
            output = (input1 == immediate)
            op_count["alu_logical"] += 1
        elif func_type == 'neq':
            cyc += 1
            output = (input1 != immediate)
            op_count["alu_logical"] += 1
        elif func_type == 'gt':
            cyc += 1
            output = (input1 > immediate)
            op_count["alu_logical"] += 1
        elif func_type == 'ge':
            cyc += 1
            output = (input1 >= immediate)
            op_count["alu_logical"] += 1
        elif func_type == 'lt':
            cyc += 1
            output = (input1 < immediate)
            op_count["alu_logical"] += 1
        elif func_type == 'le':
            cyc += 1
            output = (input1 <= immediate)
            op_count["alu_logical"] += 1
        else: assert(0)
        reg.write(reg_addr2, output) # reg write performed within 1 cycle in most cases... except div/sqrt

        return {'cyc' : cyc, 'reg' : reg, 'pc' : pc + 1}

    # v1: with imm
    # opcode        = 8 bit
    # func_type     = 4 bit
    # reg_addr1     = 5 bit
    # reg_addr2     = 5 bit
    # reg_addr3     = 5 bit
    # TOTAL         = 27 bit
    def alur_comp(self, input_state = None, func_type = None, reg_addr1 = None, reg_addr2 = None, reg_addr3 = None, reg_addr4 = None):
        cyc = input_state['cyc']
        reg = input_state['reg']
        pc = input_state['pc']

        input1 = reg.read(reg_addr1) # operand1
        if func_type != 'neg':
            input2 = reg.read(reg_addr2) # operand2
        if func_type == 'mac':
            input3 = reg.read(reg_addr3) # operand3
        op_count["reg_op"] += 1 # read 3 operand at once

        if func_type == 'neg':
            cyc += 1
            output = -input1
            op_count["alu_mac"] += 1
        elif func_type == 'add':
            cyc += 1
            output = input1 + input2
            op_count["alu_mac"] += 1
        elif func_type == 'sub':
            cyc += 1
            output = input1 - input2
            op_count["alu_mac"] += 1
        elif func_type == 'mul':
            cyc += 1
            output = input1 * input2
            op_count["alu_mac"] += 1
        elif func_type == 'div':
            cyc += 12
            output = input1 / input2
            op_count["alu_div"] += 1
            op_count["reg_op"] += 1 # reg write
        elif func_type == 'mac':
            cyc += 1
            output = input1 * input2 + input3
            op_count["alu_mac"] += 1
        elif func_type == 'eq':
            cyc += 1
            output = (input1 == input2)
            op_count["alu_logical"] += 1
        elif func_type == 'neq':
            cyc += 1
            output = (input1 != input2)
            op_count["alu_logical"] += 1
        elif func_type == 'gt':
            cyc += 1
            output = (input1 > input2)
            op_count["alu_logical"] += 1
        elif func_type == 'ge':
            cyc += 1
            output = (input1 >= input2)
            op_count["alu_logical"] += 1
        elif func_type == 'lt':
            cyc += 1
            output = (input1 < input2)
            op_count["alu_logical"] += 1
        elif func_type == 'le':
            cyc += 1
            output = (input1 <= input2)
            op_count["alu_logical"] += 1
        # elif func_type == 'mod':
        #     cyc += 1
        #     output = (input1 % input2)
        #     op_count["alu_mod"] += 1
        else: assert(0)

        reg.write(reg_addr4, output) # reg write performed within 1 cycle with reg read in most of the cases .. except div
        return {'cyc' : cyc, 'reg' : reg, 'pc' : pc + 1}

    # opcode        = 8 bit
    # reg_addr1     = 5 bit
    # reg_addr2     = 5 bit
    # immediate     = 32 bit
    # TOTAL         = 50 bit
    def set_register(self, input_state = None, reg_addr1 = None, reg_addr2 = None, immediate = None):
        cyc = input_state['cyc']
        reg = input_state['reg']
        pc = input_state['pc']

        cyc += 1
        if reg_addr2 != None:
            input1 = reg.read(reg_addr2)
        else:
            input1 = immediate
        reg.write(reg_addr1, input1)
        op_count["reg_op"] += 1
        return {'cyc' : cyc, 'reg' : reg, 'pc' : pc + 1}

    def event_trigger_imm(self, input_state = None, reg_addr1 = None, reg_addr2 = None,
                          packet_type = None, reg_addr3 = None,
                          immediate = None, func_type = None, pipelined = False):
        cyc = input_state['cyc']
        reg = input_state['reg']
        pc = input_state['pc']

        cyc += 1
        # conditionally send packet
        input1 = reg.read(reg_addr3)
        op_count["reg_op"] += 1

        # alu_logical op
        op_count["alu_logical"] += 1
        if func_type == 'eq':
            cond = (input1 == immediate)
        elif func_type == 'neq':
            cond = (input1 != immediate)
        elif func_type == 'gt':
            cond = (input1 > immediate)
        elif func_type == 'ge':
            cond = (input1 >= immediate)
        elif func_type == 'lt':
            cond = (input1 < immediate)
        elif func_type == 'le':
            cond = (input1 <= immediate)
        else:
            assert(0)

        if cond:
            # reg read in 1 cycle
            lid = reg.read(reg_addr1)
            data = reg.read(reg_addr2)
            if type(data) is list:
                assert(len(data) == 1)
                data = data[0]

            packet = {'mode' : 'packet',
                      'task_id' : self.task_id,
                      'dst_x' : reg.read(2),
                      'dst_y' : reg.read(3),
                      'lid' : lid,
                      'data' : data,
                      'event_type' : self.event_type,
                      'type' : packet_type}
            #heappush(self.router.out_event_buf, Event({'cyc': cyc, 'data': packet}))
            self.router.out_event_buf.append(Event({'cyc': cyc, 'data': packet}))
            op_count["out_event_buf_push"] += 1

        if pipelined:
            cyc = input_state['cyc']

        return {'cyc' : cyc, 'reg' : reg, 'pc' : pc + 1, 'pipelined' : pipelined}

    def event_trigger_reg(self, input_state = None, reg_addr1 = None, reg_addr2 = None,
                          packet_type = None, reg_addr3 = None, reg_addr4 = None,
                          func_type = 'always', pipelined = False):
        cyc = input_state['cyc']
        reg = input_state['reg']
        pc = input_state['pc']

        cyc += 1
        # conditionally send packet
        if func_type == 'always':
            cond = True
        else:
            op_count["reg_op"] += 1
            input1 = reg.read(reg_addr3)
            input2 = reg.read(reg_addr4)

            op_count["alu_logical"] += 1
            if func_type == 'eq':
                cond = (input1 == input2)
            elif func_type == 'neq':
                cond = (input1 != input2)
            elif func_type == 'gt':
                cond = (input1 > input2)
            elif func_type == 'ge':
                cond = (input1 >= input2)
            elif func_type == 'lt':
                cond = (input1 < input2)
            elif func_type == 'le':
                cond = (input1 <= input2)
            else:
                assert(0)

        if cond:
            cyc += 1 # stall for reg read port
            op_count["reg_op"] += 1
            lid = reg.read(reg_addr1)
            data = reg.read(reg_addr2)
            if type(data) is list:
                assert(len(data) == 1)
                data = data[0]

            packet = {'mode' : 'packet',
                      'task_id' : self.task_id,
                      'dst_x' : reg.read(2),
                      'dst_y' : reg.read(3),
                      'lid' : lid,
                      'data' : data,
                      'event_type' : self.event_type,
                      'type' : packet_type}
            #heappush(self.router.out_event_buf, Event({'cyc': cyc, 'data': packet}))
            self.router.out_event_buf.append(Event({'cyc': cyc, 'data': packet}))
            op_count["out_event_buf_push"] += 1

        if pipelined:
            cyc = input_state['cyc']

        return {'cyc' : cyc, 'reg' : reg, 'pc' : pc + 1, 'pipelined' : pipelined}

    def init_bci_send(self, task_id, latency_budget):

        event_data = {'src_pid' : None,
                      'data' : latency_budget,
                      'src_x' : None,
                      'src_y' : None}


        bci_send = self.router.packet_to_event[task_id]['commit']['synchronize'][-1]
        self.router.in_event_buf.append(task_id, bci_send, event_data)

    def increment_pos(self, input_state = None):
        cyc = input_state['cyc']
        reg = input_state['reg']
        pc = input_state['pc']

        cyc += 1
        self.mem.increment_pos(self.task_id)

        return {'cyc' : cyc, 'reg' : reg, 'pc' : pc + 1}

    def debug_func(self, input_state = None, module = None, func_type = None, reg_addr1 = None, reg_addr2 = None, reg_addr3 = None, str_debug = ''):
        cyc = input_state['cyc']
        reg = input_state['reg']
        pc = input_state['pc']

        input1 = reg.read(reg_addr1) # states / spiked / traces
        if reg_addr2 != None:
            input2 = reg.read(reg_addr2) # waccum / traces / pid
        else:
            input2 =  None
        if reg_addr3 != None:
            input3 = reg.read(reg_addr3) # lid / lid / None
        else:
            input3 = None
        if module:
            func_type(module, input1, input2, input3, str_debug, self.task_id)
        else:
            func_type(input1, input2, input3, str_debug, self.task_id)

        return {'cyc' : cyc, 'reg' : reg, 'pc' : pc + 1}

    def probe(self, input_state = None, packet_type = 'probe', reg_addr1 = None, reg_addr2 = None, reg_addr3 = None):
        cyc = input_state['cyc']
        reg = input_state['reg']
        pc = input_state['pc']

        cyc += 1
        lid = reg.read(reg_addr1)
        data = reg.read(reg_addr2)
        #timestep = reg.read(reg_addr3)

        # send packet
        packet = {'mode' : 'probe',
                  'task_id' : self.task_id,
                  'dst_x' : reg.read(2),
                  'dst_y' : reg.read(3),
                  'lid' : lid,
                  'data' : data,
                  'event_type' : self.event_type,
                  'type' : packet_type}

        self.router.out_event_buf.append(Event({'cyc': cyc, 'data': packet}))
        op_count["out_event_buf_push"] += 1

        return {'cyc' : cyc, 'reg' : reg, 'pc' : pc + 1}


