import GlobalVars as GV
import copy
import ISA
import numpy as np
from Core import Core, Register, LoopCtrl
from External import ExternalModule
from ISA import ISAController
from Router import Router
from Router import Synchronizer
from Memory import Memory
from Debug import DebugModule
from Energy import op_count

class Inst:
    def __init__(self, inst_func,  **kwargs):
        self.inst_func = inst_func # instruction function
        self.inst_params = kwargs  # instruction parameters

    def process(self, input_state, task_id, debug):
        assert(self.inst_func)
        if debug:
            print(input_state['pc'], self.inst_func, input_state['reg'].read(24))
        output_state = self.inst_func(input_state = input_state, **self.inst_params)
        if 'pipelined' not in output_state or not output_state['pipelined']:
            op_count['inst_mem'] += 1
            op_count['pc_update'] += 1
        return output_state

# predefined graph for ISA

# class PC:
#     def __init__(self, core):
#         # base and bound
#         self.base = {}
#         self.bound = {}
#         self.inst_list = []

#         # Set modules to call functions in the graph
#         self.core = core
#         self.debug_module = self.core.debug_module
#         self.isa_ctrl = self.core.isa_ctrl
#         self.processingEvent = self.core.processingEvent

#     def register_instruction(self, task_id, event_type, inst_list):
#         base = len(self.inst_list)
#         size = len(inst_list)
#         self.base[(task_id, event_type)] = base
#         self.bound[(task_id, event_type)] = base + size
#         for idx in range(size):
#             inst = eval(inst_list[idx])
#             self.inst_list.append(inst)

#     def process_event(self, input_state, task_id, event_type, debug):
#         # if the instruction is not registered, return
#         if not (task_id, event_type) in self.base: return input_state

#         base = self.base[(task_id, event_type)]
#         bound = self.bound[(task_id, event_type)]
#         inst_list = self.inst_list[base:bound]

#         pc = 0
#         if len(inst_list) > 0:
#             while True:
#                 inst = inst_list[pc]
#                 input_state = inst.process(input_state, task_id, debug)
#                 pc = input_state['pc']
#                 # process memory writeback for memories not used
#                 self.core.mem.memory_writeback(self.isa_ctrl.use_corr, self.isa_ctrl.use_hist)
#                 assert(pc <= len(inst_list))
#                 if pc == len(inst_list): break

#         return input_state

# modify PC module to load archtectural state made by compiler

class PC:
    def __init__(self, core, table_key_list, base_list, bound_list, inst_list):
        # base and bound
        self.base = {}
        self.bound = {}
        self.inst_list = []
        # Set modules to call functions in the graph
        self.core = core
        self.debug_module = self.core.debug_module
        self.isa_ctrl = self.core.isa_ctrl
        self.processingEvent = self.core.processingEvent
        self.inst_list_raw = inst_list

        # load architectural states
        # print(table_key_list)
        for i, (task_id, event_type) in enumerate(table_key_list):
            self.base[(task_id, event_type)] = base_list[i]
            self.bound[(task_id, event_type)] = bound_list[i]

        for i in range(len(inst_list)):
            inst = eval(inst_list[i])
            self.inst_list.append(inst)


    # def register_instruction(self, task_id, event_type, inst_list):
    #     base = len(self.inst_list)
    #     size = len(inst_list)
    #     self.base[(task_id, event_type)] = base
    #     self.bound[(task_id, event_type)] = base + size
    #     for idx in range(size):
    #         inst = eval(inst_list[idx])
    #         self.inst_list.append(inst)

    def process_event(self, input_state, task_id, event_type, debug):
        # if the instruction is not registered, return
        if not (task_id, event_type) in self.base: return input_state

        base = self.base[(task_id, event_type)]
        bound = self.bound[(task_id, event_type)]
        inst_list = self.inst_list[base:bound]

        pc = 0
        if len(inst_list) > 0:
            while True:
                inst = inst_list[pc]
                input_state = inst.process(input_state, task_id, debug)
                pc = input_state['pc']
                # process memory writeback for memories not used
                # self.core.mem.memory_writeback(self.isa_ctrl.use_corr, self.isa_ctrl.use_hist, self.isa_ctrl.use_state)
                assert(pc <= len(inst_list))
                if pc == len(inst_list): break

        return input_state
