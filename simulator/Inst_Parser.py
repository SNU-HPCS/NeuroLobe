import os
import re
#import Task
# import Memory
from pathlib import Path
from itertools import chain
import GlobalVars as GV
import copy
import Task



def parse_inst_file(file_name, gtask_id, core_id):
    # Parse memory-like string
    print(file_name)
    def parse_mem(argument):
        memory_type = argument.split('[')[0]
        mem_addr_list = argument.split('[')[1].split(']')[0].split(',')

        memory_name = mem_addr_list[0]
        addr_list = mem_addr_list[1:]
        
        return memory_type, memory_name, addr_list
        
    # Parse data (check register or immediate)
    def parse_data(data):
        # Three types exist
        # 1) variable: target of register allocation (data => variable name)
        # 2) immedaite: the constant immediate value (data => immediate value)
        # 3) fixed: the variable whose register address is fixed (timestep) (data => register address)

        # Remove the spacing
        data = data.replace(' ','')
        
        const_name= re.search('\[(.*)\]', data)
        if const_name:
            const_name = const_name.group(1)
            const_name = const_name.replace("'", "")
            const_name = const_name.replace('"', '')
            if '-' in const_name:
                const_name = const_name.replace('-', '')
                const_value = -Task.get_task(gtask_id).task_const[const_name]
            else:
                const_value = Task.get_task(gtask_id).task_const[const_name]
            return {'type' : 'immediate', 'data' : const_value}
        elif 'imm' in data:
            try: float(data.split('imm')[1])
            except ValueError:
                print('The variable containing imm should not be used')
            return {'type' : 'immediate', 'data' : data.split('imm')[1]}
        elif 'timestep' == data:
            ltask_id = GV.gtask_to_ltask[gtask_id][core_id]
            reg_addr = ltask_id + GV.reg_general_region_size
            return {'type' : 'fixed', 'data' : reg_addr}
        elif 'event_pid' == data:
            # The register address 0 contains the event pid
            return {'type' : 'fixed', 'data' : 0}
        elif 'event_data' == data:
            # The register address 1 contains the event data
            return {'type' : 'fixed', 'data' : 1}
        elif 'iter_count' == data:
            # The register address 9 contains the event data
            return {'type' : 'fixed', 'data' : 9}
        else:
            return {'type' : 'variable', 'data' : data}

    def parse_operator(data):
        # remove spacing
        data = data.replace(' ', '')

        # Check for SQRT operator first
        sqrt_reg = re.search('sqrt\[(.*)\]', data)
        if sqrt_reg:
            sqrt_reg = sqrt_reg.group(1)
            return "'sqrt'", [sqrt_reg]
        # Check for MAC operator second
        if '*' in data and '+' in data:
            data = data.split('*')
            if '+' in data[0]:
                reg1 = data[1]
                reg2 = data[0].split('+')[1]
                reg3 = data[0].split('+')[0]
            else:
                reg1 = data[0]
                reg2 = data[1].split('+')[0]
                reg3 = data[1].split('+')[1]
            return "'mac'", [reg1, reg2, reg3]
                

        operator_list = [('==', "'eq'"),
                         ('!=', "'neq'"),
                         ('>=', "'ge'"),
                         ('>', "'gt'"),
                         ('<=', "'le'"),
                         ('<', "'lt'"),
                         ('+', "'add'"),
                         ('-', "'sub'"),
                         ('*', "'mul'"),
                         ('%', "'mod'"),
                         ('/', "'div'")]
        for operator in operator_list:
            if operator[0] in data:
                return operator[1], data.split(operator[0])
        print(data)
        assert(0)

    #def parse_cond(argument_part):

    # Enable nested jump instructions stack (or if)

    # Check code valid (To enable nested IF/ENDIF)
    valid = [True]

    #################################
    # check for pipeline instructions
    # This is for temporary debugging purpose
    in_pipeline = 0
    first_rmw = True
    in_if = False
    pipeline_inst = []
    #################################


    lines = []
    if os.path.isfile(file_name):
        f = open(file_name, 'r')
        lines = f.readlines()
    #else:
    #    print(file_name)
    #    raise("File Does Not Exist")
    
    # Relocate instructions for pipeline
    relocated_lines = []
    for line in lines:
        if not line.split('#')[0].strip(): continue
        parsed_line = line.split('#')[0].rstrip().lstrip()

        # Check if the code starts with IF
        # if so, check the condition and append to the valid list
        is_inst = True
        if 'IF' == parsed_line.split(' ')[0]:
            is_inst = False
            condition = parsed_line.split(' ')[1]
            if valid[-1]:
                condition = condition.replace(' ','')
                if 'PARTIAL' == condition:
                    condition = GV.sim_params['use_partial'][gtask_id]
                elif 'NO_PARTIAL' == condition:
                    condition = (not GV.sim_params['use_partial'][gtask_id])
                elif 'DEBUG' == condition:
                    condition = GV.DEBUG
                elif 'NO_DEBUG' == condition:
                    condition = (not GV.DEBUG)
                else: assert(0)
                valid.append(condition)
            else:
                valid.append(False)
        elif 'ENDIF' == parsed_line.split(' ')[0]:
            is_inst = False
            valid.pop()

        if valid[-1] and is_inst:
            # Retrieve the opcode part
            opcode = parsed_line.split('(')[0].replace(' ', '').split('=')
            assert(len(opcode) < 3)
            # We set the dst register with =
            dst_reg = (opcode[0] + '=') if len(opcode) == 2 else ''
            opcode = opcode[-1]
            ####################################
            argument_part = parsed_line.split('(')[1].split(')')[0]

            # Function for getting the memory type

            if opcode == 'pipelined_loop_corr':
                opcode = 'loop_corr'
                argument_part += ', pipelined = True'
                if in_pipeline == 0:
                    pipeline_inst.append('corr_end(pipelined = True)')
                    in_pipeline += 1
                    new_line = dst_reg + opcode+ '('+argument_part+')'
                    pipeline_inst.append(new_line)
                else:
                    in_pipeline += 1
                    if not first_rmw:
                        for inst in pipeline_inst:
                            relocated_lines.append(inst)
                    new_line = dst_reg + opcode+ '('+argument_part+')'
            elif in_pipeline:
                if opcode == 'memory':
                    # parsing the memory_type
                    memory_type = argument_part.replace(' ', '').split('mem_type=')[1].split('[')[0]
                    if memory_type == 'corr_mem':
                        argument_part += ', pipelined = True'
                        new_line = dst_reg + opcode+ '('+argument_part+')'
                        pipeline_inst.append(new_line)
                    elif memory_type == 'state_mem':
                        argument_part += ', pipelined = True'
                        new_line = dst_reg + opcode+ '('+argument_part+')'
                    else:
                        assert(0)
                elif opcode == 'corr_end':
                    argument_part = 'pipelined = True'
                    new_line = dst_reg + opcode+ '('+argument_part+')'
                    in_pipeline -= 1
                    pipeline_inst = []
                elif opcode == 'rmw_op':
                    if first_rmw or in_pipeline >= 2:
                        if not in_if:
                            first_rmw = False
                        new_line = dst_reg + opcode+ '('+argument_part+')' 
                    else:
                        for inst in pipeline_inst:
                            relocated_lines.append(inst)
                        new_line = dst_reg + opcode+ '('+argument_part+')' 
                elif opcode == 'event_trigger':
                    if first_rmw:
                        new_line = dst_reg + opcode+ '('+argument_part+')'
                    else:
                        argument_part += ', pipelined = True'
                        new_line = dst_reg + opcode+ '('+argument_part+')'
                else:
                    print(opcode)
                    assert(0)
            else:
                new_line = dst_reg + opcode+ '('+argument_part+')'

            relocated_lines.append(new_line)
        #else:
        #    relocated_lines.append(parsed_line)

    # Set the list of read/write variables, opcode, argument
    inst_format = {'write_var' : [], 'read_var' : [], 'opcode' : None, 'argument_list' : {}}
    inst_list = []
    jump_stack = []
    loop_region = []


    # Set the read and write register
    read_arg_name = ['operand0', 'operand1', 'operand2', \
                     'reg_addr0', 'reg_addr1', 'reg_addr2', 'reg_addr3', \
                     'reg_data', 'reg_iter']
    write_arg_name = ['reg_write']
    # Parse and convert the API to ISA (w/o reg allocation)
    for lid in range(len(relocated_lines)):
        line = relocated_lines[lid]
        line = line.split('#')[0].rstrip().lstrip()

        # Deep copy the instruction format
        inst = copy.deepcopy(inst_format)

        ##################################################
        # 1) Parse the destination register and the opcode
        opcode = line.split('(')[0].replace(' ', '').split('=')
        assert(len(opcode) < 3)
        dst_reg = opcode[0] if len(opcode) == 2 else None
        if dst_reg: dst_reg = parse_data(dst_reg)
        opcode = opcode[-1]

        ##################################################
        # 2) Parse the argument list
        # Do not split "," in mem[a, b, c,]
        argument_list = re.split(',(?![^[]*\])', line.split('(')[1].split(')')[0])
        parsed_argument_list = {}

        # 1) Parse the argument
        for argument in argument_list:
            argument = argument.replace(' ', '')
            # Do not split "=" in ==, >=, <=, ...
            argument = re.split('(?<!\=|\<|\>|\!)=(?!\=)', argument)
            # Convert the string-like data to the string

            if len(argument) == 1:
                parsed_argument_list['argument'] = argument[0]
            else:
                # We should transform the data
                if argument[0] in read_arg_name or argument[0] in write_arg_name:
                    argument[1] = parse_data(argument[1])
                if argument[0] == 'immediate':
                    data = parse_data(argument[1])
                    assert(data['type'] == 'immediate')
                    argument[1] = data['data']
                # We should transform the constant into the register (Register Allocation)
                if argument[1] == 'timestep':
                    ltask_id = GV.gtask_to_ltask[gtask_id][core_id]
                    parsed_argument_list[argument[0]] = ltask_id + GV.reg_general_region_size
                else:
                    parsed_argument_list[argument[0]] = argument[1]
        argument_list = parsed_argument_list

        # 2) Set the write register using the dst_reg
        if dst_reg: argument_list['reg_write'] = dst_reg

        # 3) Parse the memory-related parameters, register addresses, and set the mem_offset
        # 3-1) Parse the memory APIs
        if opcode == 'memory':
            mem_type, memory_name, addr_list = parse_mem(argument_list['mem_type'])
            opcode = mem_type
            del argument_list['mem_type']

            argument_list['memory_name'] = memory_name
            if (mem_type == 'hist_mem' or mem_type == 'state_mem'):
                for ind in range(len(addr_list)):
                    addr = addr_list[ind]
                    argument_list['reg_addr{}'.format(ind)] = parse_data(addr)
            elif (mem_type == 'corr_mem'):
                assert(len(addr_list) == 1)
                argument_list['data_type'] = addr_list[0]
            else:
                print(mem_type)
                raise("Not valid memory type")

        # 3-2) Parse the memory in rmw APIs
        elif opcode == 'rmw_op':
            target_mem = ['corr_mem', 'local_mem']

            for mem_type in target_mem:
                if mem_type in argument_list:
                    argument = argument_list[mem_type]
                    _, mem_name, addr_list = parse_mem(argument)
                    argument_list[mem_type] = mem_name
                    # The corr mem does not need an explicit address
                    if mem_type == 'local_mem':
                        for ind in range(len(addr_list)):
                            addr = addr_list[ind]
                            argument_list['reg_addr{}'.format(ind)] = parse_data(addr)
                    elif mem_type == 'corr_mem':
                        assert(len(addr_list) == 1)
                        argument_list['data_type'] = addr_list[0]
                    else:
                        raise("Not valid memory type")

        # 3-3) Set the memory offset
        memory_name = None
        if 'memory_name' in argument_list or 'local_mem' in argument_list or 'corr_mem' in argument_list:
            # Get the memory name
            ltask_id = GV.gtask_to_ltask[gtask_id][core_id]
            memory_name = argument_list['memory_name'] if 'memory_name' in argument_list \
                                                       else argument_list['local_mem']
            memory_name = memory_name.replace("'", "")
            memory_name = memory_name.replace('"', '')

            if opcode == 'loop_corr':
                mem_offset = GV.corr_forward_offset[core_id][ltask_id][memory_name]
            elif opcode == 'state_mem':
                mem_offset = GV.state_mem_offset[core_id][ltask_id][memory_name]
            elif opcode == 'hist_mem':
                mem_offset = GV.hist_mem_offset[core_id][ltask_id][memory_name]
            elif opcode == 'rmw_op':
                access_type = argument_list['access_type']
                access_type = access_type.replace("'", "")
                access_type = access_type.replace('"', '')
                if 's' in access_type:
                    mem_offset = GV.state_mem_offset[core_id][ltask_id][memory_name]
                elif 'h' in access_type:
                    mem_offset = GV.hist_mem_offset[core_id][ltask_id][memory_name]
                else:
                    assert(0)

                if access_type == 'chtor':
                    argument_list['reg_data'] = argument_list['reg_write']

            if opcode != 'corr_mem':
                argument_list['mem_offset'] = mem_offset

            #

        # 4) Pase the alu / if / event_trigger opcode
        if opcode == 'alu_comp' or opcode == 'if_begin' or opcode == 'event_trigger':

            if opcode == 'event_trigger' and not 'condition' in argument_list:
                opcode = 'event_trigger_reg'
            else:
                if 'condition' in argument_list:
                    argument = argument_list['condition']
                    argument = argument.replace("'", "")
                    argument = argument.replace('"', '')
                    del argument_list['condition']
                else:
                    if not 'argument' in argument_list:
                        print(line)
                    argument = argument_list['argument']
                    del argument_list['argument']

                # Parse the operator
                parsed_op = parse_operator(argument)
                operator_name = parsed_op[0]
                argument_list['func_type'] = operator_name

                # Iterate over the operator list to set the opcode
                has_imm = False
                imm_index = -1
                for i in range(len(parsed_op[1])):
                    raw_data = parse_data(parsed_op[1][i])
                    argument_list['operand{}'.format(i)] = raw_data

                    if raw_data['type'] == 'variable' or raw_data['type'] == 'fixed':
                        pass
                    elif raw_data['type'] == 'immediate':
                        has_imm = True
                        imm_index = i
                    else:
                        assert(0)
                        

                # Check immediate for the second operand
                # if either is immediate
                if operator_name == "'sqrt'":
                    opcode = 'aluri_comp'
                elif not has_imm:
                    if opcode == 'alu_comp':
                        opcode = 'alurr_comp'
                    elif opcode == 'if_begin':
                        opcode = 'if_reg_begin'
                    else:
                        opcode = 'event_trigger_reg'
                else:
                    if opcode == 'alu_comp':
                        if imm_index == 0: opcode = 'aluir_comp'
                        if imm_index == 1: opcode = 'aluri_comp'
                    elif opcode == 'if_begin':
                        opcode = 'if_imm_begin'
                    else:
                        opcode = 'event_trigger_imm'

        # 5) Set the loop_offset+immediate / offset & width /
        # 5-1) Set the loop_corr
        if opcode == 'loop_corr':
            ## Set the offset to be incremented @ each iteration
            # assert the memory name exists!!
            memory_name = argument_list['memory_name'] 
            memory_name = memory_name.replace("'", "")
            memory_name = memory_name.replace('"', '')
            assert(memory_name in GV.loop_info[gtask_id])
            loop_offset = GV.loop_info[gtask_id][memory_name]['loop']
            loop_type = argument_list['loop_type'] if 'loop_type' in argument_list else 'outer'
            loop_type = loop_type.replace("'", "")
            loop_type = loop_type.replace('"', '')
            argument_list['loop_offset'] = loop_offset[loop_type]

            # We force a register (9) allocation for the loop
            if not 'reg_write' in argument_list:
                argument_list['reg_write'] = {'type' : 'fixed', 'data' : 9}

        # 5-2) Set the loop_unit (according to the per unit mapping)
        if opcode == 'loop_unit':
            unit_name = argument_list['unit_name']
            unit_name = unit_name.replace("'", "")
            unit_name = unit_name.replace('"', '')
            del argument_list['unit_name']
            if '_to_' in unit_name:
                src_unit_name = unit_name.split('_to_')[0]
                dst_unit_name = unit_name.split('_to_')[1]
                if not src_unit_name in GV.sim_params['unit_type_name'][gtask_id]: print(src_unit_name)
                if not dst_unit_name in GV.sim_params['unit_type_name'][gtask_id]: print(dst_unit_name)
                assert(src_unit_name in GV.sim_params['unit_type_name'][gtask_id])
                assert(dst_unit_name in GV.sim_params['unit_type_name'][gtask_id])

                src_type_id = GV.sim_params['unit_type_name'][gtask_id].index(src_unit_name)
                dst_type_id = GV.sim_params['unit_type_name'][gtask_id].index(dst_unit_name)
                per_core_unit_num = len(GV.pid_to_gid[gtask_id][core_id][src_type_id][dst_type_id])
                argument_list['immediate'] = per_core_unit_num
            else:
                if not unit_name in GV.sim_params['unit_type_name'][gtask_id]:
                    print(unit_name)
                assert(unit_name in GV.sim_params['unit_type_name'][gtask_id])
                type_id = GV.sim_params['unit_type_name'][gtask_id].index(unit_name)
                per_core_unit_num = GV.sim_params['num_unit_per_core'][gtask_id][core_id][type_id]
                argument_list['immediate'] = per_core_unit_num

        # 5-3) Check if the corr_mem exists
        if opcode == 'corr_mem' or 'corr_mem' in argument_list:
            # Retrieve the memory name
            data_type = argument_list['data_type']
            data_type = data_type.replace("'", "")
            data_type = data_type.replace('"', '')
            del argument_list['data_type']

            # assert the memory name exists!!
            memory_name = argument_list['memory_name'] if 'memory_name' in argument_list \
                                                       else argument_list['corr_mem']
            memory_name = memory_name.replace("'", "")
            memory_name = memory_name.replace('"', '')
            if not memory_name in GV.loop_info[gtask_id]:
                print(memory_name)
            assert(memory_name in GV.loop_info[gtask_id])
            total_offset = GV.loop_info[gtask_id][memory_name]['offset']
            total_width = GV.loop_info[gtask_id][memory_name]['width']
            ## Set the offset to be incremented @ each iteration
            argument_list['offset'] = total_offset[data_type]
            #argument_list['width'] = total_width[data_type]

        # 6) Set the jump addresses
        if opcode == 'loop_corr' or opcode == 'loop_unit' or \
           opcode == 'if_imm_begin' or opcode == 'if_reg_begin':
            jump_stack.append(len(inst_list))

        if opcode == 'corr_end' or opcode == 'unit_end':
            target_addr = jump_stack.pop()
            jump_addr = len(inst_list) + 1
            inst_list[target_addr]['argument_list']['jump_addr'] = jump_addr
            reg_write = inst_list[target_addr]['argument_list']['reg_write']

            # Temporary register
            argument_list['reg_iter'] = reg_write
            # We should track the loop region
            loop_region.append((target_addr, len(inst_list)))

        if opcode == 'if_end':
            target_addr = jump_stack.pop()
            jump_addr = len(inst_list)
            inst_list[target_addr]['argument_list']['jump_addr'] = jump_addr

        # "if_end" is the only non-instruction
        if opcode != 'if_end':

            # Iterate over the keys to check if the register is read or write
            for key in argument_list:
                value = argument_list[key]
                if key in read_arg_name:
                    inst['read_var'].append(value)
                if key in write_arg_name:
                    inst['write_var'].append(value)

            inst['opcode'] = opcode
            if 'reg_iter' in argument_list: del argument_list['reg_iter']
            inst['argument_list'] = argument_list

            inst_list.append(inst)

    # Set the list of variables and when they are used
    variable_list = {}
    for lid in range(len(inst_list)):
        read_var_list = inst_list[lid]['read_var']
        write_var_list = inst_list[lid]['write_var']
        
        # Set the code region
        for read_var in read_var_list:
            if read_var['type'] == 'variable':
                var = read_var['data']
                if var in variable_list:
                    if not lid in variable_list[var]['r']:
                        variable_list[var]['r'].append(lid)
                else:
                    variable_list[var] = {'r' : [lid], 'w' : []}
        for write_var in write_var_list:
            if write_var['type'] == 'variable':
                var = write_var['data']
                if var in variable_list:
                    if not lid in variable_list[var]['w']:
                        variable_list[var]['w'].append(lid)
                else:
                    variable_list[var] = {'r' : [], 'w' : [lid]}


    #print("PREV")
    #for var in variable_list:
    #    print(var, variable_list[var])

    # We should remove the redundant writes
    for var in variable_list:
        # Check if write after write exists
        # => if so, check if the 'if_begin' exists in between the WAW
        read_write = variable_list[var]
        write_idx = 0

        # Check code region
        # Read precedes Write
        waw_region = []
        while (write_idx + 1 < len(read_write['w'])):
            prev_pc = read_write['w'][write_idx]
            next_pc = read_write['w'][write_idx + 1]
            is_waw = True
            for read_pc in read_write['r']:
                if prev_pc < read_pc <= next_pc:
                    is_waw = False
                    break

            if is_waw: waw_region.append((prev_pc, next_pc))
            write_idx += 1

        # Iterate over the WAW region to check if an if statement exists
        for prev_pc, next_pc in waw_region:
            true_waw = True
            for pc in range(prev_pc + 1, next_pc):
                opcode = inst_list[pc]['opcode']
                if opcode == 'if_begin':
                    jump_addr = inst_list[pc]['argument_list']['jump_addr']
                    if next_pc < jump_addr:
                        true_waw = False
                        break
            
            # If true waw => merge the writes
            if true_waw: read_write['w'].remove(next_pc)
    
    #print("NEXT")
    #for var in variable_list:
    #    print(var, variable_list[var])


    code_region = {}
    # Greedy register setting
    for var in variable_list:
        read_idx = 0
        write_idx = 1
        # Set the list of used code region
        used_code_region = []

        # If the variable is for loop iteration => It should be preserved until loop end


        read_write = variable_list[var]

        # The RW should exist
        #assert(len(read_write['w']) > 0 and len(read_write['r']) > 0)
        assert(len(read_write['w']) > 0)
        last_pc = read_write['w'][0]

        # Get the target_pc considering the loop
        def insert_loop(last_pc, read_pc):
            # Check if loop exists between [last_pc:read_pc]
            target_pc = read_pc
            for loop_start, loop_end in loop_region:
                if last_pc <= loop_start <= read_pc:
                    target_pc = max(target_pc, loop_end)
            return target_pc

        while True:
            # If the read_idx has reached the last
            # Terminate
            if read_idx == len(read_write['r']):
                # The write-only variables should be taken care...
                for pc in read_write['w']:
                    if not pc in used_code_region:
                        used_code_region.append(pc)
                #assert(write_idx == len(read_write['w']))
                break
            # If we have reached the last write
            # The code region from the start of the write to the last read is allocated
            elif write_idx == len(read_write['w']):
                # Check for loop in between
                target_pc = insert_loop(last_pc, read_write['r'][-1])
                for pc in range(last_pc, target_pc + 1):
                    if not pc in used_code_region:
                        used_code_region.append(pc)
                read_idx = len(read_write['r'])
            # If we have yet reached the last write
            # Check if the 
            else:
                # If the current read is before the next write =>
                # Use [last_pc:read_pc]
                if read_write['r'][read_idx] <= read_write['w'][write_idx]:
                    target_pc = insert_loop(last_pc, read_write['r'][read_idx])
                    for pc in range(last_pc, target_pc + 1):
                        if not pc in used_code_region:
                            used_code_region.append(pc)
                    # Update the last_pc
                    last_pc = target_pc
                    read_idx += 1
                # If the current read is after the next write => update
                # Jump the last_pc
                else:
                    last_pc = read_write['w'][write_idx]
                    write_idx += 1

        code_region[var] = used_code_region
        #print(var, used_code_region)
        # Get arbitrary register

    # Now we should allocate the register
    # Use the greedy method
    variable_to_reg = {}
    for key in code_region.keys(): variable_to_reg[key] = None
    # The list of target lines used by the register
    reg_to_line = [[False for _ in range(len(inst_list))] \
                          for _ in range(GV.reg_free_region)]

    for key in code_region.keys():
        used_code_region = code_region[key]
        for target_reg in range(GV.reg_free_region):
            valid = True
            for line in used_code_region:
                # If the register is already in use => 
                if reg_to_line[target_reg][line]:
                    if target_reg == (GV.reg_free_region - 1):
                        raise('No valid register exist')
                    valid = False
            
            if valid:
                break

        for line in used_code_region:
            reg_to_line[target_reg][line] = True
        variable_to_reg[key] = target_reg + GV.reg_event_size
        #print('REGALLOC', key, variable_to_reg[key])

    for inst in inst_list:
        argument_list = inst['argument_list']
        for key in argument_list:
            if key in read_arg_name or key in write_arg_name:
                value = argument_list[key]
                if value['type'] == 'variable':
                    reg_addr = variable_to_reg[value['data']]
                    argument_list[key] = reg_addr
                else:
                    argument_list[key] = value['data']
                #print('RESULT', key, argument_list[key])


    program = []
    for inst in inst_list:
        opcode = inst['opcode']
        argument_list = inst['argument_list']

        #############################################
        # Append the new instruction
        inst_raw = 'inst_func = self.isa_ctrl.' + opcode
        for key in argument_list:
            value = argument_list[key]
            if not key == 'module' and not key == 'argument':
                inst_raw += ', {} = {}'.format(key, value)
        program.append('Inst({})'.format(inst_raw))

    if(core_id == 0 and gtask_id == 0):
        new_f = open(str(file_name) + '_compile_new', 'w')
        for inst_id in range(len(program)):
            new_f.write(program[inst_id] + "\n")
        #for var in code_region.keys():
        #    new_f.write('CODE REGION : ' + str(code_region[var]) + '\n')
        new_f.close()
    return program

if __name__ == "__main__":
    p = Path('.')
    for filename in (p.glob('*.inst')):
        print(str(filename))
        program = parse_inst_file(filename)
        f = open(str(filename) + '_compile_new', 'w')
        for inst_id in range(len(program)):
            f.write(program[inst_id] + "\n")
        f.close()
