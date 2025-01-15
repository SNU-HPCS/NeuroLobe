import GlobalVars as GV
import os
import Task
# import Memory

def parse_inst_file(file_name, gtask_id, core_id):
    #
    jump_stack = []

    # Check code valid
    valid = [True]
    # original list of instructions
    original_inst = []
    # parsed list of instructions
    inst_list = []
# check for pipeline instructions
    in_pipeline = 0
    first_rmw = True
    in_if = False
    pipeline_inst = []

    lines = []
    relocated_lines = []

    if os.path.isfile(file_name):
        f = open(file_name, 'r')
        lines = f.readlines()
    
    # Relocate instructions for pipeline
    for line in lines:
        if not line.split('#')[0].strip():
            continue
        parsed_line = line.split('#')[0].rstrip().lstrip()

        is_inst = True
        if 'IF' == parsed_line.split(' ')[0]:
            is_inst = False
            in_if = True
        elif 'ENDIF' == parsed_line.split(' ')[0]:
            is_inst = False
            in_if = False

        if is_inst:
            opcode_part = parsed_line.split('(')[0]
            argument_part = parsed_line.split('(')[1].split(')')[0]
            if opcode_part == 'pipelined_loop_corr':
                opcode_part = 'loop_corr'
                argument_part += ', pipelined = True'
                if in_pipeline == 0:
                    pipeline_inst.append('corr_end(pipelined = True)')
                    in_pipeline += 1
                    new_line = opcode_part+ '('+argument_part+')'
                    pipeline_inst.append(new_line)
                else:
                    in_pipeline += 1
                    if not first_rmw:
                        for inst in pipeline_inst:
                            relocated_lines.append(inst)
                    new_line = opcode_part+ '('+argument_part+')'
            elif in_pipeline:
                if opcode_part == 'corr_mem':
                    argument_part += ', pipelined = True'
                    new_line = opcode_part+ '('+argument_part+')'
                    pipeline_inst.append(new_line)
                elif opcode_part == 'corr_end':
                    argument_part = 'pipelined = True'
                    new_line = opcode_part+ '('+argument_part+')'
                    in_pipeline -= 1
                    pipeline_inst = []
                elif opcode_part == 'rmw_op':
                    if first_rmw or in_pipeline >= 2:
                        if not in_if:
                            first_rmw = False
                        new_line = opcode_part+ '('+argument_part+')' 
                    else:
                        for inst in pipeline_inst:
                            relocated_lines.append(inst)
                        new_line = opcode_part+ '('+argument_part+')' 
                elif opcode_part == 'event_trigger':
                    if first_rmw:
                        new_line = opcode_part+ '('+argument_part+')'
                    else:
                        argument_part += ', pipelined = True'
                        new_line = opcode_part+ '('+argument_part+')'
                elif opcode_part == 'state_mem':
                    argument_part += ', pipelined = True'
                    new_line = opcode_part+ '('+argument_part+')'
                else:
                    print(opcode_part)
                    assert(0)
            else:
                new_line = opcode_part+ '('+argument_part+')'

            relocated_lines.append(new_line)
        else:
            relocated_lines.append(parsed_line)

    for line in relocated_lines:
        # Check if comment and continue
        if not line.split('#')[0].strip():
            continue
        parsed_line = line.split('#')[0].rstrip().lstrip()

        is_inst = True
        # Check if the code starts with IF
        # if so, check the condition and append to the valid list
        if 'IF' == parsed_line.split(' ')[0]:
            is_inst = False
            condition = parsed_line.split(' ')[1]
            if valid[-1]:
                valid.append(eval(condition))
            else:
                valid.append(False)
        elif 'ENDIF' == parsed_line.split(' ')[0]:
            is_inst = False
            valid.pop()

        if valid[-1] and is_inst:
            opcode_part = parsed_line.split('(')[0]
            argument_part = parsed_line.split('(')[1].split(')')[0]
            argument_part = argument_part.split(',')
            # We should remove module from the instruction

            # replace const_name with immediate
            for i, argument in enumerate(argument_part):
                if 'const_name' in argument:
                    const_name = argument.split('=')[1].lstrip()
                    const_name = const_name.replace("'", "")
                    const_name = const_name.replace('"', '')
                    if '-' in const_name:
                        const_name = const_name.replace('-', '')
                        const = -Task.get_task(gtask_id).task_const[const_name]
                    else:
                        const = Task.get_task(gtask_id).task_const[const_name]
                    argument_part[i] = 'immediate = {}'.format(const)

            # replace timestep to physical addr
            for i, argument in enumerate(argument_part):
                if 'timestep' in argument:
                    ltask_id = GV.gtask_to_ltask[gtask_id][core_id]
                    reg_addr = ltask_id + GV.reg_general_region_size
                    argument_part[i] = '{} = {}'.format(argument.split('=')[0], reg_addr)

            ############ CORR LOOP PARSE BEGIN ############
            # check if an immediate argument exists:
            if opcode_part == 'loop_corr':
                jump_stack.append(len(inst_list))
                # Check if inner/outer
                func_type = None
                for i, argument in enumerate(argument_part):
                    func_type = 'outer'
                    if 'func_type' in argument:
                        func_type = argument.split('=')[1].lstrip()
                        func_type = func_type.replace("'", "")
                        func_type = func_type.replace('"', '')
                        break
                # Retrieve the memory name
                memory_name = None
                for i, argument in enumerate(argument_part):
                    if 'memory_name' in argument:
                        memory_name = argument.split('=')[1].lstrip()
                        memory_name = memory_name.replace("'", "")
                        memory_name = memory_name.replace('"', '')

                ltask_id = GV.gtask_to_ltask[gtask_id][core_id]
                # mem_offset = mem.corr_forward_offset[ltask_id][memory_name]
                mem_offset = GV.corr_forward_offset[core_id][ltask_id][memory_name]
                argument_part.append('mem_offset = {}'.format(mem_offset))
                # assert the memory name exists!!
                assert(memory_name in GV.loop_info[gtask_id])
                loop_offset = GV.loop_info[gtask_id][memory_name]['loop']
                # Set the offset to be incremented @ each iteration
                argument_part.append('loop_offset = {}'.format(loop_offset[func_type]))

            if opcode_part == 'corr_end':
                # for instruction jumps after the end
                target_addr = jump_stack.pop()
                jump_addr = len(inst_list) + 1
                inst_list[target_addr] = inst_list[target_addr] + ', jump_addr = {}'.format(jump_addr)

            if opcode_part == 'state_mem':
                # Retrieve the memory name
                memory_name = None
                for i, argument in enumerate(argument_part):
                    if 'memory_name' in argument:
                        memory_name = argument.split('=')[1].lstrip()
                        memory_name = memory_name.replace("'", "")
                        memory_name = memory_name.replace('"', '')

                ltask_id = GV.gtask_to_ltask[gtask_id][core_id]
                # mem_offset = mem.state_mem_offset[ltask_id][memory_name]
                mem_offset = GV.state_mem_offset[core_id][ltask_id][memory_name]
                argument_part.append('mem_offset = {}'.format(mem_offset))

            if opcode_part == 'hist_mem':
                # Retrieve the memory name
                memory_name = None
                for i, argument in enumerate(argument_part):
                    if 'memory_name' in argument:
                        memory_name = argument.split('=')[1].lstrip()
                        memory_name = memory_name.replace("'", "")
                        memory_name = memory_name.replace('"', '')

                ltask_id = GV.gtask_to_ltask[gtask_id][core_id]
                # mem_offset = mem.hist_mem_offset[ltask_id][memory_name]
                mem_offset = GV.hist_mem_offset[core_id][ltask_id][memory_name]
                argument_part.append('mem_offset = {}'.format(mem_offset))

            #if opcode_part == 'stack_mem' or opcode_part == 'loop_stack':
            #    # Retrieve the memory name
            #    memory_name = None
            #    for i, argument in enumerate(argument_part):
            #        if 'memory_name' in argument:
            #            memory_name = argument.split('=')[1].lstrip()
            #            memory_name = memory_name.replace("'", "")
            #            memory_name = memory_name.replace('"', '')

            #    ltask_id = GV.gtask_to_ltask[gtask_id][core_id]
            #    # mem_offset = mem.stack_mem_offset[ltask_id][memory_name]
            #    mem_offset = GV.stack_mem_offset[core_id][ltask_id][memory_name]
            #    argument_part.append('mem_offset = {}'.format(mem_offset))

            if opcode_part == 'rmw_op':
                # Retrieve the memory name
                access_type = None
                for i, argument in enumerate(argument_part):
                    if 'access_type' in argument:
                        access_type = argument.split('=')[1].lstrip()
                        access_type = access_type.replace("'", "")
                        access_type = access_type.replace('"', '')

                memory_name = None

                if access_type == 'htoc' or access_type == 'htoh':
                    for i, argument in enumerate(argument_part):
                        if 'src_mem' in argument:
                            memory_name = argument.split('=')[1].lstrip()
                            memory_name = memory_name.replace("'", "")
                            memory_name = memory_name.replace('"', '')

                    ltask_id = GV.gtask_to_ltask[gtask_id][core_id]
                    mem_offset = GV.hist_mem_offset[core_id][ltask_id][memory_name]
                    argument_part.append('mem_offset = {}'.format(mem_offset))

                # Retrieve the memory name
                if access_type == 'ctoh' or access_type == 'chtor':
                    for i, argument in enumerate(argument_part):
                        if 'dst_mem' in argument:
                            memory_name = argument.split('=')[1].lstrip()
                            memory_name = memory_name.replace("'", "")
                            memory_name = memory_name.replace('"', '')
                    ltask_id = GV.gtask_to_ltask[gtask_id][core_id]
                    mem_offset = GV.hist_mem_offset[core_id][ltask_id][memory_name]
                    argument_part.append('mem_offset = {}'.format(mem_offset))

                if access_type == 'stos':
                    for i, argument in enumerate(argument_part):
                        if 'src_mem' in argument:
                            memory_name = argument.split('=')[1].lstrip()
                            memory_name = memory_name.replace("'", "")
                            memory_name = memory_name.replace('"', '')

                    ltask_id = GV.gtask_to_ltask[gtask_id][core_id]
                    mem_offset = GV.state_mem_offset[core_id][ltask_id][memory_name]
                    argument_part.append('mem_offset = {}'.format(mem_offset))

            # Set the offset and width
            if opcode_part == 'corr_mem' or opcode_part == 'rmw_op':
                # Retrieve the memory name
                for i, argument in enumerate(argument_part):
                    if 'access_type' in argument:
                        access_type = argument.split('=')[1].lstrip()
                        access_type = access_type.replace("'", "")
                        access_type = access_type.replace('"', '')

                for i, argument in enumerate(argument_part):
                    if 'memory_name' in argument:
                        memory_name = argument.split('=')[1].lstrip()
                        memory_name = memory_name.replace("'", "")
                        memory_name = memory_name.replace('"', '')
                    elif 'src_mem' in argument and ('cto' in access_type or 'chto' in access_type):
                        memory_name = argument.split('=')[1].lstrip()
                        memory_name = memory_name.replace("'", "")
                        memory_name = memory_name.replace('"', '')
                    elif 'dst_mem' in argument and 'toc' in access_type:
                        memory_name = argument.split('=')[1].lstrip()
                        memory_name = memory_name.replace("'", "")
                        memory_name = memory_name.replace('"', '')

                # Retrieve the memory name
                for i, argument in enumerate(argument_part):
                    if 'data_type' in argument:
                        data_type = argument.split('=')[1].lstrip()
                        data_type = data_type.replace("'", "")
                        data_type = data_type.replace('"', '')
                        del argument_part[i]
                        break

                if opcode_part == 'corr_mem' or 'cto' in access_type or 'toc' in access_type or 'chto' in access_type:
                    # assert the memory name exists!!
                    assert(memory_name in GV.loop_info[gtask_id])
                    total_offset = GV.loop_info[gtask_id][memory_name]['offset']
                    total_width = GV.loop_info[gtask_id][memory_name]['width']
                    # Set the offset to be incremented @ each iteration
                    argument_part.append('offset = {}'.format(total_offset[data_type]))
                    argument_part.append('width = {}'.format(total_width[data_type]))
            ############ CORR LOOP PARSE END ############

            ############ STACK LOOP PARSE BEGIN ############
            if opcode_part == 'loop_stack':
                jump_stack.append(len(inst_list))

            #if opcode_part == 'stack_end':
            #    # for instruction jumps after the end
            #    target_addr = jump_stack.pop()
            #    jump_addr = len(inst_list) + 1

            #    # Set the jump address
            #    inst_list[target_addr] = inst_list[target_addr] + ', jump_addr = {}'.format(jump_addr)
            #    argument_part.append('jump_addr = {}'.format(target_addr))
            ############ STACK LOOP PARSE END ############

            ############ UNIT LOOP PARSE BEGIN ############
            if opcode_part == 'loop_unit':
                for i, argument in enumerate(argument_part):
                    if 'unit_name=' in argument.replace(' ', ''):
                        unit_name_str = str(argument.split('=')[1]).strip().strip("'")
                        # find unit type_id
                        type_id = GV.sim_params['unit_type_name'][gtask_id].index(unit_name_str)
                        # find per core unit number
                        per_core_unit_num = GV.sim_params['num_unit_per_core'][gtask_id][core_id][type_id]
                        argument_part[i] = f"immediate = {per_core_unit_num}"
                jump_stack.append(len(inst_list))

            if opcode_part == 'unit_end':
                # for instruction jumps after the end
                target_addr = jump_stack.pop()
                jump_addr = len(inst_list) + 1
                inst_list[target_addr] = inst_list[target_addr] + ', jump_addr = {}'.format(jump_addr)
            ############ UNIT LOOP PARSE END ############

            ############ IF PARSE BEGIN ############
            if opcode_part == 'if_begin':
                immediate = False
                for argument in argument_part:
                    if 'immediate=' in argument.replace(' ', ''):
                        immediate = True
                if immediate:
                    opcode_part = 'if_imm_begin'
                else:
                    opcode_part = 'if_reg_begin'

                jump_stack.append(len(inst_list))

            if opcode_part == 'if_end':
                # if instruction jumps to the end
                # (as if end is not a valid instruction)
                target_addr = jump_stack.pop()
                jump_addr = len(inst_list)
                inst_list[target_addr] = inst_list[target_addr] + ', jump_addr = {}'.format(jump_addr)
            ############ IF PARSE END ############

            if opcode_part == 'event_trigger':
                immediate = False
                conditional = False
                for argument in argument_part:
                    if 'immediate=' in argument.replace(' ', ''):
                        immediate = True

                # if conditional:
                if immediate:
                    opcode_part = 'event_trigger_imm'
                else:
                    opcode_part = 'event_trigger_reg'

            # We do not support if_end
            if opcode_part != 'if_end':
                #############################################
                # Append the new instruction
                inst = 'inst_func = self.isa_ctrl.' + opcode_part
                for argument in argument_part:
                    if not 'output_arr' in argument and \
                       not 'module=' in argument.replace(' ', '') and \
                       not argument == '':
                        inst += ', {}'.format(argument)
                inst_list.append(inst)

    for inst_id in range(len(inst_list)):
        inst_list[inst_id] = 'Inst({})'.format(inst_list[inst_id])

    return inst_list
