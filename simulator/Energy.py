import GlobalVars as GV

op_count = {}
nJ_per_op = {}

# register read/write operation
# op_count["reg_read"] = 0
# op_count["reg_write"] = 0
op_count["reg_op"] = 0
# alu operation
op_count["alu_mac"] = 0
op_count["alu_logical"] = 0
op_count["alu_mod"] = 0
op_count["alu_div"] = 0
op_count["alu_sqrt"] = 0
op_count["near_cache_alu"] = 0
# state memory operation
op_count["state_mem_read"] = 0
op_count["state_mem_write"] = 0
# history memory operation
op_count["hist_mem_read"] = 0
op_count["hist_mem_write"] = 0
op_count["hist_mem_cache_access"] = 0
# correlation memory operation
op_count["corr_forward_read"] = 0 # only read
# op_count["corr_forward_write"] = 0
op_count["conn_mem_read"] = 0
op_count["corr_mem_read"] = 0
op_count["corr_mem_write"] = 0
op_count["corr_mem_cache_access"] = 0
# route memory operation
op_count["pck_to_route_table"] = 0
op_count["route_mem_forward_read"] = 0 # only read
# op_count["route_mem_forward_write"] = 0
op_count["route_mem_read"] = 0
# op_count["route_mem_write"] = 0 # only read
op_count["activate_loop_ctrl"] = 0
# ack memory operation
op_count["ack_left_mem_read"] = 0
op_count["ack_left_mem_write"] = 0
op_count["ack_num_mem_read"] = 0
op_count["ack_num_mem_write"] = 0
op_count["ack_stack_mem_pop"] = 0
op_count["ack_stack_mem_push"] = 0
# instruction memory operation
op_count["inst_mem"] = 0 # only read
op_count["event_to_base_bound_table"] = 0 # only read -> read only once per event scheduling
op_count["pc_update"] = 0 # ??
# packet memory operation
op_count["recv_buf_pop"] = 0
op_count["recv_buf_push"] = 0
op_count["send_buf_pop"] = 0
op_count["send_buf_push"] = 0
op_count["pck_to_event_table"] = 0 # only read
# event memory operation
op_count["stack_mem_pop"] = 0
op_count["stack_mem_push"] = 0
op_count["out_event_buf_pop"] = 0
op_count["out_event_buf_push"] = 0 # number of out_event_buf_pop should be same with out_event_buf_push
op_count["in_event_buf_pop"] = 0
op_count["in_event_buf_push"] = 0
op_count["event_table"] = 0 # only read
# sync table operation
op_count["sync_table_read"] = 0
op_count["sync_table_update"] = 0
# noc operation
op_count['NoC_hop'] = 0
# pipeline
op_count['pipeline'] = 0

# TODO : set nJ_per_op
# register read/write operation
# nJ_per_op["reg_read"] = 0
# nJ_per_op["reg_write"] = 0
nJ_per_op["reg_op"] = 0.001611
# alu operation
nJ_per_op["alu_mac"] = 0.001232
nJ_per_op["alu_logical"] = 0.000035
nJ_per_op["alu_mod"] = 0.000013
nJ_per_op["alu_div"] = 0.011939
nJ_per_op["alu_sqrt"] = 0.001302
nJ_per_op["near_cache_alu"] = 0.001439
# state memory operation
nJ_per_op["state_mem_read"] = 0.002899
nJ_per_op["state_mem_write"] = 0.003068
# history memory operation
nJ_per_op["hist_mem_read"] = (0.006455 + 0.000843)
nJ_per_op["hist_mem_write"] = (0.005907 + 0.000843)
nJ_per_op["hist_mem_cache_access"] = 0.000786 + 0.000843

# correlation memory operation
nJ_per_op["corr_forward_read"] = 0.00334507
nJ_per_op["conn_mem_read"] = 0.007814
nJ_per_op["corr_mem_read"] = 0.0172241
nJ_per_op["corr_mem_write"] = 0.0135622
nJ_per_op["corr_mem_cache_access"] = 0.000786
# route memory operation
nJ_per_op["pck_to_route_table"] = 0.000224531
nJ_per_op["route_mem_forward_read"] = 0.000497563 # only read
# nJ_per_op["route_mem_forward_write"] = 0
nJ_per_op["route_mem_read"] = 0.00259618 # only read
# nJ_per_op["route_mem_write"] = 0
nJ_per_op["activate_loop_ctrl"] = 0.000553
# ack memory operation
nJ_per_op["ack_left_mem_read"] = 0.000554
nJ_per_op["ack_left_mem_write"] = 0.000554
nJ_per_op["ack_num_mem_read"] = 0.000909757
nJ_per_op["ack_num_mem_write"] = 0.000608293
nJ_per_op["ack_stack_mem_pop"] = 0.000909757
nJ_per_op["ack_stack_mem_push"] = 0.000608293
# instruction memory operation
nJ_per_op["inst_mem"] = 0.00389856 # only read
nJ_per_op["event_to_base_bound_table"] = 0.000334833 # only read -> read only once per event scheduling
nJ_per_op["pc_update"] = 0.000068
# packet memory operation
nJ_per_op["recv_buf_pop"] = 0.000883
nJ_per_op["recv_buf_push"] = 0.001458
nJ_per_op["send_buf_pop"] = 0.000883
nJ_per_op["send_buf_push"] = 0.001458
nJ_per_op["pck_to_event_table"] = 0.000120775 # only read
# event memory operation
nJ_per_op["stack_mem_pop"] = 0.000497563
nJ_per_op["stack_mem_push"] = 0.000562768
nJ_per_op["out_event_buf_pop"] = 0.00068713
nJ_per_op["out_event_buf_push"] = 0.00092437
nJ_per_op["in_event_buf_pop"] = 0.00377371
nJ_per_op["in_event_buf_push"] = 0.00450547
nJ_per_op["event_table"] = 0.000334833 # only read
# sync table operation
nJ_per_op["sync_table_read"] = 0.000082
nJ_per_op["sync_table_update"] = 0.000082
# noc operation
# Read + Write Buffer + Link
nJ_per_op['NoC_hop'] = 0.000882536 + 0.00145796 + 0.015790
# pipeline
nJ_per_op['pipeline'] = 0.000304

mW_static = 1.421238

def calculate_energy():
    energy = 0
    for key in op_count.keys():
        if 'pop' in key:
            push_key = key.replace('pop', 'push')
            op_count[key] = op_count[push_key] # the number of pop is same with the number of push
        energy += op_count[key] * nJ_per_op[key] * 1e-9
    energy += (GV.cyc * GV.sim_params['cyc_period'] * 1e-9 * mW_static * 1e-3 * GV.sim_params['used_core_num'])

    return energy

def save_op_count():
    op_count_file = open('op_count.dat', 'w')
    energy_file = open('energy.dat', 'w')
    for key, value in op_count.items():
        op_count_file.write('{}: {}\n'.format(key, value))
        energy_file.write('{}: {}\n'.format(key, op_count[key] * nJ_per_op[key] * 1e-9))
    energy_file.write('{}: {}\n'.format('static', (GV.cyc * GV.sim_params['cyc_period'] * 1e-9 * mW_static * 1e-3 * GV.sim_params['used_core_num'])))
    op_count_file.close()
    energy_file.close()
