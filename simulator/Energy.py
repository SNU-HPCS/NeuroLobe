import GlobalVars as GV
import re

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


def parse_energy_file():
    f = open('energy.cfg')
    lines = f.readlines()
    energy = {'static' : 0}
    for line in lines:
        line = line.rstrip()
        dat = line.split('\t')
        while '' in dat:
            dat.remove('')
        name = dat[0]
        read_energy = None
        write_energy = None
        dynamic_energy = None
        static_power = None
        if len(dat) == 4:
            read_energy = float(dat[1])
            write_energy = float(dat[2])
            static_power = float(dat[3])
            energy[name + '_read'] = read_energy
            energy[name + '_write'] = write_energy
            energy['static'] += static_power
        elif len(dat) == 3:
            dynamic_energy = float(dat[1])
            static_power = float(dat[2])
            energy[name] = dynamic_energy
            energy['static'] += static_power

    return energy

energy = parse_energy_file()


# register read/write operation
nJ_per_op["reg_op"] = energy["reg_op"]
# alu operation
nJ_per_op["alu_mac"] = energy["alu_mac"]
nJ_per_op["alu_logical"] = energy["alu_logical"]
nJ_per_op["alu_mod"] = energy["alu_mod"]
nJ_per_op["alu_div"] = energy["alu_div"] + energy["f2f"]
nJ_per_op["alu_sqrt"] = energy["alu_sqrt"] + energy["f2f"]
nJ_per_op["near_cache_alu"] = energy["near_cache_alu"]
# state memory operation
nJ_per_op["state_mem_read"] = energy["state_mem_read"]
nJ_per_op["state_mem_write"] = energy["state_mem_write"]
# history memory operation
nJ_per_op["hist_mem_read"] = energy["hist_mem_read"] + \
                             energy["hist_mem_trans"]
nJ_per_op["hist_mem_write"] = energy["hist_mem_write"] + \
                              energy["hist_mem_trans"]
nJ_per_op["hist_mem_cache_access"] = energy["hist_mem_cache"]

# correlation memory operation
nJ_per_op["corr_forward_read"] = energy["corr_forward_read"]
nJ_per_op["conn_mem_read"] = energy["conn_mem_read"]
nJ_per_op["corr_mem_read"] = energy["corr_mem_read"]
nJ_per_op["corr_mem_write"] = energy["corr_mem_write"]
nJ_per_op["corr_mem_cache_access"] = energy["corr_mem_cache"]

# route memory operation
nJ_per_op["pck_to_route_table"] = energy["pck_to_route_table_read"]
nJ_per_op["route_mem_forward_read"] = energy["route_mem_forward_read"]
nJ_per_op["route_mem_read"] = energy["route_mem_read"]
nJ_per_op["activate_loop_ctrl"] = energy["activate_loop_ctrl"]

# ack memory operation
nJ_per_op["ack_left_mem_read"] = energy["ack_left_mem"]
nJ_per_op["ack_left_mem_write"] = energy["ack_left_mem"]
nJ_per_op["ack_num_mem_read"] = energy["ack_num_mem_read"]
nJ_per_op["ack_num_mem_write"] = energy["ack_num_mem_write"]
nJ_per_op["ack_stack_mem_pop"] = energy["ack_stack_mem_read"]
nJ_per_op["ack_stack_mem_push"] = energy["ack_stack_mem_write"]

# instruction memory operation
nJ_per_op["inst_mem"] = energy["inst_mem_read"]
nJ_per_op["event_to_base_bound_table"] = energy["event_to_base_bound_table_read"]
nJ_per_op["pc_update"] = energy["pc_update"]
# packet memory operation
nJ_per_op["recv_buf_pop"] = energy["recv_buf_read"]
nJ_per_op["recv_buf_push"] = energy["recv_buf_write"]
nJ_per_op["send_buf_pop"] = energy["recv_buf_read"]
nJ_per_op["send_buf_push"] = energy["recv_buf_write"]
nJ_per_op["pck_to_event_table"] = energy["pck_to_event_table_read"]

# event memory operation
nJ_per_op["out_event_buf_pop"] = energy["out_event_buf_read"]
nJ_per_op["out_event_buf_push"] = energy["out_event_buf_write"]
nJ_per_op["in_event_buf_pop"] = energy["in_event_buf_read"]
nJ_per_op["in_event_buf_push"] = energy["in_event_buf_write"]
nJ_per_op["event_table"] = energy["event_table_read"]
# sync table operation
nJ_per_op["sync_table_read"] = energy["sync_table"]
nJ_per_op["sync_table_update"] = energy["sync_table"]
# noc operation
# Read + Write Buffer + Link
nJ_per_op['NoC_hop'] = energy["recv_buf_read"] + energy["recv_buf_write"] + energy["wire"]

mW_static = energy["static"]

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
