# import Task
# from ProcessGraph import PC
# import configparser
# import pickle
# import GlobalVars as GV
# import subprocess
# import time
# import os
import pickle
import subprocess
import sys
import time
import itertools
import configparser
import ast
import os
import shutil

import numpy as np
from task_compiler import total_tasks, save_task_config

# First, we should compile the task graph
for task in total_tasks:
    task_type = task['type']
    instructions_cfg = task['instruction']
    packets_cfg = task['packet']
    save_task_config(task_type, instructions_cfg, packets_cfg)


# NOTE
# Compiler do the followings:
# 1) Parse the instruction files and save the PC controller/ ProcessGraph per core as pickle file.
# 2) Read the config files (example_snn.cfg, example_ss.cfg ... , simulator.cfg) and determine the hardware configuration

def get_list(param):
    return ast.literal_eval(config.get('simulation_parameter', param))

num_tasks = len(sys.argv) - 2
if num_tasks == -1: raise("simulator.cfg and task.cfg do not exist")
if num_tasks == 0: raise("task.cfg does not exist")

workload_name_list = []
workload_timestep_list = []
mapping_file_name_list = []
dt_list = []
latency_list = []
ufactor_list = []
task_type_list = []
task_to_core = []
p_unit_list = []
use_partial_list = []

config = configparser.ConfigParser()
config.read(sys.argv[1])

# Set the fixed simulator parameter shared across different tasks
working_directory = os.getcwd() + "/" + config.get('path', 'runspace_path')
if not os.path.exists(working_directory): os.makedirs(working_directory)
origin = os.getcwd() + "/" + config.get('path', 'simulator_path')
workload_path = os.getcwd() + "/" + config.get('path', 'workload_path')
mapper_path = os.getcwd() + "/" + config.get('path', 'mapper_path')
max_core_xy = ast.literal_eval(config.get('sim_params', 'max_core_xy'))
max_core_x, max_core_y = max_core_xy # global
cyc_period = config.get('sim_params', 'cyc_period')
external = config.get('sim_params', 'external') == "True"
gpmetis = config.get('sim_params', 'gpmetis') == "True"
highest_priority = config.get('sim_params', 'highest_priority')
saved_arch_states = config.get('sim_params', 'saved_arch_states')

# Set the task-specific parameters for each task
for task_id in range(num_tasks):
    config = configparser.ConfigParser()
    config.read(sys.argv[task_id + 2])
    shutil.copy(sys.argv[task_id + 2], working_directory + "task{}.cfg".format(task_id))

    #### mapping parameters ####
    ufactor_list.append(int(config.get('mapping_params', 'ufactor')))
    workload_name_list.append(config.get('task_params', 'workload_name'))
    workload_timestep_list.append(config.get('task_params', 'workload_timestep'))
    task_type_list.append(config.get('task_params', 'task_type'))

    dt_list.append(int(float(config.get('task_params', 'dt'))))
    latency_list.append(int(float(config.get('task_params', 'latency'))))
    core_list = eval(config.get('task_params', 'core'))
    if len(core_list) == 0:
        core_list = list(range(max_core_x * max_core_y))
    assert(all(core in range(max_core_x * max_core_y) for core in core_list)) # per-task core should be subset of simulation cores
    task_to_core.append(core_list)
    p_unit_list.append(config.get('task_params', 'p_unit'))
    use_partial_list.append(int(config.get('task_params', 'use_partial')))

# perform mapping
for param in zip(workload_name_list, ufactor_list, dt_list, latency_list, task_to_core):
    workload_name = param[0]
    ufactor = param[1]
    dt = param[2]
    latency = param[3]
    cores = param[4]

    network_dict = np.load(workload_path + workload_name + "/network_parameter.npy", allow_pickle=True).item()
    num_internal = network_dict["num_internal"]
    num_external = network_dict["num_external"]

    used_chip_num = 1
    used_core_num = len(cores)
    assert(used_core_num == 1 or external) # non-external mode should be single-core
    total_core_num = max_core_x * max_core_y

    mapping_name = "mapping_" + str(num_internal) + "_" + str(used_core_num)

    mapping_name += ".npz"
    mapping_folder_name = mapper_path + workload_name
    mapping_file_name = mapping_folder_name + "/" + mapping_name
    mapping_file_name_list.append(mapping_file_name)

    if not os.path.isdir(mapping_folder_name):
        os.mkdir(mapping_folder_name)

    if gpmetis:
        print("Gpmetis mapping start")

        proc = subprocess.Popen(['python3 gen_metis.py {} {} {} {}' \
                                .format(workload_path, workload_name, num_internal, used_chip_num)]
                                , close_fds=True, shell=True, cwd=mapper_path)
        proc.communicate()

        if used_chip_num > 1:
            proc = subprocess.Popen(['gpmetis chip_simulation.graph -ufactor {} {}'.format(ufactor, used_chip_num)]
                                    , close_fds=True, shell=True, cwd=mapper_path + workload_name)
        proc.communicate()

        proc = subprocess.Popen(['python gen_submetis.py {} {} {} {} {}' \
                                .format(workload_path, workload_name, used_chip_num, num_internal, used_core_num)]
                                , close_fds=True, shell=True, cwd=mapper_path)
        proc.communicate()

        for i in range(used_chip_num):
            if max_core_x * max_core_y > 1:
                proc = subprocess.Popen(['gpmetis chip_sub_simulation{}.graph -ufactor {} -ptype rb {}'
                                        .format(i, ufactor, used_core_num)]
                                        , close_fds=True, shell=True, cwd=mapper_path + workload_name)
                proc.communicate()

        proc = subprocess.Popen(['python Parse.py {} {} {} {} {} {} {} {} {} {}'
                                .format(workload_name, 1, 1, max_core_x, max_core_y, num_internal, num_external, external, used_core_num, str(cores))]
                                , close_fds=True, shell=True, cwd=mapper_path)
        proc.communicate()

    else:
        node_list = {}

        # print(cores, used_core_num)
        for core in range(total_core_num):
            node_list[core] = []

        for neu in range(num_internal):
            node_list[cores[neu % used_core_num]].append(neu)

        # Set the external list
        if external:
            for external_core_idx in range(max_core_x):
                # node_list[used_core_num + external_core_idx] = []
                target_core = list(range(external_core_idx, total_core_num, max_core_x))

                # FIXME: need to fixe Parse.py as well
                for core_idx in range(used_core_num):
                    if cores[core_idx] in target_core:
                        if (total_core_num + external_core_idx) not in node_list.keys():
                            node_list[total_core_num + external_core_idx] = []
                        for i in node_list[cores[core_idx]]:
                            if i < num_external:
                                node_list[total_core_num + external_core_idx].append(num_internal + i)

        # for core in range(used_core_num):
        #     node_list[core] = []

        # for neu in range(num_internal):
        #     node_list[neu % used_core_num].append(neu)

        # # Set the external list
        # if external:
        #     for external_core_idx in range(max_core_x):
        #         # node_list[used_core_num + external_core_idx] = []
        #         target_core = list(range(external_core_idx, total_core_num, max_core_x))

        #         # FIXME: need to fixe Parse.py as well
        #         for core_idx in range(used_core_num):
        #             if cores[core_idx] in target_core:
        #                 node_list[used_core_num + external_core_idx] = []
        #                 for i in node_list[core_idx]:
        #                     if i < num_external:
        #                         node_list[used_core_num + external_core_idx].append(num_internal + i)
        else:
            # single-core
            node_list[0] += [num_internal + i for i in range(num_external)]

        for core in node_list.keys():
            node_list[core].sort()

        # print(node_list)
        np.savez(mapping_file_name, node_list = node_list, dtype=object)

max_core_x, max_core_y = max_core_xy # global

workload_name = ""
for dat in workload_name_list:
    workload_name += workload_path + str(dat) + "/"
    workload_name += " "

workload_timestep = ""
for dat in workload_timestep_list:
    workload_timestep += str(dat)
    workload_timestep += " "

mapping_file_name = ""
for dat in mapping_file_name_list:
    mapping_file_name += str(dat)
    mapping_file_name += " "

task_type = ""
for dat in task_type_list:
    task_type += str(dat)
    task_type += " "

dt = ""
for dat in dt_list:
    dt += str(dat)
    dt += " "

latency = ""
for dat in latency_list:
    latency += str(dat)
    latency += " "

use_partial = ""
for dat in use_partial_list:
    use_partial += str(dat)
    use_partial += " "


argument = " --workload_path " + workload_name + \
           " --mapping_file_path " + mapping_file_name + \
           " --highest_priority " + highest_priority + \
           " --max_core {} {}".format(max_core_x, max_core_y) + \
           " --workload_timestep " + workload_timestep + \
           " --task_type " + task_type + \
           " --dt " + dt + \
           " --latency " + latency + \
           " --cyc_period " + cyc_period + \
           " --num_tasks " + str(num_tasks) + \
           " --task_to_core " + str(task_to_core) + \
           " --p_unit " + str(p_unit_list) + \
           " --use_partial " + use_partial + \
           " --external " + str(external) + \
           " --working_directory " + str(working_directory) + \
           " --simulator_path " + str(origin) + \
           " --executable_path " + str(os.getcwd()) + \
           " --saved_arch_states " + str(saved_arch_states)

folder_name = "compiler"
subprocess.Popen(['rm', '-rf', folder_name],
                 cwd=working_directory).communicate()

time.sleep(1)

subprocess.Popen(['cp', '-rf', origin, folder_name],
                 cwd=working_directory).communicate()

time.sleep(1)

# cythonize
proc = subprocess.Popen(['python setup_compile.py build_ext --inplace'],
                        close_fds=True, shell=True, cwd=working_directory + folder_name)
out, err = proc.communicate()

time.sleep(1)
# execute
command = 'python -u Main_compile.py ' + argument
f = open(working_directory + folder_name + "/command.sh", "w")
f.write(command)
f.close()
proc = subprocess.Popen([command],
                        close_fds=True, shell=True, cwd=working_directory + folder_name)
out, err = proc.communicate()

time.sleep(1)
