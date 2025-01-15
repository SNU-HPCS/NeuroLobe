import numpy as np
import sys


workload_name = sys.argv[1]
chip_x = int(sys.argv[2])
chip_y = int(sys.argv[3])
core_x = int(sys.argv[4])
core_y = int(sys.argv[5])
n_sim = int(sys.argv[6])
n_external = int(sys.argv[7])
external = bool(sys.argv[8] == 'True')
core_per_chip = int(sys.argv[9])
cores = ' '.join(sys.argv[10:])
cores = list(cores[1:len(cores)-1].split(', '))
cores = [int(core) for core in cores]

chip_num = chip_x * chip_y
used_core_num = chip_num * core_per_chip

core_ind = np.zeros((chip_num, core_per_chip), dtype = int)
for chip in range(chip_num):
    chip_x_ind = int(chip % chip_x)
    chip_y_ind = int(chip / chip_x)
    offset = int(chip_y_ind * (chip_x * core_per_chip) + chip_x_ind * core_x)
    for core in range(core_per_chip):
        core_x_ind = int(core % core_x)
        core_y_ind = int(core / core_x)
        core_ind[chip][core] = int(offset + core_y_ind * chip_x * core_x + core_x_ind)

neu_ind = np.load(workload_name + "/gids_per_chip.npy", allow_pickle=True)


node_list = {}

for core in range(core_x * core_y):
    node_list[core] = []

for chip in range(chip_num):
    f = open(workload_name + "/chip_sub_simulation" + str(chip) + ".graph.part." + str(core_per_chip), "r")
    neu_per_chip = f.readlines()

    for neu_lid in range(len(neu_per_chip)):
        core_lid = int(neu_per_chip[neu_lid])
        core_gid = core_ind[chip][core_lid]
        neu_gid = neu_ind[chip][neu_lid]
        node_list[cores[core_gid]].append(neu_gid)

print("mapping")

# Set the external list
if external:
    for external_core_idx in range(core_x):
        # node_list[used_core_num + external_core_idx] = []
        target_core = list(range(external_core_idx, core_x * core_y, core_x))

        # FIXME: need to fixe Parse.py as well
        for core_idx in range(used_core_num):
            if cores[core_idx] in target_core:
                if (core_x * core_y + external_core_idx) not in node_list.keys():
                    node_list[core_x * core_y + external_core_idx] = []
                for i in node_list[cores[core_idx]]:
                    if i < n_external:
                        node_list[core_x * core_y + external_core_idx].append(n_sim + i)                    

# if external:
#     for external_core_idx in range(core_x):
#         node_list[used_core_num + external_core_idx] = []
#         target_core = list(range(external_core_idx, used_core_num, core_x))

#         for core_id in target_core:
#             for i in node_list[core_id]:
#                 if i < n_external:
#                     node_list[used_core_num + external_core_idx].append(n_sim + i)
else:
    # single-core
    node_list[0] += [n_sim + i for i in range(n_external)]

for core in node_list.keys():
    node_list[core].sort()

num_neu = 0
for ind in neu_ind:
    num_neu += len(ind)

np.savez(workload_name + "/mapping_" + str(num_neu) + "_" + str(used_core_num) + ".npz", node_list = node_list, dtype=object)
