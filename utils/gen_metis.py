import os
import numpy as np
import time
import sys

start = time.time()

base_path = sys.argv[1]
workload = sys.argv[2]
n_sim = int(sys.argv[3])
num_chip = int(sys.argv[4])

connection_dat = base_path + workload + "/mapping.npy"

if num_chip != 1:
    total_arr = np.load(connection_dat, allow_pickle=True)
    # faster generation for sparse matrix
    is_sparse = True
    if is_sparse:
        matrix_conn = [[] for _ in range(n_sim)]
        matrix_wgt = [[] for _ in range(n_sim)]
        input_conn = [0 for _ in range(n_sim)]

        n_syn = 0
        for syn in total_arr:
            src = int(syn[0])
            dst = int(syn[1])
            if src < n_sim:
                input_conn[dst] += 1
                if src < dst:
                    n_syn += 1
                    matrix_conn[src].append(dst)
                    matrix_conn[dst].append(src)
                    matrix_wgt[src].append(1)
                    matrix_wgt[dst].append(1)
                else:
                    # if my index is smaller just check if there is dst -> src connection
                    if not dst in matrix_conn[src]:
                        n_syn += 1
                        matrix_conn[src].append(dst)
                        matrix_conn[dst].append(src)
                        matrix_wgt[src].append(1)
                        matrix_wgt[dst].append(1)
                    else:
                        dst_index = matrix_conn[src].index(dst)
                        src_index = matrix_conn[dst].index(src)
                        matrix_wgt[src][dst_index] += 1
                        matrix_wgt[dst][src_index] += 1

        graph_path = workload
        if not os.path.isdir(graph_path):
            os.makedirs(graph_path)

        f = open(graph_path + "/chip_simulation.graph", "w")
        f.write(str(n_sim) + " " + str(n_syn) + " 011 2\n")

        for x in range(n_sim):
            f.write("1 ")
            f.write(str(input_conn[x]) + " ")
            for ind in range(len(matrix_conn[x])):
                dst = matrix_conn[x][ind]; wgt = matrix_wgt[x][ind]
                f.write(str(dst + 1) + " " + str(wgt) +" ")
            f.write("\n")

        f.close()

        end = time.time()
        print(end - start)
    else:
        matrix_conn = [[False for _ in range(n_sim)] for _ in range(n_sim)]
        matrix_wgt = [[0 for _ in range(n_sim)] for _ in range(n_sim)]
        input_conn = [0 for _ in range(n_sim)]

        n_syn = 0
        for syn in total_arr:
            src = int(syn[0])
            dst = int(syn[1])
            #(src, dst, weight, delay, _) = syn
            if src < n_sim:
                input_conn[dst] += 1
                if matrix_conn[src][dst]:
                    matrix_wgt[src][dst] = 2
                else:
                    matrix_wgt[src][dst] = 1
                    matrix_conn[src][dst] = True

                if matrix_conn[dst][src]:
                    matrix_wgt[dst][src] = 2
                else:
                    matrix_wgt[dst][src] = 1
                    matrix_conn[dst][src] = True

        graph_path = workload
        if not os.path.isdir(graph_path):
            os.makedirs(graph_path)


        f = open(graph_path + "/chip_simulation.graph", "w")
        f.write(str(n_sim) + " " + str(n_syn) + " 011 2\n")

        for x in range(n_sim):
            f.write("1 ")
            f.write(str(input_conn[x]) + " ")
            for y in range(n_sim):
                if(matrix_conn[x][y]):
                    wgt = matrix_wgt[x][y]
                    f.write(str(y + 1) + " " + str(wgt) +" ")
            f.write("\n")

        f.close()

        end = time.time()
        print(end - start)
else:
    graph_path = workload
    if not os.path.isdir(graph_path):
        os.makedirs(graph_path)
    f = open(graph_path + "/chip_simulation.graph.part.1", "w")
    for _ in range(n_sim):
        f.write("0\n")
