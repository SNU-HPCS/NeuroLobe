import subprocess
import sys
import time
import itertools
import configparser
import ast
import os
import shutil

import numpy as np
import pickle

# read simulator executables
simulator_executables_path = os.getcwd() + "/simulator_executables"
# load sim params

with open(simulator_executables_path + '/sim_params.pkl', 'rb') as f:
    loaded_sim_params = pickle.load(f)

working_directory = loaded_sim_params['working_directory']
origin = loaded_sim_params['simulator_path']

# Set the fixed simulator parameter shared across different tasks
# working_directory = os.getcwd() + "/" + config.get('path', 'runspace_path')
# if not os.path.exists(working_directory): os.makedirs(working_directory)

if not os.path.exists(working_directory):
    assert(0) # compiled files not generated

# build runspace and run simulation
folder_name = "simulator"
subprocess.Popen(['rm', '-rf', folder_name],
                 cwd=working_directory).communicate()

time.sleep(1)

subprocess.Popen(['cp', '-rf', origin, folder_name],
                 cwd=working_directory).communicate()

time.sleep(1)

# compile
proc = subprocess.Popen(['python setup_simulate.py build_ext --inplace'],
                        close_fds=True, shell=True, cwd=working_directory + folder_name)
out, err = proc.communicate()

time.sleep(1)
# execute
# proc = subprocess.Popen(['python3 Main.py ' + argument]\
#command = 'python3 Main.py ' + argument + ' > log 2> err&'
command = 'python -u Main_simulate.py --executable_path ' + str(os.getcwd())
f = open(working_directory + folder_name + "/command.sh", "w")
f.write(command)
f.close()
proc = subprocess.Popen([command],
                        close_fds=True, shell=True, cwd=working_directory + folder_name)
out, err = proc.communicate()

time.sleep(1)
