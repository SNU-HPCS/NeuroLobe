import random

SCHEDULER = 0
DEBUG = 0
SAVE_ARCH = False
mem_state = None

# Related to the memory (should be moved to the benchmark)
# corr_forward
MEM_WIDTH = { 'corr_forward' : 32,
              'corr_mem'     : 32,  # 
              'route_forward': 16,  # 16bit for indirection
              'route_mem'    : 32,  # 8bit for the core_id, 16 bit for the pid
              'hist_mem'     : 32,  # fixed to 32 bit
              'state_mem'     : 32,  # fixed to 32 bit
              #'stack_mem'    : 8,   # 8bit (lid)
              'ack_left_mem' : 8,   # 8bit (# of cores)
              'ack_num_mem'  : 8,   # 8bit (maximum of 8 bit units per core)
              'ack_stack_mem': 8,   # 8bit for the core_id
              'default'      : 32}

# This should be set in the bci_processor_api
LID_PRECISION = 8

# Set the task-related parameters
MAX_LOCAL_TASK = 8
NUM_SCHEDULED_TASKS = None
NUM_COMPLETED_TASKS = None
TOTAL_TASKS = 0

# debug a specific neuron
debug_list = {}

# To initialization + debugging
sim_params = {}
lid_to_gid = None
pid_to_gid = None
spike_out = None

# Set the number of lines in corr mem
loop_info = None

# For initialization @ sync
external_id = []

# The total cores / NoC
cores = None
NoC = None

# ICN property
timing_params = {}
# NoC Timing
timing_params["core_to_sw_bw"] = 1
timing_params["sw_to_core_bw"] = 1
timing_params["sw_to_sw_bw"] = 1

external_modules = []

max_buffer_size = 0
# The simulator cycle
cyc = 0
# The next workload cycle
workload_cyc = None
# The target cycle to execute an event
target_cyc = None
prev_time = None

# The leading timestep (among the cores)
leading_timestep = None
# The timestep for each core (for profiling + debugging)
per_core_timestep = None
ts_profile = {'done': False, 'task_id': 0}
initial_timestep = None

# Check if any of the cores has performed a valid function
# If not, we skip the simulation
valid_advance = False
