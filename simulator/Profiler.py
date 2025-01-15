import Task
import GlobalVars as GV
import numpy as np
import pandas as pd

# Generate empty dataset
# 1) The total cycles for each ts (per task) (including sync + commit)
# 2) The total cycles for each ts
# 3) The total number of events for each ts
profile_result_pandas = None
profile_result_numpy = None
row_indir = None
row_idx = None
#total_cyc = None
#total_num = None
#event_cyc = None
#event_num = None

# Init vs. Calc
# 1) Init: Reset the cycles after timestep end 
# 2) Calc: Increment the counter
# Total_Cyc vs. Event_Cyc vs. Event_Num
# 1) Total_Cyc: total cycles @ timestep (including sync)
# 2) Event_Cyc: per event cycles @ timestep
# 3) Event_Num: per event number

def initialize():
    global profile_result_numpy
    global row_indir
    global row_idx
    # The rows are hierarchically organized with
    # 1) The task
    # 2) event name
    # 3) core id
    # 4) number of cycle
    row_list = []
    row_indir = {}
    for gtask_id in range(GV.TOTAL_TASKS):
        for event_name in Task.get_task(gtask_id).event:
            for profile_type in ['num', 'cyc']:
                for core_id in range(GV.sim_params['used_core_num']):
                    # Convert the tuple
                    row_indir[(gtask_id, event_name, profile_type, core_id)] = len(row_list)
                    row_list.append((gtask_id, event_name, core_id, profile_type))
        for event_name in {'total', 'violation'}:
            for profile_type in ['num', 'cyc']:
                row_indir[(gtask_id, event_name, profile_type, 0)] = len(row_list)
                row_list.append((gtask_id, event_name, 0, profile_type))

    row_idx = pd.MultiIndex.from_tuples(row_list, names = ['task', 'event', 'core', 'profile'])

    # transpose later on
    profile_result_numpy = np.zeros((1, len(row_idx)))
    #profile_result_numpy = np.array([])

def finalize():
    global profile_result_numpy
    global profile_result_pandas
    global row_indir
    global row_list

    col_list = np.arange(0, profile_result_numpy.shape[0])
    profile_result_numpy = np.transpose(profile_result_numpy)
    profile_result_pandas = pd.DataFrame(profile_result_numpy, index = row_idx, columns = col_list)

def delete_last():
    global profile_result_numpy

    # Delete the last timestep for each task (not finished)
    if GV.debug_list['result']:
        profile_result_numpy = profile_result_numpy[:-1, :]

def add_data(dt, latency, cyc_period, num_tasks, task_list, event_list):
    global profile_result_pandas

    # Add columns with NaN values
    profile_result_pandas['dt'] = pd.NA
    profile_result_pandas['latency'] = pd.NA
    profile_result_pandas['cyc_period'] = pd.NA
    profile_result_pandas['num_tasks'] = pd.NA
    profile_result_pandas['task_list'] = pd.NA
    profile_result_pandas['event_list'] = pd.NA

    length = len(profile_result_pandas.index)
    _dt = [np.NaN for _ in range(length)]
    _latency = [np.NaN for _ in range(length)]
    _cyc_period = [np.NaN for _ in range(length)]
    _num_tasks = [np.NaN for _ in range(length)]
    _task_list = [np.NaN for _ in range(length)]
    _event_list = [np.NaN for _ in range(length)]

    # Append the dt, cyc_period, latency value to each column
    for i in range(num_tasks):
        _dt[i] = dt[i]
        _latency[i] = latency[i]
        _cyc_period[i] = cyc_period
        _num_tasks[i] = num_tasks
        _task_list[i] = task_list[i]
        _event_list[i] = event_list[i]
    # Add columns with NaN values
    profile_result_pandas['dt'] = _dt
    profile_result_pandas['latency'] = _latency
    profile_result_pandas['cyc_period'] = _cyc_period
    profile_result_pandas['num_tasks'] = _num_tasks
    profile_result_pandas['task_list'] = _task_list
    profile_result_pandas['event_list'] = _event_list

####################################
def append_timestep(gtask_id):
    global profile_result_numpy
    global row_list

    profile_result_numpy = np.append(profile_result_numpy, np.zeros((1, len(row_idx))), axis=0)

def calc_total_cyc(gtask_id, cyc, timestep):
    global profile_result_numpy
    global row_indir
    if GV.debug_list['result']:
        row_idx = row_indir[(gtask_id, 'total', 'cyc', 0)]
        profile_result_numpy[timestep][row_idx] += cyc

def calc_violation_cyc(gtask_id, cyc, timestep):
    global profile_result_numpy
    global row_indir
    if GV.debug_list['result']:
        row_idx = row_indir[(gtask_id, 'violation', 'cyc', 0)]
        profile_result_numpy[timestep][row_idx] += cyc

def calc_event_cyc(gtask_id, event_name, timestep, core_id):
    global profile_result_numpy
    global row_indir
    if GV.debug_list['result']:            
        row_idx = row_indir[(gtask_id, event_name, 'cyc', core_id)]
        profile_result_numpy[timestep][row_idx] += 1

def calc_event_number(gtask_id, event_name, timestep, core_id):
    global profile_result_numpy

    global row_indir
    if GV.debug_list['result']:
        row_idx = row_indir[(gtask_id, event_name, 'num', core_id)]
        profile_result_numpy[timestep][row_idx] += 1

def set_bci_send_cyc(gtask_id, event_name, cyc, timestep, core_id):
    global profile_result_numpy
    global row_indir
    if GV.debug_list['result']:            
        row_idx = row_indir[(gtask_id, event_name, 'cyc', core_id)]
        profile_result_numpy[timestep][row_idx] -= cyc
