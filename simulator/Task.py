import GlobalVars as GV
import pickle

SNN = pickle.load(open('sync_config/snn.pickle','rb'))
SNN_direct = pickle.load(open('sync_config/snn_direct.pickle','rb'))
ANN = pickle.load(open('sync_config/ann.pickle','rb'))
ANN_direct = pickle.load(open('sync_config/ann_direct.pickle','rb'))
SS = pickle.load(open('sync_config/ss.pickle','rb'))
SS_nocascade = pickle.load(open('sync_config/ss_nocascade.pickle','rb'))
SS_emulate = pickle.load(open('sync_config/ss_emulate.pickle','rb'))
TM = pickle.load(open('sync_config/tm.pickle','rb'))
TM_direct = pickle.load(open('sync_config/tm_direct.pickle','rb'))
TM_emulate = pickle.load(open('sync_config/tm_emulate.pickle','rb'))
PC = pickle.load(open('sync_config/pc.pickle','rb'))
PC_emulate = pickle.load(open('sync_config/pc_emulate.pickle','rb'))

# Set the total task information
total_info = {}
total_info[SNN['type']] = SNN
total_info[ANN['type']] = ANN
total_info[SS['type']] = SS
total_info[TM['type']] = TM
total_info[PC['type']] = PC

# Task
class Task:
    def __init__(self, task_info, task_id):
        self.task_id = task_id
        self.type = task_info['type']
        self.event = task_info['event']
        self.packet = task_info['packet']
        self.packet_to_event = task_info['packet_to_event']
        self.packet_to_type = task_info['packet_to_type']
        self.event_type = task_info['event_type']
        self.task_const = {}
        self.num_phase = task_info['num_phase']

# Define the total tasks
tasks = []

def add_task(task_type):
    global tasks

    # initialize a task for the given type
    if GV.sim_params['baseline'] and total_info[task_type]['type']=='pc':
        task = Task(PC_emulate, len(tasks))
    elif GV.sim_params['baseline'] and total_info[task_type]['type']=='tm':
        task = Task(TM_emulate, len(tasks))
    elif GV.sim_params['baseline'] and total_info[task_type]['type']=='ss':
        task = Task(SS_emulate, len(tasks))
    elif GV.sim_params['use_partial'][len(tasks)]==0 and total_info[task_type]['type']=='snn':
        task = Task(SNN_direct, len(tasks))
    elif GV.sim_params['use_partial'][len(tasks)]==0 and total_info[task_type]['type']=='ann':
        task = Task(ANN_direct, len(tasks))
    elif GV.sim_params['use_partial'][len(tasks)]==0 and total_info[task_type]['type']=='tm':
        task = Task(TM_direct, len(tasks))
    elif GV.sim_params['no_cascade'] and total_info[task_type]['type'] == 'ss':
        task = Task(SS_nocascade, len(tasks))
    else:
        task = Task(total_info[task_type], len(tasks))
    tasks.append(task)
    GV.TOTAL_TASKS += 1

def get_task(task_id):
    global tasks
    return tasks[task_id]

def set_const(task_id, key, data):
    global tasks
    tasks[task_id].task_const[key] = data
