total_tasks = []

# SS
total_tasks.append({'type'         : 'ss',
                    'instruction'  :  [['ss_bci_send', 'ss_bci_recv'],
                                      ['ss_bci_comp', 'ss_peak_comp', 'ss_template_comp', 'ss_partial_comp'],
                                      ['ss_match_check', 'ss_match_comp']],
                    'packet'       : {'bci':(2,0), 'ss_peak':(0,1), 'ss_template':(1,0), 'ss_partial':(None,None), 'ss_match':(1,0)}
})
"""
8) ss_bci_send: post processing (accumulate ts + fifo index)
1) ss_bci_recv: event-driven (store the external data to the history)
2) ss_bci_comp: check event (check if there is a peak)
3) ss_peak_comp: event-driven (check correlated electrodes)
4) ss_template_comp: event-driven (calculate the partial distance)
5) ss_partial_comp: event-driven (accumulate the partial distance)
6) ss_match_check: check event (check if the template occupied)
7) ss_match_comp: event-driven (subtract overlap)
"""


# TM
total_tasks.append({'type'      :  'tm',
                    'instruction': [['tm_bci_send', 'tm_bci_recv'],
                                    ['tm_bci_comp', 'tm_spike_recv'],
                                    ['tm_psum_send', ['tm_psum_recv_P1','tm_psum_recv_P2','tm_psum_recv_P3']]],
                    'packet' : {'bci':(2,0), 'tm_spike':(0,3), 'tm_psum_P1':(3,1), 'tm_psum_P2':(3,1), 'tm_psum_P3':(3,1)}})

# TM_direct
total_tasks.append({'type': 'tm_direct',
                    'instruction': [['tm_bci_send', 'tm_bci_recv'],
                                    ['tm_bci_comp', 'tm_spike_direct_recv']],
                    'packet': {'bci':(2,0), 'tm_spike_direct':(0,1)}
})
"""
4) tm_bci_send: post processing (accumulate ts + fifo index + calculate correlation)
1) tm_bci_recv: event-driven (accumulate the external spike to the history buffer)
2) tm_bci_comp: check event (check if the bin count == 0 and trigger event)
3) tm_spike_direct_recv: event-driven (iterate over the templates and partial template comp)
"""

# PC
total_tasks.append({'type': 'pc',
                    'instruction': [['pc_bci_send', 'pc_bci_recv'],
                                    ['pc_spike_send', 'pc_spike_recv'],
                                    ['pc_sum_send', 'pc_sum_recv']],
                    'packet': {'bci':(1,0), 'pc_spike':(0,0), 'pc_sum':(0,0)}
})      
"""
5) pc_bci_send: post processing (reset the binned spikes from the history)
1) pc_bci_recv: event-driven (bin the spikes)
2) pc_spike send: check event (check if the binned spike count > 0)
3) pc_spike_recv: event-driven (accumulate the data to the correlogram)
4) pc_sum_send/receive: post processing (iterate over the connections, calculate pcc if src/dst fired a spike)
"""


# SNN
total_tasks.append({'type': 'snn',
                    'instruction': [['nn_bci_send', 'nn_bci_recv', 'nn_ext_comp'],
                                    ['snn_bci_comp', 'nn_spike_comp'],
                                    ['nn_spike_sum_send', 'nn_spike_sum_comp']],
                    'packet': {'bci':(2,1), 'nn_ext':(1,0), 'nn_spike':(0,3), 'nn_spike_sum':(3,0)}
})

# SNN_direct
total_tasks.append({'type': 'snn_direct',
                    'instruction': [['nn_bci_send', 'nn_bci_recv', 'nn_ext_comp'],
                                    ['snn_bci_comp', 'nn_spike_direct_comp']],
                    'packet': {'bci':(2,1), 'nn_ext':(1,0), 'nn_spike_direct':(0,0)}
})
"""
4) nn_bci_send: post processing (accumulate ts + fifo index)
1) nn_bci_recv: event-driven (bin the spikes or filter the spikes)
2) nn_ext_comp/nn_spike_direct_comp: event-driven (accumulate the weight * input to the history memory)
3) snn_bci_comp: check event (update the internal states and threshold)
"""

# ANN
total_tasks.append({'type': 'ann',
                    'instruction': [['ann_bci_send', 'ann_bci_recv'], 
                                    ['ann_ext_send', 'nn_ext_comp'],
                                    ['ann_bci_comp', 'nn_spike_comp'],
                                    ['nn_spike_sum_send', 'nn_spike_sum_comp']],
                    'packet': {'bci':(2,1), 'nn_ext':(1,0), 'nn_spike':(0,3), 'nn_spike_sum':(3,0)}
})

# ANN_direct
total_tasks.append({'type': 'ann_direct',
                    'instruction': [['ann_bci_send', 'ann_bci_recv'],
                                    ['ann_ext_send', 'nn_ext_comp'],
                                    ['ann_bci_comp', 'nn_spike_direct_comp']],
                    'packet': {'bci':(2,1), 'nn_ext':(1,0), 'nn_spike_direct':(0,0)}
})
"""
4) nn_bci_send: post processing (accumulate ts + fifo index)
1) nn_bci_recv: event-driven (bin the spikes or filter the spikes)
2) nn_ext_comp/nn_spike_direct_comp: event-driven (accumulate the weight * input to the history memory)
3) ann_bci_comp: check event (update the internal states and threshold)
"""

# Add your task here

####################################################
import pickle

def save_task_config(task_type, instructions_cfg, packets_cfg):
#packet
    packet_set = set(packets_cfg.keys())
    packet_set.update({'init', 'sync', 'commit'})

# event
    event_set = set()
    phase_start_event = []
    for phase, inst_list in enumerate(instructions_cfg):
        for depth, inst in enumerate(inst_list):
            if depth == 0:
                phase_start_event.append(inst)
            if type(inst) != list:
                event_set.add(inst)
                if 'bci' not in inst: # ack events
                    if '_comp' in inst:
                        event_set.add(inst.split('_comp')[0]+'_ack')
                    elif '_recv' in inst:
                        event_set.add(inst.split('_recv')[0]+'_ack')
                    else:
                        pass
            else:
                event_set.update(inst)
                inst = inst[0] # only one ack event
                if 'bci' not in inst:
                    if '_comp' in inst:
                        event_set.add(inst.split('_comp')[0]+'_ack')
                    elif '_recv' in inst:
                        event_set.add(inst.split('_recv')[0]+'_ack')
                    else:
                        pass
    event_set.update({'bci_ack', 'bci_init', 'sync_done'})

#event_to_packet
    event_to_packet_dict = dict()
    for event in event_set:
        if 'bci' in event:
            if event == 'bci_init':
                event_to_packet_dict[event] = 'init'
            else:
                event_to_packet_dict[event] = 'bci'
        else:
            if '_comp' in event:
                event_to_packet_dict[event] = event.split('_comp')[0]+event.split('_comp')[1]
            elif '_recv' in event:
                event_to_packet_dict[event] = event.split('_recv')[0]+event.split('_recv')[1]
            else:
                if event == 'sync_done':
                    event_to_packet_dict[event] = 'sync'
                else:
                    pass

# packet_to_event
    packet_to_event_dict = dict()
    for event in event_to_packet_dict.keys():
        packet = event_to_packet_dict[event]
        if packet == 'init':
            packet_to_event = dict()
            packet_to_event['packet'] = 'bci_init'
            packet_to_event['ack'] = 'bci_ack'
            packet_to_event_dict[packet] = packet_to_event
        elif packet == 'bci':
            if '_bci_recv' in event:
                packet_to_event = dict()
                packet_to_event['packet'] = event
                packet_to_event['ack'] = 'bci_ack'
                packet_to_event_dict[packet] = packet_to_event
        elif packet == 'sync':
            packet_to_event = dict()
            packet_to_event['synchronize'] = event
            packet_to_event_dict[packet] = packet_to_event
        else:
            packet_to_event = dict()
            packet_to_event['packet'] = event
            if '_comp' in event:
                packet_to_event['ack'] = event.split('_comp')[0]+'_ack'
            elif '_recv' in event:
                packet_to_event['ack'] = event.split('_recv')[0]+'_ack'
            else:
                assert(0)
            packet_to_event_dict[packet] = packet_to_event
    packet_to_event_dict['commit'] = {'synchronize': phase_start_event[1:]+phase_start_event[:1]} #rotate list

# event_type
    event_type_dict = dict()
    for phase, inst_list in enumerate(instructions_cfg):
        for depth, event in enumerate(inst_list):
            if type(event) == list:
                depth = depth - 1
                max_depth = len(inst_list)-2
                for e in event:
                    event_type = dict()
                    if depth == -1:
                        event_type['start'] = 'commit'
                        event_type['end'] = 'done'
                        event_type['depth'] = depth
                        event_type['prev'] = None
                    elif depth != max_depth:
                        event_type['start'] = None
                        event_type['end'] = 'gen_ack'
                        event_type['depth'] = depth
                        event_type['prev'] = event_to_packet_dict[e]
                    else: # last depth
                        event_type['start'] = 'send_ack'
                        event_type['end'] = None
                        event_type['depth'] = depth
                        event_type['prev'] = event_to_packet_dict[e]
                    event_type_dict[e] = event_type
                event = event[0]
                if 'bci' not in event: # ack events
                    if '_comp' in event:
                        ack_event_type = dict()
                        ack_event = event.split('_comp')[0]+'_ack'
                        ack_event_type['start'] = 'dec_ack'
                        ack_event_type['end'] = None
                        ack_event_type['depth'] = depth
                        if depth == 0:
                            ack_event_type['prev'] = None
                        else:
                            ack_event_type['prev'] = event_to_packet_dict[inst_list[depth]]
                        event_type_dict[ack_event] = ack_event_type
                    elif '_recv' in event:
                        ack_event_type = dict()
                        ack_event = event.split('_recv')[0]+'_ack'
                        ack_event_type['start'] = 'dec_ack'
                        ack_event_type['end'] = None
                        ack_event_type['depth'] = depth
                        if depth == 0:
                            ack_event_type['prev'] = None
                        else:
                            ack_event_type['prev'] = event_to_packet_dict[inst_list[depth]]
                        event_type_dict[ack_event] = ack_event_type
            else:
                depth = depth - 1
                max_depth = len(inst_list)-2
                event_type = dict()
                if depth == -1:
                    event_type['start'] = 'commit'
                    event_type['end'] = 'done'
                    event_type['depth'] = depth
                    event_type['prev'] = None
                elif depth != max_depth:
                    event_type['start'] = None
                    event_type['end'] = 'gen_ack'
                    event_type['depth'] = depth
                    event_type['prev'] = event_to_packet_dict[event]
                else: # last depth
                    event_type['start'] = 'send_ack'
                    event_type['end'] = None
                    event_type['depth'] = depth
                    event_type['prev'] = event_to_packet_dict[event]
                event_type_dict[event] = event_type
                if 'bci' not in event: # ack events
                    if '_comp' in event:
                        ack_event_type = dict()
                        ack_event = event.split('_comp')[0]+'_ack'
                        ack_event_type['start'] = 'dec_ack'
                        ack_event_type['end'] = None
                        ack_event_type['depth'] = depth
                        if depth == 0:
                            ack_event_type['prev'] = None
                        else:
                            ack_event_type['prev'] = event_to_packet_dict[inst_list[depth]]
                        event_type_dict[ack_event] = ack_event_type
                    elif '_recv' in event:
                        ack_event_type = dict()
                        ack_event = event.split('_recv')[0]+'_ack'
                        ack_event_type['start'] = 'dec_ack'
                        ack_event_type['end'] = None
                        ack_event_type['depth'] = depth
                        if depth == 0:
                            ack_event_type['prev'] = None
                        else:
                            ack_event_type['prev'] = event_to_packet_dict[inst_list[depth]]
                        event_type_dict[ack_event] = ack_event_type

    for event in event_set:
        if event in event_type_dict.keys():
            pass
        else:
            event_type = dict()
            if event == 'bci_init':
                event_type['start'] = 'send_ack'
                event_type['end'] = None
                event_type['depth'] = 0
                event_type['prev'] = event_to_packet_dict[event]
            elif event == 'bci_ack':
                event_type['start'] = 'dec_ack'
                event_type['end'] = None
                event_type['depth'] = 0
                event_type['prev'] = None
            elif event == 'sync_done':
                event_type['start'] = event_to_packet_dict[event]
                event_type['end'] = None
                event_type['depth'] = None
                event_type['prev'] = None
            else:
                print(event)
                assert(0)
            event_type_dict[event] = event_type

# packet_to_type
    packet_to_type_dict = packets_cfg
    packet_to_type_dict['done'] = (None, None)

# task_info
    task_info = dict()
    task_info['type'] = task_type.split('_')[0]
    task_info['packet'] = packet_set
    task_info['event'] = event_set
    task_info['event_type'] = event_type_dict
    task_info['packet_to_event'] = packet_to_event_dict
    task_info['packet_to_type'] = packet_to_type_dict
    task_info['num_phase'] = len(packet_to_event_dict['commit']['synchronize'])

    with open('./simulator/task_config/'+task_type+'.pickle','wb') as pkl_file:
        pickle.dump(task_info, pkl_file)


if __name__ == "__main__":
    for task in total_tasks:
        task_type = task['type']
        instructions_cfg = task['instruction']
        packets_cfg = task['packet']
        save_task_config(task_type, instructions_cfg, packets_cfg)

