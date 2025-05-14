from pathlib import Path

total_tasks = []

# SS
total_tasks.append({'type'         : 'ss',
                    'instruction'  :  [[{'prev' : None, 'next' : 'ss_bci_send'}, \
                                        {'prev' : 'ss_bci_send', 'next' : 'ss_bci_recv'}],
                                       [{'prev' : None, 'next' : 'ss_bci_comp'}, \
                                        {'prev' : 'ss_bci_comp', 'next' : 'ss_peak_comp'}, \
                                        {'prev' : 'ss_peak_comp', 'next' : 'ss_template_comp'}, \
                                        {'prev' : 'ss_template_comp', 'next' : 'ss_partial_comp'}],
                                       [{'prev' : None, 'next' : 'ss_match_check'}, \
                                        {'prev' : 'ss_match_check', 'next' : 'ss_match_comp'}] \
                                      ],
                    'packet'       : {'bci':('bci','electrode'), 'ss_peak':('electrode','template'), 'ss_template':('template','electrode'), 'ss_partial':(None,None), 'ss_match':('template','electrode')}
})

# SS_no_cascade
total_tasks.append({'type'         : 'ss_nocascade',
                    'instruction'  :  [[{'prev' : None, 'next' : 'ss_bci_send'}, \
                                        {'prev' : 'ss_bci_send', 'next' : 'ss_bci_recv'}], \
                                       [{'prev' : None, 'next' : 'ss_bci_comp'}, \
                                        {'prev' : 'ss_bci_comp', 'next' : 'ss_peak_recv'}], \
                                       [{'prev' : None, 'next' : 'ss_peak_send'}, \
                                        {'prev' : 'ss_peak_send', 'next' : 'ss_template_recv'}], \
                                       [{'prev' : None, 'next' : 'ss_partial_send'}, \
                                        {'prev' : 'ss_partial_send', 'next' : 'ss_partial_comp'}], \
                                       [{'prev' : None, 'next' : 'ss_match_check'}, \
                                        {'prev' : 'ss_match_check', 'next' : 'ss_match_comp'}] \
                                      ],
                    'packet'       : {'bci':('bci','electrode'), 'ss_peak':('electrode','template'), 'ss_template':('template','electrode'), 'ss_partial':(None,None), 'ss_match':('template','electrode')}
})

# SS_emulate
total_tasks.append({'type'         : 'ss_emulate',
                    'instruction'  :  [[{'prev' : None, 'next' : 'ss_bci_send'}, \
                                        {'prev' : 'ss_bci_send', 'next' : 'ss_bci_recv'}],
                                       [{'prev' : None, 'next' : 'ss_bci_comp'}, \
                                        {'prev' : 'ss_bci_comp', 'next' : 'ss_peak_comp'}, \
                                        {'prev' : 'ss_peak_comp', 'next' : 'ss_template_comp'}, \
                                        {'prev' : 'ss_template_comp', 'next' : 'ss_partial_comp'}],
                                       [{'prev' : None, 'next' : 'ss_match_check'}, \
                                        {'prev' : 'ss_match_check', 'next' : 'ss_match_comp'}] \
                                      ],
                    'packet'       : {'bci':('bci','electrode'), 'ss_peak':('electrode','template'), 'ss_template':('template','electrode'), 'ss_partial':(None,None), 'ss_match':('template','electrode')}
})

# TM
total_tasks.append({'type'      :  'tm',
                    'instruction': [[{'prev' : None, 'next' : 'tm_bci_send'},
                                     {'prev' : 'tm_bci_send', 'next' : 'tm_bci_recv'}],
                                    [{'prev' : None, 'next' : 'tm_bci_comp'},
                                     {'prev' : 'tm_bci_comp', 'next' : 'tm_spike_recv'}],
                                    [{'prev' : None, 'next' : 'tm_psum_send'},
                                     [{'prev' : 'tm_psum_send', 'next' : 'tm_psum_recv_P1'},
                                      {'prev' : 'tm_psum_send', 'next' : 'tm_psum_recv_P2'},
                                      {'prev' : 'tm_psum_send', 'next' : 'tm_psum_recv_P3'}]]],
                    'packet' : {'bci':('bci','neuron'), 'tm_spike':('neuron','partial_template'), 'tm_psum_P1':('partial_template','template'), 'tm_psum_P2':('partial_template','template'), 'tm_psum_P3':('partial_template','template')}})


# TM_direct
total_tasks.append({'type': 'tm_direct',
                    'instruction': [[{'prev' : None, 'next' : 'tm_bci_send'},
                                     {'prev' : 'tm_bci_send', 'next' : 'tm_bci_recv'}],
                                    [{'prev' : None, 'next' : 'tm_bci_comp'},
                                     {'prev' : 'tm_bci_comp', 'next' : 'tm_spike_direct_recv'}]],
                    'packet': {'bci':('bci','neuron'), 'tm_spike_direct':('neuron','template')}
})

# TM_emulate no cascade
total_tasks.append({'type': 'tm_emulate',
                    'instruction': [[{'prev' : None, 'next' : 'tm_bci_send'},
                                     {'prev' : 'tm_bci_send', 'next' : 'tm_bci_recv'}],
                                    [{'prev' : None, 'next' : 'tm_bci_comp'},
                                     {'prev' : 'tm_bci_comp', 'next' : 'tm_spike_recv'}],
                                    [{'prev' : None, 'next' : 'tm_psum_send'},
                                     [{'prev' : 'tm_psum_send', 'next' : 'tm_psum_recv_P1'},
                                      {'prev' : 'tm_psum_send', 'next' : 'tm_psum_recv_P2'},
                                      {'prev' : 'tm_psum_send', 'next' : 'tm_psum_recv_P3'}]]],
                    'packet' : {'bci':('bci','neuron'), 'tm_spike':('neuron','partial_template'), 'tm_psum_P1':('partial_template','template'), 'tm_psum_P2':('partial_template','template'), 'tm_psum_P3':('partial_template','template')}})

# PC
total_tasks.append({'type': 'pc',
                    'instruction': [[{'prev' : None, 'next' : 'pc_bci_send'},
                                     {'prev' : 'pc_bci_send', 'next' : 'pc_bci_recv'}],
                                    [{'prev' : None, 'next' : 'pc_spike_send'},
                                     {'prev' : 'pc_spike_send', 'next' : 'pc_spike_recv'}],
                                    [{'prev' : None, 'next' : 'pc_sum_send'},
                                     {'prev' : 'pc_sum_send', 'next' : 'pc_sum_recv'}]],
                    'packet': {'bci':('bci','neuron'), 'pc_spike':('neuron','neuron'), 'pc_sum':('neuron','neuron')}
})

# PC_emulate
total_tasks.append({'type': 'pc_emulate',
                    'instruction': [[{'prev' : None, 'next' : 'pc_bci_send'},
                                     {'prev' : 'pc_bci_send', 'next' : 'pc_bci_recv'}],
                                    [{'prev' : None, 'next' : 'pc_spike_send'},
                                     {'prev' : 'pc_spike_send', 'next' : 'pc_spike_recv'}],
                                    [{'prev' : None, 'next' : 'pc_sum_send'},
                                     {'prev' : 'pc_sum_send', 'next' : 'pc_sum_recv'}]],
                    'packet': {'bci':('bci','neuron'), 'pc_spike':('neuron','neuron'), 'pc_sum':('neuron','neuron')}
})

# SNN
total_tasks.append({'type': 'snn',
                    'instruction': [[{'prev' : None, 'next' : 'nn_bci_send'},
                                     {'prev' : 'nn_bci_send', 'next' : 'nn_bci_recv'},
                                     {'prev' : 'nn_bci_recv', 'next' : 'nn_ext_comp'}],
                                    [{'prev' : None, 'next' : 'snn_bci_comp'},
                                     {'prev' : 'snn_bci_comp', 'next' : 'nn_spike_comp'}],
                                    [{'prev' : None, 'next' : 'nn_spike_sum_send'},
                                     {'prev' : 'nn_spike_sum_send', 'next' : 'nn_spike_sum_comp'}]],
                    'packet': {'bci':('bci','bci_neuron'), 'nn_ext':('bci_neuron','neuron'), 'nn_spike':('neuron','pseudo_neuron'), 'nn_spike_sum':('pseudo_neuron','neuron')}
})

# SNN_direct
total_tasks.append({'type': 'snn_direct',
                    'instruction': [[{'prev' : None, 'next' : 'nn_bci_send'},
                                     {'prev' : 'nn_bci_send', 'next' : 'nn_bci_recv'},
                                     {'prev' : 'nn_bci_recv', 'next' : 'nn_ext_comp'}],
                                    [{'prev' : None, 'next' : 'snn_bci_comp'},
                                     {'prev' : 'snn_bci_comp', 'next' : 'nn_spike_direct_comp'}]],
                    'packet': {'bci':('bci','bci_neuron'), 'nn_ext':('bci_neuron','neuron'), 'nn_spike_direct':('neuron','neuron')}
})

# ANN
total_tasks.append({'type': 'ann',
                    'instruction': [[{'prev' : None, 'next' : 'ann_bci_send'},
                                     {'prev' : 'ann_bci_send', 'next' : 'ann_bci_recv'}],
                                    [{'prev' : None, 'next' : 'ann_ext_send'},
                                     {'prev' : 'ann_ext_send', 'next' : 'nn_ext_comp'}],
                                    [{'prev' : None, 'next' : 'ann_bci_comp'},
                                     {'prev' : 'ann_bci_comp', 'next' : 'nn_spike_comp'}],
                                    [{'prev' : None, 'next' : 'nn_spike_sum_send'},
                                     {'prev' : 'nn_spike_sum_send', 'next' : 'nn_spike_sum_comp'}]],
                    'packet': {'bci':('bci','bci_neuron'), 'nn_ext':('bci_neuron','neuron'), 'nn_spike':('neuron','pseudo_neuron'), 'nn_spike_sum':('pseudo_neuron','neuron')}
})

# ANN_direct
total_tasks.append({'type': 'ann_direct',
                    'instruction': [[{'prev' : None, 'next' : 'ann_bci_send'},
                                     {'prev' : 'ann_bci_send', 'next' : 'ann_bci_recv'}],
                                    [{'prev' : None, 'next' : 'ann_ext_send'},
                                     {'prev' : 'ann_ext_send', 'next' : 'nn_ext_comp'}],
                                    [{'prev' : None, 'next' : 'ann_bci_comp'},
                                     {'prev' : 'ann_bci_comp', 'next' : 'nn_spike_direct_comp'}]],
                    'packet': {'bci':('bci','bci_neuron'), 'nn_ext':('bci_neuron','neuron'), 'nn_spike_direct':('neuron','neuron')}
})

# Add your task here

####################################################
import pickle
import os

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
                phase_start_event.append(inst['next'])
            if type(inst) != list:
                inst = inst['next']
                event_set.add(inst)
                if 'bci' not in inst: # ack events
                    if '_comp' in inst:
                        event_set.add(inst.split('_comp')[0]+'_ack')
                    elif '_recv' in inst:
                        event_set.add(inst.split('_recv')[0]+'_ack')
                    else:
                        pass
            else:
                for dat in inst:
                    event_set.add(dat['next'])
                #event_set.update(inst)
                inst = inst[0] # only one ack event
                inst = inst['next']
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
                    elif depth != max_depth:
                        event_type['start'] = None
                        event_type['end'] = 'gen_ack'
                        event_type['depth'] = depth
                    else: # last depth
                        event_type['start'] = 'send_ack'
                        event_type['end'] = None
                        event_type['depth'] = depth
                    event_type['prev'] = event_to_packet_dict[e['next']]
                    event_type_dict[e['next']] = event_type

                curr_event = event[0]['next']
                prev_event = event[0]['prev']
                if 'bci' not in curr_event: # ack events
                    if '_comp' in curr_event or '_recv' in curr_event:
                        ack_event_type = dict()
                        if '_comp' in curr_event:
                            ack_event = curr_event.split('_comp')[0]+'_ack'
                        elif '_recv' in curr_event:
                            ack_event = curr_event.split('_recv')[0]+'_ack'
                        ack_event_type['start'] = 'dec_ack'
                        ack_event_type['end'] = None
                        ack_event_type['depth'] = depth
                        ack_event_type['prev'] = event_type_dict[prev_event]['prev']
                        event_type_dict[ack_event] = ack_event_type
            else:
                depth = depth - 1
                max_depth = len(inst_list)-2
                event_type = dict()
                curr_event = event['next']
                prev_event = event['prev']
                event = None
                if depth == -1:
                    event_type['start'] = 'commit'
                    event_type['end'] = 'done'
                    event_type['depth'] = depth
                    event_type['prev'] = None
                elif depth != max_depth:
                    event_type['start'] = None
                    event_type['end'] = 'gen_ack'
                    event_type['depth'] = depth
                    event_type['prev'] = event_to_packet_dict[curr_event]
                else: # last depth
                    event_type['start'] = 'send_ack'
                    event_type['end'] = None
                    event_type['depth'] = depth
                    event_type['prev'] = event_to_packet_dict[curr_event]
                event_type_dict[curr_event] = event_type
                if 'bci' not in curr_event: # ack events
                    if '_comp' in curr_event or 'recv' in curr_event:
                        ack_event_type = dict()
                        if '_comp' in curr_event:
                            ack_event = curr_event.split('_comp')[0]+'_ack'
                        elif '_recv' in curr_event:
                            ack_event = curr_event.split('_recv')[0]+'_ack'
                        ack_event_type['start'] = 'dec_ack'
                        ack_event_type['end'] = None
                        ack_event_type['depth'] = depth
                        # The previous should be
                        curr = inst_list[depth + 1]
                        ack_event_type['prev'] = event_type_dict[prev_event]['prev']
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
    Path("./simulator/sync_config").mkdir(parents=True, exist_ok=True)
    with open('./simulator/sync_config/'+task_type+'.pickle','wb') as pkl_file:
        pickle.dump(task_info, pkl_file)

if __name__ == "__main__":
    for task in total_tasks:
        task_type = task['type']
        instructions_cfg = task['instruction']
        packets_cfg = task['packet']
        save_task_config(task_type, instructions_cfg, packets_cfg)

