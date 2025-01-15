import sys
import numpy as np
import matplotlib.pyplot as plt
#from dandi.dandiapi import DandiAPIClient
from nlb_tools.nwb_interface import NWBDataset
import pandas as pd
import time
import argparse

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, Dataset
from datetime import datetime
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence
from torch.utils.data import TensorDataset, DataLoader, Dataset
import torch.nn.utils.prune as prune
from bci_processor_api import *


from decoders import *

import tqdm

# Set the seed
torch.manual_seed(5)

# Set training arguments
parser = argparse.ArgumentParser()
parser.add_argument('--dev', type=int, dest='GPU_ID', default=0, help='GPU ID')
parser.add_argument('--model', type=str, dest='model_name', default='RNN', help='model name (RNN / FC / FCSNN / RNNSNN)')
parser.add_argument('--neuron', type=str, dest='neuron_name', default='cuba', help='model name (lif / cuba / ...)')
parser.add_argument('--bins', type=int, dest='bins', default=5, help='bin width(ms)')
parser.add_argument('--DT', type=int, dest='DT', default=1, help='DT (ms)')
parser.add_argument('--ep_wgt', type=int, dest='EPOCHS_WEIGHT', default=1, help='Epochs Weight')
parser.add_argument('--ep_delay', type=int, dest='EPOCHS_DELAY', default=1, help='Epochs Delay')
parser.add_argument('--lr_weight', type=float, dest='LR_WEIGHT', default=0.1, help='weight learning rate')
parser.add_argument('--lr_delay', type=float, dest='LR_DELAY', default=0.1, help='delay learning rate')
parser.add_argument('--lr_step', type=float, dest='LR_STEP', default=100, help='delay learning rate')
parser.add_argument('--delay', type=int, dest='delay', default=0, help='delay')
parser.add_argument('--gaussian', type=int, dest='gaussian', default=0, help='gaussian')
parser.add_argument('--split', type=int, dest='split', default=0, help='split')
parser.add_argument('--inference', type=int, dest='inference', default=False, help='split')

parser.add_argument('--h_dim', type=int, dest='h_dim', default=256, help='list of hidden layer dimension')
parser.add_argument('--original_num', type=int, dest='original_num', default=256, help='original electrode number')
parser.add_argument('--batch_size', type=int, dest='batch_size', default=-1, help='batch_size')
parser.add_argument('--duplicate', type=int, dest='duplicate', default=1, help='duplicate electrodes')
parser.add_argument('--target_cell_num', type=int, dest='target_cell_num', default=1, help='target_cell_num')
parser.add_argument('--fr_scale', type=int, dest='fr_scale', default=1, help='firing rate scale')
parser.add_argument('--num_data', type=int, dest='num_data', default=1, help='num data to use')

args = parser.parse_args()
print(args)

GPU_ID          = args.GPU_ID
bin_width_ms    = args.bins

model_name      = args.model_name
neuron_name     = args.neuron_name

EPOCHS_WEIGHT   = args.EPOCHS_WEIGHT
EPOCHS_DELAY    = args.EPOCHS_DELAY
LR_WEIGHT       = args.LR_WEIGHT
LR_DELAY        = args.LR_DELAY
LR_STEP         = args.LR_STEP
delay           = bool(args.delay)
electrode_num   = args.original_num
h_dim           = args.h_dim
split           = args.split

duplicate       = args.duplicate
target_cell_num = args.target_cell_num
fr_scale        = args.fr_scale
NUM_DATA        = args.num_data

# Set the GPU configuration (Multi GPU support will be added soon
if GPU_ID == -1:
    device = torch.device('cpu')
else:
    device = torch.device('cuda:' + str(GPU_ID) if torch.cuda.is_available() else 'cpu')

# Set the Dataloader configuration
BATCH_SIZE          = args.batch_size

# Set result path
result_dir="../dataset/"
# result_path = result_dir + str(model_name) + "_" + str(target_cell_num) + "_" + str(fr_scale) + "x" + "_" + str(bin_width_ms)
result_path = result_dir + str(model_name) + "_" + str(target_cell_num) + "_" + "16x" + "_" + str(bin_width_ms)
print("result_path:", result_path)

# Load the data
save_dir = "./spiking_data"

gaussian            = args.gaussian

if fr_scale == 1:
    dat = np.load(save_dir + "/spikes_{}_duplicate_{}.npz".format("MAZE", duplicate))
else:
    dat = np.load(save_dir + "/spikes_{}_duplicate_{}_{}x.npz".format("MAZE", duplicate, fr_scale))

X_train             = dat["X_train"]
X_train_spike       = dat["X_train_spike"]

X_valid             = dat["X_valid"]
X_valid_spike       = dat["X_valid_spike"]

y_train             = dat["y_train"]
y_valid             = dat["y_valid"]

if gaussian:
    assert(not "SNN" in model_name)
    if fr_scale == 1:
        dat = np.load(save_dir + "/spikes_{}_duplicate_{}_gaussian_{}.npz".format("MAZE", duplicate, filter_dat), allow_pickle=True)
    else:
        dat = np.load(save_dir + "/spikes_{}_duplicate_{}_{}x_gaussian_{}.npz".format("MAZE", duplicate, fr_scale, filter_dat), allow_pickle=True)
    X_train_gaussian    = dat["X_train_gaussian"]
    X_valid_gaussian    = dat["X_valid_gaussian"]
    X_train = X_train_gaussian
    X_valid = X_valid_gaussian
input_dim           = X_train.shape[-1]

# slice data to match target cell num
if target_cell_num != -1:
    assert(target_cell_num <= input_dim)
    remove_list = np.array([], dtype=np.int32)
    index = 0
    while len(remove_list) < input_dim - target_cell_num:
        remove_candidate = np.arange(electrode_num, dtype=np.int32) * duplicate + index
        remove_list = np.append(remove_list, remove_candidate[:min(len(remove_candidate), input_dim - target_cell_num - len(remove_list))])
        index += 1
    X_train = np.delete(X_train, remove_list, 2)
    X_valid = np.delete(X_valid, remove_list, 2)
    X_train_spike = np.delete(X_train_spike, remove_list, 2)
    X_valid_spike = np.delete(X_valid_spike, remove_list, 2)
    input_dim = target_cell_num

    print("removed", len(remove_list), "neurons to match target cell num")
    print("removed electrode indicies:", remove_list)
    print("new input dimension:", input_dim)

    print(X_train.shape, y_train.shape)
    print(X_valid.shape, y_valid.shape)


# generate dataset using spiking data if necessary (transposed, boolean)
DT = bin_width_ms
if "SNN" in model_name:
    DT = 1
    X_train = X_train_spike
    X_valid = X_valid_spike

X_train_dat = X_train
y_train_dat = y_train

X_valid_dat = X_valid
y_valid_dat = y_valid

print(X_train_dat.shape, y_train_dat.shape)
print(X_valid_dat.shape, y_valid_dat.shape)
X_train, y_train=get_spikes_with_history(X_train_dat, y_train_dat, bin_width_ms, DT)
X_valid, y_valid=get_spikes_with_history(X_valid_dat, y_valid_dat, bin_width_ms, DT)
X_train_original = X_train_dat
X_valid_original = X_valid_dat

if "SNN" in model_name:
    X_train = np.transpose(X_train, (0, 2, 1))
    X_valid = np.transpose(X_valid, (0, 2, 1))
    X_train_original = np.transpose(X_train, (0, 2, 1))
    X_valid_original = np.transpose(X_valid, (0, 2, 1))
    y_train = np.transpose(y_train, (0, 2, 1))
    y_valid = np.transpose(y_valid, (0, 2, 1))

X_train = torch.from_numpy(X_train).float()
X_valid = torch.from_numpy(X_valid).float()
y_train = torch.from_numpy(y_train).float()
y_valid = torch.from_numpy(y_valid).float()

X_train_original = torch.from_numpy(X_train_original).float()
X_valid_original = torch.from_numpy(X_valid_original).float()

train_data = TensorDataset(X_train, y_train)
train_loaders = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
valid_data = TensorDataset(X_valid, y_valid)
valid_loaders = DataLoader(valid_data, batch_size=BATCH_SIZE, shuffle=False)

# Train a single epoch
def process_one_epoch(model, loader, loss_fn, optimizer = None, train=True, profile=False):
    running_loss = 0.
    pred_total = []
    actual_total = []
    batch_idx_final = None
    # for batch_idx, (inputs, labels) in enumerate(tqdm.tqdm(loader)):
    for batch_idx, (inputs, labels) in enumerate(loader):
        batch_idx_final = batch_idx
        inputs = inputs.to(device)
        labels = labels.to(device)
        output = model(inputs, profile)

        output = output.to(device)

        loss = loss_fn(output, labels) # calculate loss

        if not train:
            output = output.cpu().detach().numpy()
            labels = labels.cpu().detach().numpy()
            if "SNN" in model_name:
                output = np.transpose(output, (0, 2, 1)).reshape(-1, 2)
                labels = np.transpose(labels, (0, 2, 1)).reshape(-1, 2)
            else:
                output = output.reshape(-1, 2)
                labels = labels.reshape(-1, 2)
            for pred_dat in output:
                pred_total.append(pred_dat)
            for actual_dat in labels:
                actual_total.append(actual_dat)

        if train:
            for opt in optimizer:
                if opt != None: opt.zero_grad()
            loss.backward()
            for opt in optimizer:
                if opt != None: opt.step()

        running_loss += loss.item()
        if profile: break
    
    if not train:
        return running_loss / (batch_idx_final + 1), get_R2(actual_total, pred_total)
    else:
        return running_loss / (batch_idx_final + 1)

# Initializing in a separate cell so we can easily add more epochs to the same run
from pathlib import Path
checkpoint_name = "./checkpoint/{}/{}/{}x".format(model_name, input_dim, fr_scale)
Path(checkpoint_name).mkdir(parents=True, exist_ok=True)

print('\n##### Bin Width : %d #####'%bin_width_ms)
if model_name == "SVMSNN" or model_name == "SVM":   
    if model_name == "SVMSNN":
        model = DecoderSVM(model_name = "SNN", input_dim = input_dim, bin_width_ms = bin_width_ms, DT = DT)
    else:
        model = DecoderSVM(model_name = "ANN", input_dim = input_dim, bin_width_ms = bin_width_ms, DT = DT)
elif model_name == "RNN" or model_name == "FC" or model_name == "RNNRELU" or model_name == "RNNTANH":
    if split == 1:
        model = DecoderDNN(model_name = model_name, input_dim = input_dim, h_dim = h_dim)
    else:
        model = DecoderSplitDNN(model_name = model_name, input_dim = input_dim, h_dim = h_dim, split = split)
elif "Slayer" in model_name and ("FCSNN" in model_name or "RNNSNN" in model_name):
    if split == 1:
        model = DecoderSlayerSNN(model_name = model_name, input_dim = input_dim, h_dim = h_dim, neuron_name = neuron_name, delay = delay, bin_width_ms = bin_width_ms, DT = DT)
    else:
        model = DecoderSplitSlayerSNN(model_name = model_name, input_dim = input_dim, h_dim = h_dim, neuron_name = neuron_name, delay = delay, bin_width_ms = bin_width_ms, DT = DT, split = split)
elif "Torch" in model_name and ("FCSNN" in model_name or "RNNSNN" in model_name):
    model = DecoderTorchSNN(model_name = model_name, input_dim = input_dim, h_dim = h_dim, neuron_name = neuron_name, bin_width_ms = bin_width_ms, DT = DT)
else:
    assert(0)

model.to(device)

loss_fn = nn.MSELoss()

optimizer1, optimizer2 = None, None

if "Slayer" in model_name:
    weight_params = [p for n, p in model.named_parameters() if n.find('weight') != -1]
    optimizer1 = torch.optim.Adam(weight_params, lr=LR_WEIGHT, maximize = False)
    delay_params = [p for n, p in model.named_parameters() if n.find('weight') == -1]
    #delay_params = [p for n, p in model.named_parameters() if n.find('delay') != -1]
    if delay_params:
        optimizer2 = torch.optim.Adam(delay_params, lr=LR_DELAY, maximize = False)
else:
    optimizer1 = torch.optim.Adam(model.parameters(), lr=LR_WEIGHT, maximize = False)

best_vloss = float('inf')

# epoch number to start from
if args.inference:
    # Fake inference to initiate delay
    model.train(False)
    avg_valid_loss, R2 = process_one_epoch(model, valid_loaders, loss_fn, train=False, profile=True)
    checkpoint = torch.load('{}/checkpoint_best_bin_{}_gaussian_{}.pt'.format(checkpoint_name, \
                                                                              bin_width_ms, \
                                                                              gaussian), \
                                                                             map_location=device)
    model.load_state_dict(checkpoint['model'])

    # Retrieve the dimension
    i_dim = model.input_dim
    if "SVM" in model_name:
        h_dim = 0
    else:
        h_dim = model.h_dim
    o_dim = model.o_dim

    # bci / bci neuron / compute neuron
    num_types = 3
    num_neurons = h_dim + o_dim
    num_external = i_dim

    # Initialize the parameter list
    if "SNN" in model_name:
        decay_g = []
        decay_v = []

    def parse_parameter(neuron, dim, type, decay_g = None, decay_v = None):
        decay_g += [(1 - float(neuron.current_decay)) for i in range(dim)]
        decay_v += [(1 - float(neuron.voltage_decay)) for i in range(dim)]

    if "SlayerFCSNN" == model_name:
        parse_parameter(model.First.neuron, h_dim, "Slayer", decay_g, decay_v)
        parse_parameter(model.FC.neuron, o_dim, "Slayer", decay_g, decay_v)
    elif "SVMSNN" == model_name:
        #parse_parameter(model.First_n, h_dim, "Torch", decay_g, decay_v)
        parse_parameter(model.last_layer, o_dim, "Torch", decay_g, decay_v)

    if "SNN" in model_name:
        neu_params = [{
            "r_ar_max" : 0,
            "decay_g" : float(decay_g[i]),
            "decay_v" : float(decay_v[i]),
            "E_0" : 0.,
            "E_L" : 0.,
            "model" : "DSRM0"
        } for i in range(h_dim + o_dim)]
    else:
        if "FC" == model_name or "RNNRELU" == model_name:
            neu_params = [{
                "r_ar_max" : 0,
                "E_0" : 0.,
                "model" : "RELU"
            } for i in range(h_dim)]
        elif "RNNTANH" == model_name:
            neu_params = [{
                "r_ar_max" : 0,
                "E_0" : 0.,
                "model" : "TANH"
            } for i in range(h_dim)]
        else:
            assert(0)
        neu_params += [{
            "r_ar_max" : 0,
            "E_0" : 0.,
            "model" : "OUTPUT"
        } for i in range(o_dim)]

    I_list = []
    threshold_list = []

    def parse_states(dim, I_list, threshold_list, layer, layer_type, spike, neuron = None):
        bias = False
        for n, p in layer.named_parameters():
            if n.find('bias') != -1: bias = True
        if bias == True:
            if "RNN" in layer_type:
                I_list += [float(layer.bias_ih_l0[ind] + layer.bias_hh_l0[ind]) for ind in range(dim)]
            else:
                I_list += [float(layer.bias[ind]) for ind in range(dim)]
        else: I_list += [0 for _ in range(dim)]
        if neuron:
            if spike: threshold_list += [float(neuron.threshold) for _ in range(dim)]
            else:     threshold_list += [float('inf') for _ in range(dim)]
        else:
            threshold_list += [float('inf') for _ in range(dim)]

    state_dict = {}
    if "SNN" in model_name:
        if "Slayer" in model_name:
            parse_states(h_dim, I_list, threshold_list, model.First.synapse, "SNN", True, model.First.neuron)
            parse_states(o_dim, I_list, threshold_list, model.FC.synapse, "SNN", False, model.FC.neuron)
            state_dict = {"neuron_state" : [{
                          "k_minus_t" : 0,
                          "v_t" : 0,
                          "g_t" : 0,
                          "threshold" : threshold_list[gid],
                          "I_t" : I_list[gid],
                          "refr" : 1 if gid < h_dim else 2, } | neu_params[gid]
                          for gid in range(h_dim + o_dim)]}
        elif "SVM" in model_name:
            parse_states(o_dim, I_list, threshold_list, model.First, "FC", False, model.Frist.neuron)
            state_dict = {"neuron_state" : [{
                          "k_minus_t" : 0,
                          "v_t" : 0,
                          "g_t" : 0,
                          "threshold" : threshold_list[gid],
                          "I_t" : I_list[gid],
                          "refr" : 1, } | neu_params[gid]
                          for gid in range(o_dim)]}
        else:
            assert(0)
    else:
        if "SVM" == model_name:
            parse_states(o_dim, I_list, threshold_list, model.FC, "FC", False)

            state_dict = {'neuron_state' : [{
                "k_minus_t" : 0,
                "h_t" : 0,
                "I_t" : I_list[gid],
                "refr" : 1, } | neu_params[gid]
                for gid in range(o_dim)]}
        elif "RNN" in model_name or "FC" == model_name:
            parse_states(h_dim, I_list, threshold_list, model.First, model_name, False)
            parse_states(o_dim, I_list, threshold_list, model.FC, "FC", False)

            state_dict = {'neuron_state' : [{
                "k_minus_t" : 0,
                "h_t" : 0,
                "I_t" : I_list[gid],
                "refr" : 1 if gid < h_dim else 2, } | neu_params[gid]
                for gid in range(h_dim + o_dim)]}
        else:
            assert(0)

    dt = 1.0

    ######################################################################

    # Normal, External
    conn_dat = [[[] for _ in range(num_types)] for _ in range(num_types)]
    def parse_connection(conn_dat, input_dim, output_dim, input_offset, output_offset, weight = None, delay = None):
        if delay == None: delay = [0 for _ in range(input_dim)]
        if weight != None:
            for i_ind in range(input_dim):
                for o_ind in range(output_dim):
                    wgt_dat = weight[i_ind, o_ind]
                    wgt_shape = list(wgt_dat.shape)
                    if len(wgt_shape) == 0 and wgt_dat != 0:
                        wgt_dat = float(wgt_dat)
                        #load = {'weight': wgt_dat, 'delay' : int(delay[i_ind]) + 1}
                        if "SNN" in model_name:
                            load = [('weight', wgt_dat, 16), ('delay', int(delay[i_ind]) + 1, 8)]
                        if "FC" in model_name:
                            load = [('weight', wgt_dat, 16), ('delay', int(delay[i_ind]) + 1, 1)]
                        conn = (i_ind + input_offset,
                                o_ind + output_offset, load)
                        conn_dat.append(conn)
                    elif len(wgt_shape) != 0:
                        wgt_dat = list(wgt_dat.detach().cpu().numpy())
                        #load = {'weight': wgt_dat, 'delay' : int(delay[i_ind]) + 1}
                        if "SNN" in model_name:
                            load = [('weight', wgt_dat, 16), ('delay', int(delay[i_ind]) + 1, 8)]
                        if "FC" in model_name:
                            load = [('weight', wgt_dat, 16), ('delay', int(delay[i_ind]) + 1, 1)]
                        conn = (i_ind + input_offset,
                                o_ind + output_offset, load)
                        conn_dat.append(conn)
        else:
            assert(input_dim == output_dim)
            assert(input_offset == output_offset)
            for i_ind in range(input_dim):
                conn = (i_ind + input_offset, i_ind + input_offset)
                conn_dat.append(conn)

    src_type_id = 2
    dst_type_id = 1
    conn_dat_temp = []
    parse_connection(conn_dat_temp, i_dim, i_dim, 0, 0)
    conn_dat[src_type_id][dst_type_id] = conn_dat_temp

    if "Slayer" in model_name:
        # Neuron-to-Neuron connection
        weight = model.FC.synapse.weight
        weight = torch.transpose(weight, 0, 1)
        weight = weight.view(weight.shape[0], -1)
        src_type_id = 0
        dst_type_id = 0
        conn_dat_temp = []
        parse_connection(conn_dat_temp, h_dim, o_dim, 0, h_dim, weight, model.First.delay.delay)
        conn_dat[src_type_id][dst_type_id] = conn_dat_temp
        # External Connection
        weight = model.First.synapse.weight
        weight = torch.transpose(weight, 0, 1)
        weight = weight.view(weight.shape[0], -1)
        src_type_id = 1
        dst_type_id = 0
        conn_dat_temp = []
        parse_connection(conn_dat_temp, i_dim, h_dim, 0, 0, weight)
        conn_dat[src_type_id][dst_type_id] = conn_dat_temp
    elif "SVMSNN" == model_name:
        weight = model.First.weight
        weight = torch.transpose(weight, 0, 1)
        weight = weight.view(weight.shape[0], -1)
        src_type_id = 1
        dst_type_id = 0
        conn_dat_temp = []
        parse_connection(conn_dat_temp, i_dim, o_dim, 0, 0, weight)
        conn_dat[src_type_id][dst_type_id] = conn_dat_temp
    elif "FC" == model_name:
        # Neuron-to-Neuron connection
        weight = model.FC.weight
        weight = torch.transpose(weight, 0, 1)
        weight = weight.view(weight.shape[0], -1)
        src_type_id = 0
        dst_type_id = 0
        conn_dat_temp = []
        parse_connection(conn_dat_temp, h_dim, o_dim, 0, h_dim, weight)
        conn_dat[src_type_id][dst_type_id] = conn_dat_temp
        # External Connection
        weight = model.First.weight
        weight = torch.transpose(weight, 0, 1)
        weight = weight.view(weight.shape[0], -1)
        src_type_id = 1
        dst_type_id = 0
        conn_dat_temp = []
        parse_connection(conn_dat_temp, i_dim, h_dim, 0, 0, weight)
        conn_dat[src_type_id][dst_type_id] = conn_dat_temp
    elif "SVM" == model_name:
        # External Connection
        weight = model.First.weight
        weight = torch.transpose(weight, 0, 1)
        weight = weight.view(weight.shape[0], -1)
        src_type_id = 1
        dst_type_id = 0
        conn_dat_temp = []
        parse_connection(conn_dat_temp, i_dim, o_dim, 0, 0, weight)
        conn_dat[src_type_id][dst_type_id] = conn_dat_temp
    elif "RNN" in model_name:
        # Neuron-to-Neuron connection
        # Parse self connection in the RNN layer
        weight = model.First.weight_hh_l0
        weight = torch.transpose(weight, 0, 1)
        weight = weight.view(weight.shape[0], -1)
        src_type_id = 0
        dst_type_id = 0
        conn_dat_temp = []
        parse_connection(conn_dat_temp, h_dim, h_dim, 0, 0, weight)
        # Parse FC in the last layer
        weight = model.FC.weight
        weight = torch.transpose(weight, 0, 1)
        weight = weight.view(weight.shape[0], -1)
        parse_connection(conn_dat_temp, h_dim, o_dim, 0, h_dim, weight)
        # Sort the list
        conn_dat_temp.sort(key=lambda a: a[0])
        conn_dat[src_type_id][dst_type_id] = conn_dat_temp
        # Parse forward connection in the RNN layer
        weight = model.First.weight_ih_l0
        weight = torch.transpose(weight, 0, 1)
        weight = weight.view(weight.shape[0], -1)
        src_type_id = 1
        dst_type_id = 0
        conn_dat_temp = []
        parse_connection(conn_dat_temp, i_dim, h_dim, 0, 0, weight)
        conn_dat[src_type_id][dst_type_id] = conn_dat_temp


    # Initialize the input data

    if not "SNN" in model_name:
        external_dat = X_valid_original[:NUM_DATA].numpy()
        external_dat = external_dat.reshape((-1, input_dim))
    else:
        external_dat = np.transpose(X_valid_original[:NUM_DATA].numpy(), (0, 2, 1))
        external_dat = external_dat.reshape((-1, input_dim))

    external_spikes = []
    num_spikes = 0
    tot_spikes = 0
    for timestep in range(external_dat.shape[0]):
        spike_list = external_dat[timestep]
        for neuron_id in range(len(spike_list)):
            # Append the spike to the external spikes list
            tot_spikes += 1
            if spike_list[neuron_id] != 0:
                external_spikes.append((timestep // 4, neuron_id, spike_list[neuron_id]))
                num_spikes += 1

    duration = external_dat.shape[0]
    state_dict['bin_width'] = bin_width_ms

    init_simulation(simtime = duration,
                    dt = dt, 
                    num_neurons = num_neurons,
                    num_external = num_external,
                    result_path = result_path)
    create_neurons(initial_states = state_dict)
    create_connections(conn_dat = conn_dat, num_types = num_types)

    create_external_stimulus(num_external = i_dim, external_stim = external_spikes)
    
    end_simulation()


    avg_valid_loss, R2 = process_one_epoch(model, valid_loaders, loss_fn, train=False, profile=True)

    ## Prune the NN if needed
    valid_loaders = DataLoader(valid_data, batch_size=64, shuffle=False)
    avg_valid_loss, R2 = process_one_epoch(model, valid_loaders, loss_fn, train=False, profile=False)
    print('Valid loss: {:.3f}, R2: {:.3f}'.format(avg_valid_loss, R2[0]))
    if "SNN" in model_name:
        parameter = (
            (model.First.synapse, "weight"),
            (model.FC.synapse, "weight"),
        )
    elif "RNN" in model_name:
        parameter = (
            (model.First, "weight_ih_l0"),
            (model.First, "weight_hh_l0"),
            (model.FC, "weight"),
        )
    else:
        parameter = (
            (model.First, "weight"),
            (model.FC, "weight"),
        )
    prune.global_unstructured(
        parameter,
        pruning_method=prune.L1Unstructured,
        amount=0.3,
    )
    avg_valid_loss, R2 = process_one_epoch(model, valid_loaders, loss_fn, train=False, profile=False)
    print('Valid loss: {:.3f}, R2: {:.3f}'.format(avg_valid_loss, R2[0]))
    # assert(0)
    print()
else:
    start_epoch = 0
    if start_epoch != 0:
        checkpoint = torch.load('{}/checkpoint_bin_{}_ep_{}.pt'.format(checkpoint_name, bin_width_ms, start_epoch), map_location=device)
        model.load_state_dict(checkpoint['model'])
        start_epoch += 1
    optimizer = [optimizer1, optimizer2]
    for epoch in range(start_epoch, EPOCHS_WEIGHT):
        #print(delay_params)
        print('EPOCH %d:'%epoch)
        # Make sure gradient tracking is on, and do a pass over the data
        model.train(True)

        # Heuristically stop training the delay after 30 epochs
        if (epoch + 1) % LR_STEP == 0:
            for opt in optimizer:
                if opt != None: opt.param_groups[0]['lr'] *= 0.1
            #if optimizer[1] != None:
            #    optimizer[1].param_groups[0]['lr'] = 0
        
        avg_train_loss = process_one_epoch(model, train_loaders, loss_fn, optimizer, train=True)
        
        print("Learning Rate", end = "")
        for opt in optimizer:
            if opt == None: continue
            print(' {}'.format(opt.param_groups[0]['lr']), end = "")
        print()
        print('Train loss: {:.3f}'.format(avg_train_loss))
        
        model.train(False)
        
        avg_valid_loss, R2 = process_one_epoch(model, valid_loaders, loss_fn, train=False)
            
        print('Valid loss: {:.3f}, R2: {:.3f}'.format(avg_valid_loss, R2[0]))

        # Track best performance, and save the model's state
        if avg_valid_loss < best_vloss:
            print('BEST')
            best_vloss = avg_valid_loss
            torch.save({'epochs' : epoch,
                        'model' : model.state_dict(),
                        'optimizer1' : optimizer1.state_dict(),
                        'optimizer2' : optimizer2.state_dict() if optimizer2 != None else optimizer2,
                        'epoch' : epoch,
                        'loss' : avg_valid_loss,
                       }, '{}/checkpoint_best_bin_{}_gaussian_{}.pt'.format(checkpoint_name, \
                                                                            bin_width_ms, \
                                                                            gaussian))
        print()
