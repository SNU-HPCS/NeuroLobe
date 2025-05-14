import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader, Dataset

import os, sys
import glob
import zipfile
import numpy as np
import time
import timeit


###$$ GET_SPIKES_WITH_HISTORY #####
def get_spikes_with_history(neural_data, vel_data, bin_width_ms, DT):

    num_neurons     = neural_data.shape[2] #Number of neurons
    ms_duration     = neural_data.shape[1]
    num_examples    = neural_data.shape[0]
    num_vel         = vel_data.shape[2]

    # Depends on the SNN or RNN
    X=np.empty([num_examples, ms_duration // DT, num_neurons])
    Y=np.empty([num_examples, ms_duration // bin_width_ms, num_vel])

    X[:] = np.NaN
    Y[:] = np.NaN
    num_bins = ms_duration // bin_width_ms
    for i in range(num_examples): #The first ms_before and last ms_after ms don't get filled in
        # retrieve the data for the whole duration
        neural_data_temp=neural_data[i,:,:]
        vel_data_temp=vel_data[i,:,:]

        neural_data_temp = np.reshape(neural_data_temp, (-1, DT, num_neurons))
        neural_data_temp = np.sum(neural_data_temp, axis=1)
        vel_data_temp = np.reshape(vel_data_temp, (-1, bin_width_ms, num_vel))
        vel_data_temp = np.mean(vel_data_temp, axis=1)

        X[i,:,:]=neural_data_temp
        Y[i,:,:]=vel_data_temp

    return X, Y

def get_neuron(name):
    name = "custom_slayer.block." + name
    components = name.split('.')
    mod = __import__(components[0])
    #mod = components[0]
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod

class DecoderDNN(nn.Module):
    def __init__(self, model_name = "FC", input_dim=182, h_dim=256, output_dim=2):
        super(DecoderDNN, self).__init__()

        self.model_name = model_name
        self.input_dim = input_dim
        self.h_dim = h_dim
        self.o_dim = output_dim

        if model_name == "FC":
            self.First = nn.Linear(input_dim, self.h_dim)
        else: assert(0)
        self.FC = nn.Linear(self.h_dim, self.o_dim)

        self.voltage_first = None
        self.voltage_last = None
        self.elapsed_time = []
        self.process_iter = 0

        self.bin_width = 120

    def forward(self, t_offset, inputs, spike_buf, spike_sum, profile=False, measure=False, single_timestep=True):

        if measure:
            if single_timestep == False:
                _, chunk_width, input_dim = inputs.shape
                self.process_iter += chunk_width
                fc_out = torch.empty(chunk_width, self.o_dim)
                time_sum = 0
                for i in range(chunk_width):
                    t = t_offset + i
                    start_time = time.time()
                    # Dynamic binning
                    spike_sum -= spike_buf[t % self.bin_width, :]
                    spike_buf[t % self.bin_width, :] = 0

                    spike_buf[t % self.bin_width, :] += inputs[0, i, :]
                    spike_sum += inputs[0, i, :]

                    output = self.First(spike_sum)
                    output = F.relu(output)
                    output = self.FC(output)

                    end_time = time.time()

                    fc_out[i, :] = output

                    time_sum += end_time - start_time
                output = torch.mean(fc_out, 0)
                self.elapsed_time.append(time_sum)
            else:
                assert(0)
        else:
            output = self.First(inputs[0, 0])
            output = F.relu(output)
            output = self.FC(output)

        return output, spike_buf, spike_sum

class DecoderSlayerSNN(nn.Module):
    def __init__(self, model_name = "SlayerFCSNN", input_dim=182, h_dim=256, output_dim=2, neuron_name = "cuba", delay = True, bin_width_ms = 1, DT = 1):
        super(DecoderSlayerSNN, self).__init__()

        self.model_name = model_name
        self.input_dim = input_dim
        self.neuron_name = neuron_name
        self.h_dim = h_dim
        self.o_dim = output_dim

        self.bin_width_ms = bin_width_ms
        self.DT = DT

        neuron = get_neuron(neuron_name)

        neuron_params1 = {}
        neuron_params1['threshold']         = 1.0
        neuron_params1['tau_grad']          = 1.0
        neuron_params1['scale_grad']        = 1.0
        neuron_params1['requires_grad']     = False
        neuron_params1['voltage_decay']     = 0.05
        neuron_params1['current_decay']     = 0.05

        if self.model_name == "SlayerFCSNN":
            self.First = neuron.Dense(neuron_params1, input_dim, self.h_dim, weight_norm = False, delay = delay, delay_shift = False)
        elif self.model_name == "SlayerRNNSNN":
            self.First = neuron.Recurrent(neuron_params, input_dim, self.h_dim, weight_norm = False, delay = delay, bias = False, delay_shift = False)
        else: assert(0)

        neuron_params2 = {}
        neuron_params2['threshold']         = 1.0
        neuron_params2['tau_grad']          = 1.0
        neuron_params2['scale_grad']        = 1.0
        neuron_params2['requires_grad']     = False
        neuron_params2['voltage_decay']     = 0.8
        neuron_params2['current_decay']     = 0.8
        neuron_params2['no_spike']  = True

        self.FC = neuron.Dense(neuron_params2, self.h_dim, self.o_dim, weight_norm = False, delay = False, delay_shift = False)
        self.voltage_list = []
        self.elapsed_time = []
        self.process_iter = 0

    def forward(self, inputs, profile=False, measure=False, batch_idx=0):
        if measure:
            self.process_iter += inputs.shape[1]
            fc_out = torch.empty(inputs.shape[0], self.o_dim, inputs.shape[1])
            time_sum = 0
            for i in range(inputs.shape[1]):
                start_time = time.time()
                # retrieve all the spikes
                spike = self.First(inputs[:,i])
                # permute the previous output spike
                output = self.FC(spike)
                end_time = time.time()
                output = torch.mean(output, -1) # (1, 2)
                fc_out[:, :, i] = output
                time_sum += end_time - start_time
            output = fc_out
            self.elapsed_time.append(time_sum)
        else:
            inputs = inputs[:,0]
            spike = self.First(inputs)
            output = self.FC(spike)
            output = output.view(output.shape[0], output.shape[1], -1, (self.bin_width_ms // self.DT))
            output = torch.mean(output, -1)

        if profile:
            for b in range(output.shape[0]):
                for t in range(output.shape[2]):
                    self.voltage_list.append(output[b,:,t])
        return output
