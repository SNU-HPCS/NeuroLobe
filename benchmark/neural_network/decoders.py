import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader, Dataset

import os, sys
import glob
import zipfile
import h5py
import numpy as np

# import slayer from lava-dl
#import custom_slayer as slayer

def get_R2(y_test,y_test_pred):

    """
    Function to get R2

    Parameters
    ----------
    y_test - the true outputs (a matrix of size number of examples x number of outputs)
    y_test_pred - the predicted outputs (a matrix of size number of examples x number of outputs)

    Returns
    -------
    R2_array: An array of R2s for each output
    """

    #y_test_flat = y_test.flatten()
    #y_test_pred_flat = y_test_pred.flatten()
    #y_mean=np.mean(y_test_flat)
    #R2_VAL=1-np.sum((y_test_pred_flat-y_test_flat)**2)/np.sum((y_test_flat-y_mean)**2)
    y_test = np.asarray(y_test)
    y_test_pred = np.asarray(y_test_pred)

    R2_list=[] #Initialize a list that will contain the R2s for all the outputs
    R2_mean = 0
    for i in range(y_test.shape[1]): #Loop through outputs
        #Compute R2 for each output
        y_mean=np.mean(y_test[:,i])
        R2=1-np.sum((y_test_pred[:,i]-y_test[:,i])**2)/np.sum((y_test[:,i]-y_mean)**2)
        R2_list.append(R2) #Append R2 of this output to the list
        R2_mean += R2
    R2_array=np.array(R2_list)
    R2_mean /= y_test.shape[1]
    return R2_mean, R2_array #Return an array of R2s

###$$ GET_SPIKES_WITH_HISTORY #####
def get_spikes_with_history(neural_data, vel_data, bin_width_ms, DT):

    num_neurons     = neural_data.shape[2] #Number of neurons
    ms_duration     = neural_data.shape[1]
    num_examples    = neural_data.shape[0]
    num_vel         = vel_data.shape[2]
    
    #X=np.empty([num_examples * num_subexamples,ms_surrounding // DT,num_neurons])
    #Y=np.empty([num_examples * num_subexamples,num_vel])

    # Depends on the SNN or RNN
    X=np.empty([num_examples, ms_duration - bin_width_ms, num_neurons])
    Y=np.empty([num_examples, ms_duration - bin_width_ms, num_vel])
    
    X[:] = np.NaN
    Y[:] = np.NaN
    num_bins = ms_duration // bin_width_ms
    for i in range(num_examples): #The first ms_before and last ms_after ms don't get filled in
        # retrieve the data for the whole duration
        neural_data_temp=neural_data[i,:,:]
        vel_data_temp=vel_data[i,:,:]

        neural_data_binned = np.empty([ms_duration - bin_width_ms, num_neurons])
            #np.reshape(neural_data_temp, (-1, DT, num_neurons))
        vel_data_binned = np.empty([ms_duration - bin_width_ms, num_vel])
            #np.reshape(vel_data_temp, (-1, bin_width_ms, num_vel))

        for j in range(ms_duration):
            if j >= bin_width_ms:
                neural_data_binned[j-bin_width_ms] = np.sum(neural_data_temp[j-DT:j,:], axis = 0)
                vel_data_binned[j-bin_width_ms] = vel_data_temp[j]
            #neural_data_temp = np.sum(neural_data_temp, axis=1)
            #vel_data_temp = np.mean(vel_data_temp, axis=1)

        X[i,:,:]=neural_data_binned
        Y[i,:,:]=vel_data_binned

    return X, Y

def get_neuron(name):
    name = "custom_slayer.block." + name
    components = name.split('.')
    mod = __import__(components[0])
    #mod = components[0]
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod

class DecoderSVM(nn.Module):
    def __init__(self, model_name = "ANN", input_dim=182, output_dim=2, bin_width_ms = 1, DT = 1):
        super(DecoderSVM, self).__init__()

        self.model_name = model_name
        self.input_dim = input_dim
        self.o_dim = output_dim

        self.bin_width_ms = bin_width_ms
        self.DT = DT

        self.First = nn.Linear(input_dim, self.o_dim)
        self.voltage_list_last = None
        self.current_list_last = None

        spike_grad = surrogate.fast_sigmoid() # surrogate gradient function
        beta = torch.rand(output_dim)
        alpha = torch.rand(output_dim)
        self.last_layer = snn.Synaptic(alpha = alpha, beta=beta, threshold=1.0, learn_alpha=True, learn_beta=True, spike_grad=spike_grad, reset_mechanism="none")

    def forward(self, inputs, profile=False):
        self.voltage_list_last = []
        self.current_list_last = []

        def init_layer(layer):
            states = list(layer.init_synaptic())
            states.insert(0, None)
            return states
        # Define
        def comp_layer(layer_prev, layer_next, spikes, states, is_first=False):
            curr = layer_prev(spikes)
            states = list(layer_next(curr, states[1], states[2]))
            return states

        # Empty lists to record outputs
        # if ann
        if self.model_name == "SNN":
            # initialize the states
            states = init_layer(self.last_layer)
            mem_rec = []
            input_spikes = torch.permute(inputs, (2, 0, 1))
            num_steps = input_spikes.shape[0]
            for step in range(num_steps):
                spikes = input_spikes[step, :, :]
                states = comp_layer(self.First, self.last_layer, spikes, states)
                    
                if (step + 1) % (self.bin_width_ms // self.DT) == 0:
                    mem_rec.append(states[-1])
                if profile:
                    self.current_list_last.append(states[1][0].detach().cpu().numpy())
                    self.voltage_list_last.append(states[2][0].detach().cpu().numpy())
            output = torch.stack(mem_rec)
            output = torch.transpose(output, 0, 1)
        
        elif self.model_name == "ANN":
            output = self.First(inputs)
        
        return output


class DecoderSplitDNN(nn.Module):
    def __init__(self, model_name = "FC", input_dim=182, h_dim=256, output_dim=2, split = 2):
        super(DecoderSplitDNN, self).__init__()

        self.model_name = model_name
        self.input_dim = input_dim
        self.h_dim = h_dim
        self.o_dim = output_dim
        self.split = split

        self.num_inputs = [(input_dim // split) + (i < input_dim % split) for i in range(split)]

        self.First_layers = nn.ModuleList()
        self.FCs = nn.ModuleList()
        for i in range(self.split):
            if model_name == "LSTM":
                First = nn.LSTM(input_size=self.num_inputs[i], hidden_size=self.h_dim, batch_first=True)
            elif model_name == "RNNRELU":
                First = nn.RNN(nonlinearity='relu', input_size=self.num_inputs[i], hidden_size=self.h_dim, batch_first=True)
            elif model_name == "RNNTANH":
                First = nn.RNN(nonlinearity='tanh', input_size=self.num_inputs[i], hidden_size=self.h_dim, batch_first=True)
            elif model_name == "FC":
                First = nn.Linear(self.num_inputs[i], self.h_dim)
            else: assert(0)
            self.First_layers.append(First)
            self.FCs.append(nn.Linear(self.h_dim, self.o_dim))
        self.Merge = nn.Linear(self.o_dim * split, self.o_dim)

    def forward(self, inputs, profile=False):

        input_list = []
        start_idx = 0
        for i in range(self.split):
            end_idx = start_idx + self.num_inputs[i]
            input_list.append(inputs[:,:,start_idx:end_idx])
            start_idx = end_idx

        output_list = []
        fc_out = None
        for i in range(self.split):
            if self.model_name == "RNN" or self.model_name == "LSTM" or self.model_name == "RNNRELU" or self.model_name == "RNNTANH":
                self.First_layers[i].flatten_parameters()
                output, hidden = self.First_layers[i](input_list[i], None)
            else:
                output = self.First_layers[i](input_list[i])
                output = F.relu(output)

            output = self.FCs[i](output)

            if fc_out != None: fc_out = torch.cat([fc_out, output], dim = 2)
            else: fc_out = output

        output = self.Merge(fc_out)

        return output

class DecoderDNN(nn.Module):
    def __init__(self, model_name = "FC", input_dim=182, h_dim=256, output_dim=2):
        super(DecoderDNN, self).__init__()

        self.model_name = model_name
        self.input_dim = input_dim
        self.h_dim = h_dim
        self.o_dim = output_dim

        if model_name == "LSTM":
            self.First = nn.LSTM(input_size=input_dim, hidden_size=self.h_dim, batch_first=True)
        elif model_name == "RNNRELU":
            self.First = nn.RNN(nonlinearity='relu', input_size=input_dim, hidden_size=self.h_dim, batch_first=True)
        elif model_name == "RNNTANH":
            self.First = nn.RNN(nonlinearity='tanh', input_size=input_dim, hidden_size=self.h_dim, batch_first=True)
        elif model_name == "FC":
            self.First = nn.Linear(input_dim, self.h_dim)
        else: assert(0)
        self.FC = nn.Linear(self.h_dim, self.o_dim)

        self.voltage_first = None
        self.voltage_last = None

    def forward(self, inputs, profile=False):
        
        self.voltage_first = []
        self.voltage_last = []

        if self.model_name == "RNN" or self.model_name == "LSTM" or self.model_name == "RNNRELU" or self.model_name == "RNNTANH":
            self.First.flatten_parameters()
            output, hidden = self.First(inputs, None)
            # Should add functions to count the number of operations in the LSTM

            output = output
        else:
            output = self.First(inputs)
            output = F.relu(output)

        if profile:
            voltage_temp = output.detach().cpu().numpy()
            voltage_temp = np.reshape(voltage_temp, (-1, voltage_temp.shape[-1]))
            self.voltage_first = voltage_temp

        output = self.FC(output)
        
        if profile:
            voltage_temp = output.detach().cpu().numpy()
            voltage_temp = np.reshape(voltage_temp, (-1, voltage_temp.shape[-1]))
            self.voltage_last = voltage_temp
        
        return output

class DecoderSplitSlayerSNN(nn.Module):
    def __init__(self, model_name = "SlayerFCSNN", input_dim=182, h_dim=256, output_dim=2, neuron_name = "cuba", delay = True, bin_width_ms = 1, DT = 1, split = 2):
        super(DecoderSplitSlayerSNN, self).__init__()

        self.model_name = model_name
        self.input_dim = input_dim
        self.neuron_name = neuron_name
        self.h_dim = h_dim
        self.o_dim = output_dim

        self.bin_width_ms = bin_width_ms
        self.DT = DT

        self.split = split
        self.num_inputs = [(input_dim // split) + (i < input_dim % split) for i in range(split)]

        neuron = get_neuron(neuron_name)

        neuron_params1 = {}
        neuron_params1['threshold']         = 1.0
        neuron_params1['tau_grad']          = 1.0
        neuron_params1['scale_grad']        = 1.0
        neuron_params1['requires_grad']     = False
        neuron_params1['voltage_decay']     = 0.05
        neuron_params1['current_decay']     = 0.05

        neuron_params2 = {}
        neuron_params2['threshold']         = 1.0
        neuron_params2['tau_grad']          = 1.0
        neuron_params2['scale_grad']        = 1.0
        neuron_params2['requires_grad']     = False
        neuron_params2['voltage_decay']     = 0.8
        neuron_params2['current_decay']     = 0.8
        neuron_params2['no_spike']  = True

        self.First_layers = nn.ModuleList()
        self.FCs = nn.ModuleList()

        for i in range(self.split):
            if self.model_name == "SlayerFCSNN":
                First = neuron.Dense(neuron_params1, self.num_inputs[i], self.h_dim, weight_norm = False, delay = delay, delay_shift = False)
            elif self.model_name == "SlayerRNNSNN":
                First = neuron.Recurrent(neuron_params1, self.num_inputs[i], self.h_dim, weight_norm = False, delay = delay, delay_shift = False)
            else: assert(0)
            self.First_layers.append(First)
            FC = neuron.Dense(neuron_params2, self.h_dim, self.o_dim, weight_norm = False, delay = delay, delay_shift = False)
            self.FCs.append(FC)
        self.Merge = nn.Linear(self.o_dim * split, self.o_dim)

    def forward(self, spike, profile=False):
        spike_list = []
        start_idx = 0
        for i in range(self.split):
            end_idx = start_idx + self.num_inputs[i]
            spike_list.append(spike[:,start_idx:end_idx,:])
            start_idx = end_idx

        # retrieve all the spikes
        output = None
        for i in range(self.split):
            spike = self.First_layers[i](spike_list[i])
            #input_spike = torch.permute(spike, (2, 0, 1))
            voltage = self.FCs[i](spike)
            if output != None: output = torch.cat([output, voltage], dim = 1)
            else: output = voltage
        output = output.view(output.shape[0], output.shape[1], -1, (self.bin_width_ms // self.DT))
        output = torch.mean(output, -1)
        output = torch.transpose(output, 1, 2)
        output = self.Merge(output)

        return output


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
            self.First = neuron.Dense(neuron_params1, input_dim, self.h_dim, weight_norm = False, delay = delay, bias = False, delay_shift = False)
        else: assert(0)

        neuron_params2 = {}
        neuron_params2['threshold']         = 1.0
        neuron_params2['tau_grad']          = 1.0
        neuron_params2['scale_grad']        = 1.0
        neuron_params2['requires_grad']     = False
        neuron_params2['voltage_decay']     = 0.95
        neuron_params2['current_decay']     = 0.5
        neuron_params2['no_spike']  = True

        self.FC = neuron.Dense(neuron_params2, self.h_dim, self.o_dim, weight_norm = False, delay = False, bias = False, delay_shift = False)
        self.voltage_list = []

    def forward(self, spike, profile=False):

        # retrieve all the spikes
        spike = self.First(spike)

        # permute the previous output spike
        output = self.FC(spike)

        if profile:
            self.voltage_list = []
            for b in range(output.shape[0]):
                for t in range(output.shape[2]):
                    self.voltage_list.append(output[b,:,t])

        #output = output.view(output.shape[0], output.shape[1], -1, (self.bin_width_ms // self.DT))
        #output = torch.mean(output, -1)
        return output
