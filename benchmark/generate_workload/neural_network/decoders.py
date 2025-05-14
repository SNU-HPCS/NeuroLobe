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
        vel_data_binned = np.empty([ms_duration - bin_width_ms, num_vel])

        for j in range(ms_duration):
            if j >= bin_width_ms:
                neural_data_binned[j-bin_width_ms] = np.sum(neural_data_temp[j-DT:j,:], axis = 0)
                vel_data_binned[j-bin_width_ms] = vel_data_temp[j]

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

    def forward(self, inputs, profile=False):
        
        self.voltage_first = []
        self.voltage_last = []

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

        return output
