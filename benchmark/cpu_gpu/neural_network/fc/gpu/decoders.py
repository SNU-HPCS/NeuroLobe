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

    #X = np.reshape(X, (1, num_examples * ms_duration // bin_width_ms, bin_width_ms, num_neurons))
    #Y = np.reshape(Y, (1, num_examples * ms_duration // bin_width_ms, 1, num_vel))
    #X = np.reshape(X, (1, -1, num_neurons))
    #Y = np.reshape(Y, (1, -1, num_vel))

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

        # self.First = None
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
        self.elapsed_time = []
        self.process_iter = 0

        self.bin_width = 120

    def forward(self, t_offset, inputs, spike_buf, spike_sum, profile=False, measure=False, single_timestep=True):

        if measure:
            if single_timestep == False:
                # assume input shape : (chunk_width, 1600)
                # for now, chunk width = bin width = 120
                _, chunk_width, input_dim = inputs.shape
                self.process_iter += chunk_width
                fc_out = torch.empty(chunk_width, self.o_dim)
                time_sum = 0
                for i in range(chunk_width):
                    t = t_offset + i
                    start_time = time.time()

                    # dynamic binning
                    # print("spike sum", spike_sum.shape)
                    # print("spike buf", spike_buf.shape)
                    # print("t", t)
                    # print("bin_width", bin_width)

                    # Dynamic binning
                    spike_sum -= spike_buf[t % self.bin_width, :]
                    spike_buf[t % self.bin_width, :] = 0

                    spike_buf[t % self.bin_width, :] += inputs[0, i, :]
                    spike_sum += inputs[0, i, :]

                    output = self.First(spike_sum)
                    #output = self.First(inputs[0,i])
                    output = F.relu(output)
                    output = self.FC(output)

                    end_time = time.time()

                    fc_out[i, :] = output

                    time_sum += end_time - start_time
                # for i in range(inputs.shape[1]):
                #     if self.model_name == "RNN" or self.model_name == "LSTM" or self.model_name == "RNNRELU" or self.model_name == "RNNTANH":
                #         self.First.flatten_parameters()
                #         output, hidden = self.First(inputs[:,i], None)
                #         output = output
                #     else:
                #         output = self.First(inputs[:,i])
                #         output = F.relu(output)

                #     output = self.FC(output).reshape(1, 1, -1)
                #     fc_out[:,i] = output
                #output = fc_out
                output = torch.mean(fc_out, 0)
                self.elapsed_time.append(time_sum)
            else:
                assert(0)
        else:
            output = self.First(inputs[0, 0])
            output = F.relu(output)
            output = self.FC(output)

            # else:
            #     self.process_iter += 1
            #     # fc_out = torch.empty(inputs.shape[0], inputs.shape[1], self.o_dim)
            #     start_time = timeit.default_timer()
            #     if self.model_name == "RNN" or self.model_name == "LSTM" or self.model_name == "RNNRELU" or self.model_name == "RNNTANH":
            #         self.First.flatten_parameters()
            #         output, hidden = self.First(inputs, None)
            #         output = output
            #     else:
            #         # inference
            #         output = self.First(inputs)
            #         output = F.relu(output)

            #     output = self.FC(output)
            #         # output = self.FC(output).reshape(1, 1, -1)
            #         # fc_out[:,i] = output
            #     end_time = timeit.default_timer()
            #     # output = fc_out
            #     self.elapsed_time.append(end_time - start_time)


        # else:
            # if self.model_name == "RNN" or self.model_name == "LSTM" or self.model_name == "RNNRELU" or self.model_name == "RNNTANH":
            #     self.First.flatten_parameters()
            #     output, hidden = self.First(inputs, None)
            #     output = output
            # else:
            #     output = self.First(inputs)
            #     output = F.relu(output)
            # output = self.FC(output)

        # self.voltage_first = []
        # self.voltage_last = []

        # if profile:
            # voltage_temp = output.detach().cpu().numpy()
            # voltage_temp = np.reshape(voltage_temp, (-1, voltage_temp.shape[-1]))
            # self.voltage_first = voltage_temp

        return output, spike_buf, spike_sum

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
            #self.FCs.append(nn.Linear(self.h_dim, self.o_dim))
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
        # self.saved_state_First = {'V_t':0, 'I_t':0}
        # self.saved_state_FC = {'V_t':0, 'I_t':0}


    def forward(self, inputs, profile=False, measure=False, batch_idx=0):
        if measure:
            self.process_iter += inputs.shape[1]
            fc_out = torch.empty(inputs.shape[0], self.o_dim, inputs.shape[1])
            time_sum = 0
            # print("input shape", inputs.shape)
            for i in range(inputs.shape[1]):
                start_time = time.time()
                # retrieve all the spikes
                spike = self.First(inputs[:,i])
                # permute the previous output spike
                output = self.FC(spike)
                # output = output.view(output.shape[0], output.shape[1], -1, (self.bin_width_ms // self.DT))
                # output = torch.mean(output, -1)
                # fc_out[:,:,i:i+1] = output
                end_time = time.time()
                output = torch.mean(output, -1) # (1, 2)
                fc_out[:, :, i] = output
                time_sum += end_time - start_time
            output = fc_out
            # end_time = time.time()
            self.elapsed_time.append(time_sum)
        else:
            inputs = inputs[:,0]
            spike = self.First(inputs)
            output = self.FC(spike)
            output = output.view(output.shape[0], output.shape[1], -1, (self.bin_width_ms // self.DT))
            output = torch.mean(output, -1)

        # output = torch.transpose(output, 1, 2) # FIXME
        if profile:
            for b in range(output.shape[0]):
                for t in range(output.shape[2]):
                    self.voltage_list.append(output[b,:,t])
        return output
        # if measure:
            # 1. for loop
            # self.process_iter += inputs.shape[1]
            # fc_out = torch.empty(inputs.shape[0], self.o_dim, inputs.shape[1]) # (1, 2, 30)
            # start_time = time.time()
            # for i in range(inputs.shape[1]): # 30
            #     if i != 0:
            #         self.First.neuron.current_state = self.saved_state_First['I_t']
            #         self.First.neuron.voltage_state = self.saved_state_First['V_t']
            #         self.FC.neuron.current_state = self.saved_state_FC['I_t']
            #         self.FC.neuron.voltage_state = self.saved_state_FC['V_t']
            #     # retrieve all the spikes
            #     # print("input shape", inputs.shape)
            #     spike = self.First(inputs[:,i])
            #     print("input shape2 ", inputs[:,i].shape)
            #     # permute the previous output spike
            #     output = self.FC(spike)
            #     # print("output shape", output.shape)
            #     # output = output.view(output.shape[0], output.shape[1], -1, (self.bin_width_ms // self.DT))
            #     # print("output shape", output.shape)
            #     output = torch.mean(output, -1) # shape (1, 2, 1)
            #     # print("output shape", output.shape)
            #     # print("fc out shape", fc_out[:,:,i].shape)
            #     fc_out[:,:,i] = output
            #     self.saved_state_First['V_t'] = self.First.neuron.debug_voltage[..., -1].clone()
            #     self.saved_state_First['I_t'] = self.First.neuron.debug_current[..., -1].clone()
            #     self.saved_state_FC['V_t'] = self.FC.neuron.debug_voltage[..., -1].clone()
            #     self.saved_state_FC['I_t'] = self.FC.neuron.debug_current[..., -1].clone()
            # # self.saved_state_First['V_t'] = torch.zeros_like(self.saved_state_First['V_t'])
            # # self.saved_state_First['I_t'] = torch.zeros_like(self.saved_state_First['I_t'])
            # # self.saved_state_FC['V_t'] = torch.zeros_like(self.saved_state_FC['V_t'])
            # # self.saved_state_FC['I_t'] = torch.zeros_like(self.saved_state_FC['I_t'])
            # output = fc_out
            # end_time = time.time()
            # self.elapsed_time.append(end_time - start_time)
            # output = output.transpose(1, 2)
            # output = output.view(output.shape[0], output.shape[1], output.shape[2], 1)



            # 2. loop through single timestep
            # load previous timestep's state - for First, FC
            # TODO

            # print('initial current state', self.First.neuron.current_state)

            # if self.process_iter != 0:
            #     self.First.neuron.current_state = self.saved_state_First['I_t']
            #     self.First.neuron.voltage_state = self.saved_state_First['V_t']
            #     self.FC.neuron.current_state = self.saved_state_FC['I_t']
            #     self.FC.neuron.voltage_state = self.saved_state_FC['V_t']

            # # duration = 30
            # # if self.process_iter % duration != 0:
            # #     self.First.neuron.current_state = self.saved_state_First['I_t']
            # #     self.First.neuron.voltage_state = self.saved_state_First['V_t']
            # #     self.FC.neuron.current_state = self.saved_state_FC['I_t']
            # #     self.FC.neuron.voltage_state = self.saved_state_FC['V_t']
            # #     # print("loaded current state:", self.First.neuron.current_state)
            # #     # print("loaded voltage state:", self.First.neuron.voltage_state)
            # #     # self.First.neuron.current_state = torch.zeros_like(self.First.neuron.current_state)
            # #     # self.First.neuron.voltage_state = torch.zeros_like(self.First.neuron.voltage_state)
            # #     # self.FC.neuron.current_state = torch.zeros_like(self.FC.neuron.current_state)
            # #     # self.FC.neuron.voltage_state = torch.zeros_like(self.FC.neuron.voltage_state)
            # # else:
            # #     self.First.neuron.current_state = torch.zeros_like(self.First.neuron.current_state)
            # #     self.First.neuron.voltage_state = torch.zeros_like(self.First.neuron.voltage_state)
            # #     self.FC.neuron.current_state = torch.zeros_like(self.FC.neuron.current_state)
            # #     self.FC.neuron.voltage_state = torch.zeros_like(self.FC.neuron.voltage_state)
            # #     print("initial current state:", self.First.neuron.current_state)
            # #     print("initial voltage state:", self.First.neuron.voltage_state)
            # # print('First current state: ', self.First.neuron.current_state.shape)

            # self.process_iter += 1
            # # print('debug voltage', self.First.neuron.debug_voltage.shape, self.First.neuron.debug_voltage)
            # # print('debug current', self.First.neuron.debug_current.shape, self.First.neuron.debug_current)

            # start_time = time.time()

            # # retrieve all the spikes
            # spike = self.First(inputs)
            # # permute the previous output spike
            # output = self.FC(spike)
            # output = output.view(output.shape[0], output.shape[1], -1, (self.bin_width_ms // self.DT))
            # output = torch.mean(output, -1)

            # end_time = time.time()
            # # save current timestep's state
            # # TODO
            # self.saved_state_First['V_t'] = self.First.neuron.debug_voltage[..., -1].clone()
            # self.saved_state_First['I_t'] = self.First.neuron.debug_current[..., -1].clone()
            # self.saved_state_FC['V_t'] = self.FC.neuron.debug_voltage[..., -1].clone()
            # self.saved_state_FC['I_t'] = self.FC.neuron.debug_current[..., -1].clone()
            # # print('saved state first', self.saved_state_First['V_t'].shape)
            # # print('debug voltage?', self.First.neuron.debug_voltage.shape)

            # self.elapsed_time.append(end_time - start_time)

            # 3. original code
            # self.process_iter += inputs.shape[1]
            # fc_out = torch.empty(inputs.shape[0], self.o_dim, inputs.shape[1]) # (1, 2, 30)
            # start_time = time.time()
            # for i in range(inputs.shape[1]):
            #     # retrieve all the spikes
            #     spike = self.First(inputs[:,i])
            #     # permute the previous output spike
            #     output = self.FC(spike)
            #     output = output.view(output.shape[0], output.shape[1], -1, (self.bin_width_ms // self.DT))
            #     output = torch.mean(output, -1)
            #     fc_out[:,:,i:i+1] = output
            # output = fc_out
            # print("output shape", output.shape)
            # end_time = time.time()
            # self.elapsed_time.append(end_time - start_time)

        # else:
            # inputs = inputs[:,0]
            # spike = self.First(inputs)
            # output = self.FC(spike)
            # output = output.view(output.shape[0], output.shape[1], -1, (self.bin_width_ms // self.DT))
            # output = torch.mean(output, -1)
        # spike = self.First(inputs)
        # output = self.FC(spike)
        # output = output.view(output.shape[0], output.shape[1], -1, (self.bin_width_ms // self.DT))

        # output = torch.mean(output, -1)

        # output = torch.mean(output, -1)

        # output = torch.transpose(output, 1, 2) # FIXME
        # if profile:
        #     for b in range(output.shape[0]):
        #         for t in range(output.shape[2]):
        #             self.voltage_list.append(output[b,:,t])
        return output
