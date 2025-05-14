import sys
import numpy as np
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

from decoders import *

import tqdm
from pathlib import Path
import configparser

class CustomDataset(Dataset):
  def __init__(self, x_filename, y_filename):
    self.x_filename = x_filename
    self.y_filename = y_filename
    self.trial_num = np.load(self.x_filename, mmap_mode='r').shape[0]

  def __len__(self):
    return self.trial_num

  def __getitem__(self, idx):
    x_file = np.load(self.x_filename, mmap_mode='r')
    y_file = np.load(self.y_filename, mmap_mode='r')
    x = torch.FloatTensor(x_file[idx])
    y = torch.FloatTensor(y_file[idx])
    del x_file
    del y_file
    return x, y


def profile_wrapper():
# Set the seed
    torch.manual_seed(5)

# Set training arguments

    # Load simulation parameters
    properties = configparser.ConfigParser()
    properties.read('snn.params')

    dataset_dir = properties.get('general','dataset_dir')
    valid_dir = properties.get('general','valid_dir')
    GPU_ID = properties.getint('general','gpu_id')
    model_name = properties.get('general','model')
    neuron_name = properties.get('general','neuron')
    bin_width_ms = properties.getint('general','bin_width')

    EPOCHS_WEIGHT   = properties.getint('general','epochs_weight')
    EPOCHS_DELAY    = properties.getint('general','epochs_delay')
    LR_WEIGHT       = float(properties.get('general','lr_weight'))
    LR_DELAY        = float(properties.get('general','lr_delay'))
    LR_STEP         = properties.getint('general','lr_step')
    delay           = bool(properties.getint('general','delay'))
    h_dim           = properties.getint('general','h_dim')

    original_num    = properties.getint('general','original_num')
    target_cell_num = properties.getint('general','target_cell_num')
    NUM_DATA        = properties.getint('general','num_data')

    inference       = True

# Set the GPU configuration (Multi GPU support will be added soon
    if GPU_ID == -1:
        device = torch.device('cpu')
    else:
        device = torch.device('cuda:' + str(GPU_ID) if torch.cuda.is_available() else 'cpu')

# Set the Dataloader configuration
    print("using device : {}".format(device))

    BATCH_SIZE          = properties.getint('general','batch_size')

# Load the data
    save_dir = "./spiking_data"

    gaussian            = properties.getint('general','gaussian')

    input_dim = target_cell_num

# generate dataset using spiking data if necessary (transposed, boolean)
    DT = bin_width_ms
    if "SNN" in model_name:
        DT = 1

    train_loaders = None
    valid_loaders = None

    if not inference: # for train only1
        train_data = torch.load('train_data_{}_{}.pt'.format(model_name, target_cell_num))
        train_loaders = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)

    valid_data = CustomDataset(valid_dir+'valid_data_{}_{}_X_valid.npy'.format(model_name, target_cell_num), valid_dir+'valid_data_{}_{}_y_valid.npy'.format(model_name, target_cell_num))

    valid_loaders = DataLoader(valid_data, batch_size=BATCH_SIZE, shuffle=False)

# Train a single epoch
    def process_one_epoch(model, loader, loss_fn, optimizer = None, train=True, profile=False, measure=False):
        running_loss = 0.
        pred_total = []
        actual_total = []
        batch_idx_final = None
        for batch_idx, (inputs, labels) in enumerate(tqdm.tqdm(loader)):
            batch_idx_final = batch_idx
            inputs = inputs.to(device)
            labels = labels.to(device)
            output = model(inputs, profile, measure)

            output = output.to(device)

            labels = torch.mean(labels, -1)
            labels = torch.transpose(labels, 1, 2)

            loss = loss_fn(output, labels) # calculate loss

            if not train:
                if "SNN" in model_name:
                    output = output.transpose(1, 2)
                for pred_dat in output.reshape(-1, output.shape[-1]).cpu().detach().numpy():
                    pred_total.append(pred_dat)

                if "SNN" in model_name:
                    labels = labels.transpose(1, 2)
                for actual_dat in labels.reshape(-1, labels.shape[-1]).cpu().detach().numpy():
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
            return running_loss / (batch_idx_final + 1), None
        else:
            return running_loss / (batch_idx_final + 1)

# Initializing in a separate cell so we can easily add more epochs to the same run
    checkpoint_name = dataset_dir+"{}/{}".format(model_name, input_dim)
    Path(checkpoint_name).mkdir(parents=True, exist_ok=True)
    p = os.path.abspath(valid_dir)
    sys.path.insert(1,p)

    print('\n##### Bin Width : %d #####'%bin_width_ms)
    if "Slayer" in model_name and ("FCSNN" in model_name or "RNNSNN" in model_name):
        model = DecoderSlayerSNN(model_name = model_name, input_dim = input_dim, h_dim = h_dim, neuron_name = neuron_name, delay = delay, bin_width_ms = bin_width_ms, DT = DT)
    else:
        assert(0)

    model.to(device)

    loss_fn = nn.MSELoss()

    optimizer1, optimizer2 = None, None

    if not inference:
        if "Slayer" in model_name:
            weight_params = [p for n, p in model.named_parameters() if n.find('weight') != -1]
            optimizer1 = torch.optim.Adam(weight_params, lr=LR_WEIGHT)
            delay_params = [p for n, p in model.named_parameters() if n.find('weight') == -1]
            #delay_params = [p for n, p in model.named_parameters() if n.find('delay') != -1]
            if delay_params:
                optimizer2 = torch.optim.Adam(delay_params, lr=LR_DELAY)
        else:
            optimizer1 = torch.optim.Adam(model.parameters(), lr=LR_WEIGHT)

    best_vloss = float('inf')

# epoch number to start from
    if inference:
        # Fake inference to initiate delay
        avg_valid_loss, R2 = process_one_epoch(model, valid_loaders, loss_fn, train=False, profile=True)
        checkpoint = torch.load('{}/checkpoint_best_bin_{}_gaussian_{}.pt'.format(checkpoint_name, \
                                                                                  bin_width_ms, \
                                                                                  gaussian), \
                                                                                 map_location=device)
        model.load_state_dict(checkpoint['model'])

        avg_valid_loss, R2 = process_one_epoch(model, valid_loaders, loss_fn, train=False, profile=False, measure=True)
        average_time = sum(model.elapsed_time) / len(model.elapsed_time)
        average_time_per_ts = sum(model.elapsed_time) * 1000 / model.process_iter
        print("average time {} (s)".format(average_time))
        print("average time per ts: {} ms to process {} ms".format(average_time_per_ts, bin_width_ms))
    else:
        start_epoch = 0
        if start_epoch != 0:
            checkpoint = torch.load('{}/checkpoint_bin_{}_ep_{}.pt'.format(checkpoint_name, bin_width_ms, start_epoch), map_location=device)
            model.load_state_dict(checkpoint['model'])
            start_epoch += 1
        optimizer = [optimizer1, optimizer2]
        for epoch in range(start_epoch, EPOCHS_WEIGHT):
            print('EPOCH %d:'%epoch)
            # Make sure gradient tracking is on, and do a pass over the data
            model.train(True)

            if (epoch + 1) % LR_STEP == 0:
                for opt in optimizer:
                    if opt != None: opt.param_groups[0]['lr'] *= 0.1

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

    # valid_data.close()

if __name__ == '__main__':
    profile_wrapper()
