# reference :
# - MEArec doc
# https://mearec.readthedocs.io/en/latest/index.html
# - MEA definition (custom probe)
# https://meautility.readthedocs.io/en/latest/mea_definitions.html

import MEArec as mr
import MEAutility as mu
import yaml
from pprint import pprint
import matplotlib.pylab as plt
#%matplotlib notebook

import configparser
import yaml, os, sys

properties = configparser.ConfigParser()

properties.read('gen.params')

# 1. general params
sim_mode = properties.get('general', 'mode')
assert sim_mode in ("full", "template", "recording")
sim_elec_num = properties.getint('general', 'elec_num')

# 2. gen template params
# sim_temp_num = properties.getint('template', 'temp_num')
sim_sampling_freq = properties.getfloat('template', 'sampling_freq')

# 3. gen recording params
sim_duration = properties.getint('recording', 'duration')
sim_f_exc = properties.getint('recording', 'f_exc')
sim_f_inh = properties.getint('recording', 'f_inh')

# Check params
########################
print(f"sim_mode : {sim_mode}")
print(f"sim_exc_num : {sim_elec_num}")
print(f"sim_sampling_freq : {sim_sampling_freq}")
print(f"sim_duration : {sim_duration}")
print(f"sim_f_exc : {sim_f_exc}")
print(f"sim_f_inh : {sim_f_inh}")
########################

if sim_mode == 'full':
    gen_template = True
    gen_recording = True
elif sim_mode == 'template':
    gen_template = True
    gen_recording = False
elif sim_mode == 'recording':
    gen_template = False
    gen_recording = True
else:
    assert(0)

default_info, mearec_home = mr.get_default_config()
pprint(default_info)

# define cell_models folder
cell_folder = default_info['cell_models_folder']
print(cell_folder)
template_params = mr.get_default_templates_params()

# Neuropixel

mu.return_mea()
sys.stdout.flush()

# Neuropixel_info = mu.return_mea_info('Neuropixels2-384')
# pprint(Neuropixel_info)
# assert(0)

user_info5 = {'electrode_name': 'Neuropixels-100',
	      'description': 'Neuropixels probe. 100 square contacts in 4 staggered columns.',
		'sortlist': None, 'pitch': [40.0, 16.0],
		'dim': [25, 4],
		'stagger': 20,
		'size': 6.0,
		'plane': 'yz',
		'shape': 'square',
		'type': 'mea'}

user_info = {'electrode_name': 'Neuropixels-200',
	      'description': 'Neuropixels probe. 200 square contacts in 4 staggered columns.',
		'sortlist': None, 'pitch': [40.0, 16.0],
		'dim': [50, 4],
		'stagger': 20,
		'size': 6.0,
		'plane': 'yz',
		'shape': 'square',
		'type': 'mea'}

user_info2 = {'electrode_name': 'Neuropixels-400',
	      'description': 'Neuropixels probe. 400 square contacts in 4 staggered columns.',
		'sortlist': None, 'pitch': [40.0, 16.0],
		'dim': [100, 4],
		'stagger': 20,
		'size': 6.0,
		'plane': 'yz',
		'shape': 'square',
		'type': 'mea'}

user_info3 = {'electrode_name': 'Neuropixels-800',
	      'description': 'Neuropixels probe. 800 square contacts in 4 staggered columns.',
		'sortlist': None, 'pitch': [40.0, 16.0],
		'dim': [200, 4],
		'stagger': 20,
		'size': 6.0,
		'plane': 'yz',
		'shape': 'square',
		'type': 'mea'}

user_info4 = {'electrode_name': 'Neuropixels-1600',
	      'description': 'Neuropixels probe. 1600 square contacts in 4 staggered columns.',
		'sortlist': None, 'pitch': [40.0, 16.0],
		'dim': [400, 4],
		'stagger': 20,
		'size': 6.0,
		'plane': 'yz',
		'shape': 'square',
		'type': 'mea'}

user_info6 = {'electrode_name': 'Neuropixels-1024',
	      'description': 'Neuropixels probe. 1024 square contacts in 4 staggered columns.',
		'sortlist': None, 'pitch': [40.0, 16.0],
		'dim': [256, 4],
		'stagger': 20,
		'size': 6.0,
		'plane': 'yz',
		'shape': 'square',
		'type': 'mea'}

# print(mu.return_mea_info('Neuropixels-384'))
# assert(0)

with open('Neuropixels-100.yaml', 'w') as f:
    yaml.dump(user_info5, f)
with open('Neuropixels-200.yaml', 'w') as f:
    yaml.dump(user_info, f)
with open('Neuropixels-400.yaml', 'w') as f:
    yaml.dump(user_info2, f)
with open('Neuropixels-800.yaml', 'w') as f:
    yaml.dump(user_info3, f)
with open('Neuropixels-1600.yaml', 'w') as f:
    yaml.dump(user_info4, f)
with open('Neuropixels-1024.yaml', 'w') as f:
    yaml.dump(user_info6, f)

yaml_files = [f for f in os.listdir('.') if f.endswith('.yaml')]
print(yaml_files)

# add custom mea file
mu.add_mea('Neuropixels-100.yaml')
mu.add_mea('Neuropixels-200.yaml')
mu.add_mea('Neuropixels-400.yaml')
mu.add_mea('Neuropixels-800.yaml')
mu.add_mea('Neuropixels-1600.yaml')
mu.add_mea('Neuropixels-1024.yaml')

probe_name = f"Neuropixels-{sim_elec_num}"
# probe_name = "Neuronexus-32"
template_params['probe'] = probe_name # select type of electrode probe

template_params['n'] = sim_elec_num
template_params['dt'] = 1 / sim_sampling_freq * 1000
pprint(template_params)
# the templates are not saved, but the intracellular simulations are saved in 'templates_folder'
# set parameters
sim_exc_num = int((sim_elec_num * 250 / 384) * (4/5))
sim_inh_num = int((sim_elec_num * 250 / 384) * (1/5))
# sim_exc_num = 16
# sim_inh_num = 4

num_neurons = sim_exc_num + sim_inh_num
n_exc = sim_exc_num
# n_inh

### 2 step procedure : 1. generate template / 2. generate recording using template (NOTE : different from the template used for spike sorting)

### 1. generate template
if gen_template:
# set intracellular frequency

    tempgen = mr.gen_templates(cell_models_folder=cell_folder, params=template_params, n_jobs=24, verbose=True)
# this will take a few minutes...

# save templates in h5 format
    mr.save_template_generator(tempgen, filename=f'data/{probe_name}_{num_neurons}_templates.h5')

### 2. generate recording
if gen_recording:
# set electrode sampling frequency

    recording_params = mr.get_default_recordings_params()
    # Set parameters
    # recording_params['cell_types'] = {'excitatory': ['STPC'], 'inhibitory': ['DBC']}
    recording_params['spiketrains']['n_exc'] = n_exc
    recording_params['spiketrains']['n_inh'] = num_neurons - n_exc
    recording_params['spiketrains']['duration'] = sim_duration
    recording_params['spiketrains']['seed'] = 0
    recording_params['spiketrains']['f_exc'] = sim_f_exc
    recording_params['spiketrains']['f_inh'] = sim_f_inh


    recording_params['templates']['min_amp'] = 40
    recording_params['templates']['max_amp'] = 300
    recording_params['templates']['seed'] = 0

    recording_params['recordings']['modulation'] = 'electrode'
    recording_params['recordings']['noise_mode'] = 'uncorrelated'
    recording_params['recordings']['noise_level'] = 10
# use chunk options
    recording_params['recordings']['chunk_conv_duration'] = 20
    recording_params['recordings']['chunk_noise_duration'] = 20
    recording_params['recordings']['chunk_filter_duration'] = 20
    recording_params['recordings']['seed'] = 0

    pprint(recording_params)

### Comment : tmp_mode should be disabled, or Out-Of-Memory
# to avoid out of memory, move your default tmp folder path to your root directory
# add following line to your .bashrc
# export TMPDIR="$HOME/tmp"
    recgen = mr.gen_recordings(templates=f'data/{probe_name}_{num_neurons}_templates.h5', params=recording_params, verbose=True)

    print('Recordings shape', recgen.recordings.shape)
    print('Selected templates shape', recgen.recordings.shape)
    print('Sample template locations', recgen.template_locations[:3])
    print('Number of neurons', len(recgen.spiketrains))
    print('Sample spike train', recgen.spiketrains[0].times)

# save recordings in h5 format
    mr.save_recording_generator(recgen, filename=f'data/{probe_name}_{num_neurons}_{sim_duration}s_recordings.h5')

# Successfully done
    print("Successfully saved recording files")
