conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch
cd custom_slayer
cd cuda_toolkit
python setup.py install
pip install nlb-tools
conda install -c anaconda networkx
pip install elephant
