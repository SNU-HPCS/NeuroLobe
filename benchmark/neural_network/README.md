# Dataset

## How to clone this repository onto your local machine

### On Windows

Use these steps to clone from SourceTree, our client for using the repository command-line free. Cloning allows you to work on your files locally. If you don't yet have SourceTree, [download and install first](https://www.sourcetreeapp.com/). If you prefer to clone from the command line, see [Clone a repository](https://confluence.atlassian.com/x/4whODQ).

1. You’ll see the clone button under the **Source** heading. Click that button.
2. Now click **Check out in SourceTree**. You may need to create a SourceTree account or log in.
3. When you see the **Clone New** dialog in SourceTree, update the destination path and name if you’d like to and then click **Clone**.
4. Open the directory you just created to see your repository’s files.

### On Linux

Set up personal SSH keys on Linux. Please refer to [here](https://support.atlassian.com/bitbucket-cloud/docs/set-up-personal-ssh-keys-on-linux/).

And then git clone this repo with SSH.

## How to use this repository

This repo contains python code for generating neural spike dataset for movement decoding. Original dataset is from [Neural Latent Benchmark](https://neurallatents.github.io/datasets). Therefore, you need to download original data from [DANDI repo](https://dandiarchive.org/dandiset/). Provided notebook files include the code for downloading these files.

There are 2 notebook files each dedicated for MC-Maze dataset and MC-RTT dataset. Generated dataset will be saved as .csv format. 

These .csv files can be loaded and used for decoding hand movement velocity. Provided decoders include: WienerFilterDecoder, WienerCascadeDecoder, Support Vector Regression Decoder, XGBoostDecoder, DenseNNDecoder, SimpleRNNDecoder, GRUDecoder, LSTMDecoder. 
