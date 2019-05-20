# MONet
An implementation of the DeepMind's MONet model for unsupervised scene decomposition in PyTorch. The
model was presented in the paper "[MONet: Unsupervised Scene Decomposition and
Representation](https://arxiv.org/abs/1901.11390)" by Christopher P. Burgess, Loic Matthey,
Nicholas Watters, Rishabh Kabra, Irina Higgins, Matt Botvinick and Alexander Lerchner.

Similarly to previous models such as [AIR](https://arxiv.org/abs/1603.08575), MONet learns to
decompose scenes into objects and background in an unsupervised setting. Unlike AIR however, it
learns attention masks to obtain real segmentations instead of just bounding boxes. Objects and
background appearances are modelled by a VAE.

## Dependencies
We ran our experiments using Python 3.6 and CUDA 9.0, making use of the following Python packages:

 * torch 1.0
 * numpy
 * visdom

These may be installed via `pip install -r requirements.txt`. Other versions might also work but
were not tested.

## Project Structure

 * `model.py` contains the model, implemented as a set of PyTorch modules
 * `main.py` contains the training loops
 * `config.py` contains adjustable parameters, including directories and hyperparameters
 * `datasets.py` contains routines for loading the data

## Run
Simply run `python main.py`. Adjust the configuration object created at the bottom of the file as
needed, or use one configurations to reproduce the results above. Note that the experiments on CLEVR
were run on a V100 GPU with 32GB of memory, so you may need to reduce the model size in order to fit
it on a smaller GPU.
