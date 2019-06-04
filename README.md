# MONet
This repository contains an implementation of the DeepMind's MONet model for unsupervised scene decomposition in PyTorch. The
model was presented in the paper "[MONet: Unsupervised Scene Decomposition and
Representation](https://arxiv.org/abs/1901.11390)" by Christopher P. Burgess, Loic Matthey,
Nicholas Watters, Rishabh Kabra, Irina Higgins, Matt Botvinick and Alexander Lerchner.

Similarly to previous models such as [AIR](https://arxiv.org/abs/1603.08575), MONet learns to
decompose scenes into objects and background in an unsupervised setting. Unlike AIR however, it
learns attention masks to obtain real segmentations instead of just bounding boxes. Objects and
background appearances are modelled by a VAE.

## Sample Results
The following image shows a sample of results on a homemade version of the sprite dataset. The first line
of images depicts the input, the second the inferred segmentation, and the third the reconstruction.

<img src="https://raw.githubusercontent.com/stelzner/MONet/master/images/sprite-results.png" alt="MONet on sprites" width="600">

The attention network successfully learns to segment the images.
One issue appears to be that distinct objects of the same color tend to not be separated. Since the model
structure does not force objects to be spatially coherent, this is perhaps to be expected.

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
needed, or use one of the provided configurations to reproduce the results above. Note that the experiments
on CLEVR were run on a V100 GPU with 32GB of memory, so you may need to reduce the model size in order to fit
it on a smaller GPU.
