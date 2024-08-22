# SPyRiT: an open source package for single-pixel imaging based on deep learning

We provide here the code to reproduce the results reported in

> JFJP Abascal, T Baudier, R Phan, A Repetti, N Ducros, "SPyRiT: an open source package for single-pixel imaging based on deep learning," Preprint (2024). 

*Preprint view (main PDF + supplemental document):* https://hal.science/hal-04662876v1

*Preprint download (main PDF):* https://hal.science/hal-04662876v1/file/submitted.pdf

*Preprint download (supplemental document):* https://hal.science/hal-04662876v1/file/supplemental.pdf

*Contact:* nicolas.ducros@insa-lyon.fr, CREATIS Laboratory, University of Lyon, France.

## Installation
### Create a conda environment
```shell
conda create --name spyrit-dev
conda activate spyrit-dev
```
### First, install pytorch using conda
Use the following command or visit https://pytorch.org/get-started/locally/ if you need a different installation.
```shell
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
```

### [Developper mode] Then, clone spyrit and install it using pip
```shell
git clone https://github.com/openspyrit/spyrit.git
cd spyrit
git reset --hard 21db0562c38833de6a9f9298c6952105b248e1ba # specific commit
pip install -e .
```
### [User mode] Should work, but not tested
```shell
pip install spyrit==2.3.3
```

### Some other packages
The `pandas` package is only needed to save some results in a csv file. You may not need to install it.
```shell
pip install ipykernel
pip install girder-client
pip install scikit-image
pip install pandas
```

### Install SPAS
Follow the guidelines given in the SPAS ReadMe: https://github.com/openspyrit/spas. You don't need the DLLs.

## Get code and data
First, get the source code. and navigate to the

```shell
git clone https://github.com/openspyrit/spyrit-examples.git
```

Then, navigate to the `/dev/2024_Optica/` folder and download the models and data by running `download_data.py`:
```shell
cd spyrit-examples/dev/2024_Optica/ 
python3 download_data.py
```

The directory structure should be as follows:

```
|---spyrit-examples
|   |---dev
|   |   |---2024_Optica
|   |   |   |---data
|   |   |   |   |---
|   |   |   |---model
|   |   |   |   |---
|   |   |   |---stat
|   |   |   |   |---
|   |   |   |---download_data.py
|   |   |   |---figure_2.py
|   |   |   |---reconstruct_all_figures.py
|   |   |   |---train.py
|   |   |   |---aux_functions.py
|   |   |   |---models_helper.py
|   |   |   |---networks.py
|   |   |   |---
|   |   |   |---recon
```

## Generation of the results of the paper
1. Run `figure2_py` to reproduce the sampling masks, acquisition matrices, measurements, and images in Fig. 2.

1. Run `reconstruct_all_figures.py` to reconstruct all the images displayed in both the main and supplementary documents.

## Training of the networks from scratch
```powershell
./train.py --M 2048 --img_size 128 --batch_size 256
```