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
conda create --name spyrit-spas
conda activate spyrit-spas
```
### First, install pytorch using conda
Use the following command or visit https://pytorch.org/get-started/locally/ if you need a different installation.
```shell
conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia
```

### Install SPyRiT and a few more packages
```shell
pip install spyrit==2.3.3
pip install ipykernel
pip install girder-client
pip install scikit-image
```

The `pandas` package may be needed to save some results in a csv file. 

### Install SPAS (single-pixel acquisition software)
```shell
git clone -b tmp-oe --single-branch https://github.com/openspyrit/spas.git
cd spas # update path in necessary
pip install -e .
cd ..
```

For more information, check https://github.com/openspyrit/spas. 

## Get code and data
First, get the source code

```shell
git clone https://github.com/openspyrit/spyrit-examples.git
cd spyrit-examples/dev/2024_Optica/ 
```

Download the models and data

```shell
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