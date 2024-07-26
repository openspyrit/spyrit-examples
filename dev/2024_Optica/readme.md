# SPyRiT: an open source package for single-pixel imaging based on deep learning

We provide here the code to reproduce the results reported in

> JFJP Abascal, T Baudier, R Phan, A Repetti, N Ducros, "SPyRiT: an open source package for single-pixel imaging based on deep learning," Preprint (2024). 

*Preprint (main PDF):* https://hal.science/hal-xxx

*Preprint (supplemental document):* https://hal.science/hal-XXX 

*Contact:* nicolas.ducros@insa-lyon.fr, CREATIS Laboratory, University of Lyon, France.

## Installation
### Create a conda environment
```shell
conda create --name spyrit-dev
conda activate spyrit-dev
```
### First, install pytorch using conda
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
```shell
pip install ipykernel
pip install girder-client
```

## Get code and data
Get the source code and navigate to the `/dev/2024_Optica/` folder

```shell
git clone https://github.com/openspyrit/spyrit-examples.git
cd spyrit-examples/dev/2024_Optica/ 
```

Get the models and data by running `download_data.py`
```shell
python3 download_data.py
```

The directory structure should be

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