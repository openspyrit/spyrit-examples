# SPyRiT: an open source package for single-pixel imaging based on deep learning

We provide here the code to reproduce the results reported in

> JFJP Abascal, T Baudier, R Phan, A Repetti, N Ducros, "SPyRiT: an open source package for single-pixel imaging based on deep learning," Preprint (2024). 

*Preprint view (main PDF + supplemental document):* https://hal.science/hal-04662876v1

*Preprint download (main PDF):* https://hal.science/hal-04662876v1/file/submitted.pdf

*Preprint download (supplemental document):* https://hal.science/hal-04662876v1/file/supplemental.pdf

*Contact:* nicolas.ducros@insa-lyon.fr, CREATIS Laboratory, University of Lyon, France.

## Installation

### Method 1 (preferred): using environment.yml
Using a environment manager (e.g. conda), create an environment using the `environment.yml` file.
```shell
cd 2024_Optics_Express
conda env create -f environment.yml
```

### Method 2: install each module independently

#### Create a conda environment
```shell
conda create --name spyrit_optics_express_2024
conda activate spyrit_optics_express_2024
```

#### First, install pytorch using conda
Use the following command or visit https://pytorch.org/get-started/locally/ if you need a different installation.
```shell
conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia
```

#### Install SPyRiT and a few more packages
Still using the conda shell, run these lines:
```shell
pip install spyrit==2.3.3
pip install ipykernel
pip install girder-client
pip install scikit-image
```

## Get code and data

### Get the code from Github

You can clone only the code for this paper using a `sparse-checkout` in git. In a new directory, run these commands:
```shell
git clone -n --depth=1 --filter=tree:0 https://github.com/openspyrit/spyrit-examples
cd spyrit-examples
git sparse-checkout set 2024_Optics_Express
git checkout
```

Alternatively, you can clone the whole `spyrit-examples` repository (which includes code for some other papers):
```shell
git clone https://github.com/openspyrit/spyrit-examples.git
```

### Download the models and data

Run this in a python command line or from your favorite IDE:
```shell
cd spyrit-examples/2024_Optics_Express/ 
python3 download_data.py
```

The directory structure should be as follows:

```
|---spyrit-examples
|   |---dev
|   |   |---2024_Optics_Express
|   |   |   |---data
|   |   |   |   |---
|   |   |   |---model
|   |   |   |   |---
|   |   |   |---stat
|   |   |   |   |---
|   |   |   |---recon
|   |   |   |   |---
|   |   |   |---aux_functions.py
|   |   |   |---download_data.py
|   |   |   |---figure_2.py
|   |   |   |---figure_3.py
|   |   |   |---figure_4.py
|   |   |   |---table_1.py
|   |   |   |---train.py
|   |   |   |---utility_dpgd.py
```

## Paper results generation

1. Run `figure_2.py`, `figure_3.py`, `figure_4.py` to reproduce the sampling masks, acquisition matrices, measurements, and images in Fig. 2.

2. Run `figure_3.py` and `figure_4.py` to reproduce the reconstruction images obtained from simulated measurements (figure 3) or from experimental measurements (figure 4).

3. Run `table_1.py` to reproduce the statistical results presented in table 1. This code runs on a subset of the the ImageNet validation data. For this reason, you will get values close to but not equal to those presented in the paper. 

## Training of the networks from scratch

```powershell
./train.py --M 2048 --img_size 128 --batch_size 256
```