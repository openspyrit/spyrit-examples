# Single-pixel Image Reconstruction from Experimental Data using Neural Networks 

This repository contains the code that produces the results reported in

> Antonio Lorente Mur, Pierre Leclerc, Françoise Peyrin, and Nicolas Ducros, "Single-pixel image reconstruction from experimental data using neural networks," Opt. Express 29, 17097-17110 (2021). [DOI (open access)](https://doi.org/10.1364/OE.424228).

*Contact:* [nicolas.ducros@insa-lyon.fr](mailto:nicolas.ducros@insa-lyon.fr), CREATIS Laboratory, University of Lyon, France.

# Get the scripts

Clone the repo and navigate to the folder that corresponds to this study

```shell
git clone https://github.com/openspyrit/spyrit-examples.git
cd ./spyrit-examples/2021_Optics_express/
```

This folder contains two script

* main.py:  generates the figures in the paper  (typically run in Spyder). It requires trained networks.
* train.py: can be used to train the reconstruction networks (typically run in a terminal)

# Install the dependencies

Our scripts primarily relies on the [SPyRiT ](https://github.com/openspyrit/spyrit) package that can be installed via `pip`. We recommend creating a virtual (e.g., conda) environment first.

NB: On Windows, you need to install [torch](https://pytorch.org/get-started/locally/) before SPyRiT.

```shell
# conda (or pip) install
conda create --name spyrit-env
conda activate spyrit-env
#conda install -c anaconda spyder=5  
conda install -c anaconda pip
# for windows only
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch

# pip install
pip install spyrit # tested with spyrit==1.1.0
pip install spyder
pip install pylops
```

# Experimental data

We exploit some of the single-pixel raw acquisitions that constitute the [SPIHIM](https://github.com/openspyrit/spihim) dataset. The preprocessed data considered in the paper can be downloaded [here](https://www.creatis.insa-lyon.fr/~ducros/spyritexamples/2021_OpticsExpress/expe.zip). Unzip in `./spyrit-examples/2021_Optics_express/expe/`. 

# Trained neural networks

We provide the trained network [here](https://www.creatis.insa-lyon.fr/~ducros/spyritexamples/2021_OpticsExpress/model.zip). Unzip in `./spyrit-examples/2021_Optics_express/model/`. 

# Training the neural networks from scratch [check]

We train our networks by simulating single-pixel measurements from the STL-10 image database.

In a terminal:

```shell
python train.py --CR 512 --intensity_max 2500 --precompute_root ./model/ --num_epochs 20
```

NB: This will download the STL-10 database. 

[optional] If you already have the STL-10 on your computer, create a symbolic link.

* Linux: [CHECK]

```shell
ln -s <stl-10 parent folder> /data/ 
```

* Windows Powershell:

``` powershell
New-Item -ItemType SymbolicLink -Name \data\ -Target <stl-10 parent folder>
```

### 
