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

# Install the dependencies

Our scripts primarily relies on the [SPyRiT ](https://github.com/openspyrit/spyrit) package that can be installed via `pip`. We recommend creating a virtual (e.g., conda) environment first.

NB: On Windows, you need to install [torch](https://pytorch.org/get-started/locally/) before SPyRiT.

```shell
# conda (or pip) install
conda create --name spyrit-env
conda activate spyrit-env
#conda install -c anaconda spyder=5  # Not working!
conda install -c anaconda pip
# windows only
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch

# pip install
pip install spyrit # tested with spyrit==1.1.0
pip install spyder
```

# Experimental data

We exploit some of the single-pixel raw acquisitions that constitute the [SPIHIM](https://github.com/openspyrit/spihim) dataset. The preprocessed data considered in the paper can be downloaded [here](https://www.creatis.insa-lyon.fr/~ducros/spyritexamples/2021_OpticsExpress/expe.zi). In a terminal, 

```shell
wget https://www.creatis.insa-lyon.fr/~ducros/spyritexamples/2021_OpticsExpress/expe.zip
```

Unzip and save in `./spyrit-examples/2021_Optics_express/data/` [CHECK!]

# Trained neural networks

We provide the trained network [here](https://www.creatis.insa-lyon.fr/~ducros/spyritexamples/2021_OpticsExpress/model.zip). In a terminal, 

```shell
wget https://www.creatis.insa-lyon.fr/~ducros/spyritexamples/2021_OpticsExpress/model.zip
```

Unzip and save in `./spyrit-examples/2021_Optics_express/models/`

# Installation 

```
start new envoriment
install spyrit v1.0
Install any possible extra dependencies
Launch jupyter-Lab on that new environment
Run the jupyter-Notebook
```



# Training the neural networks from scratch

```
python train_neural_networks --CR 512 --N0 50 --sig 0.5
```

