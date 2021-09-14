# Single-pixel image reconstruction from experimental data using neural networks 

This repository contains the code related to the paper `Antonio Lorente Mur, Pierre Leclerc, Françoise Peyrin, and Nicolas Ducros, "Single-pixel image reconstruction from experimental data using neural networks," Opt. Express 29, 17097-17110 (2021)` [https://doi.org/10.1364/OE.424228](https://doi.org/10.1364/OE.424228).

# Dependencies and installation

This code is functionnal with the V1.0 of spyrit (and all it's following dependencies). It also uses the data from spihim.

# Downloading Data and trained neural networks

## Downloading Spihim experimental data
```
wget https://www.creatis.insa-lyon.fr/~ducros/spyritexamples/2021_OpticsExpress/expe.zip
```

## Downloading the trained neural networks
```
wget https://www.creatis.insa-lyon.fr/~ducros/spyritexamples/2021_OpticsExpress/model.zip
```

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

