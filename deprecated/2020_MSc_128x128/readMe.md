# MSc student project 
## Single-Pixel Reconstruction of 128 x 128 images 
We assess the method introduced in the paper below on 128 x 128 images

> Nicolas Ducros, A Lorente Mur, F. Peyrin. A Completion Network for Reconstruction from Compressed Acquisition. 2020 IEEE 17th International Symposium on Biomedical Imaging (ISBI), Apr 2020, Iowa City, United States, pp.619-623, ⟨10.1109/ISBI45749.2020.9098390⟩.
> [Download PDF](https://hal.archives-ouvertes.fr/hal-02342766/document/).

*License:* The code is distributed under the Creative Commons Attribution 4.0 International license ([CC-BY 4.0](https://creativecommons.org/licenses/by/4.0/))

*Authors:* Marc Chanet and Juliette Coumert, Electrical Engineering Department, INSA-Lyon, France.

*Contact:* nicolas.ducros@insa-lyon.fr, CREATIS Laboratory, University of Lyon, France.

## Running the code
1. Make sure you have adequate database in `..\..\data\` to run the code. In this case :
* STL10
* ImageNet 2012 (IMNET)
2. Simply launch JupiterLab from the current folder
```
$ jupyter notebook
```
and run `main.ipynb`.

Note that this notebook relies on trained networks that can be downloaded from this [url](https://www.creatis.insa-lyon.fr/~ducros/spyritexamples/2021_MSc_128x128/2021_MSc_128x128.zip). 

3. Alternatively, we provide `train_imnet.py` and `train_STL10.py` to train the different variants of the network. 
For example to train imnet. In a terminal:
```
$ python train_imnet.py
```
Note that 
* the models are saved in the default folder `.\models\`. To save them at another location consider
```
$ python train_imnet.py --model_root myfolder
```
* The defaults training parameters can be changed. For instance, run 
```
$ python train_imnet.py --img_size 64 --num_epochs 20 --batch_size 256
```
to train your network for 20 epochs, with a batch size of 256, and with 64x64 images. 
* you can keep `Average_64x64.npy`, `Cov_64x64.npy` and `Var_64x64.npy` in `.\models\`, if you have downloaded them, to avoid their computation that can be time-consumming.

4. You'll find a cc_blocs.py script in this file. It reconstruct 128x128 images using 64x64 networks and reconstructing by blocs. It still has some bugs which prevent it from running but it worked at some point so you are free to try to use it.
