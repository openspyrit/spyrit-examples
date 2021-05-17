# Deep neural network for limited view X-ray tomography

*Contact:* nicolas.ducros@insa-lyon.fr, CREATIS Laboratory, University of Lyon, France.

## Running the code
1. Simply launch JupiterLab from the current folder
```
$ jupyter notebook
```
and run `main.ipynb`. Note that this notebook downloads trained networks from this [url](https://www.creatis.insa-lyon.fr/~ducros/spyritexamples/2020_ISBI_CNet/2020_Radon_CNet.zip). 

2. Alternatively, we provide `train.py` to train the network. In a terminal:
```
$ train 
```
Note that 
* the models are saved in the default folder `.\models\`. To save them at another location consider
```
$ train --model_root myfolder
```
* The defaults training parameters can be changed. For instance, run 
```
$ train --num_epochs 10 --batch_size 512 --lr 1e-8
```
to train your network for 10 epochs, with a batch size of 512, and a learning rate of 1e-8. 
* you can keep `Mean_Q64D64.pt` and `Mean_Q64D64.pt`, if you have downloaded them, to avoid their computation that can be time-consumming.

## Generate more forward matrix with matlab
We provide code to compute various forward matrix with diferent number of measurement angles or different image resolution.

1. Go to the matlab folder and run main.m

2. The function compute_radon_matrix and compute_pinv_radon_matrix compute and export respectively the forward and backpropagation matrixes.

3. verify_radon_matrix let you compare results between the forward operator and the matlab function radon. compare_backprojection let you observe reconstruction for different numbers of acquisition angles and see the influence of receptor's pixel number.