# A Completion Network for Reconstruction from Compressed Acquisition (ISBI 2020)
We provide the code that produces the results that we report in 

> Nicolas Ducros, A Lorente Mur, F. Peyrin. A Completion Network for Reconstruction from Compressed Acquisition. 2020 IEEE 17th International Symposium on Biomedical Imaging (ISBI), Apr 2020, Iowa City, United States, pp.619-623, ⟨10.1109/ISBI45749.2020.9098390⟩.
> [Download PDF](https://hal.archives-ouvertes.fr/hal-02342766/document/).

*License:* The SPIHIM datasets are distributed under the Creative Commons Attribution 4.0 International license ([CC-BY 4.0](https://creativecommons.org/licenses/by/4.0/))

*Contact:* nicolas.ducros@insa-lyon.fr, CREATIS Laboratory, University of Lyon, France.

## Running the main notebook
1. Install SPyRiT and all dependencies

```shell
pip install -e spyrit
```
```shell
pip install -r extra_requirements.txt
```
2. The STL-10 dataset will automatically be downloaded.  If you already have it on you computer, you can create a symbolic link to the repository containing STL-10

```
ln -s <name of parent folder of stl-10> /data/
```
3. Launch JupiterLab from the current folder

```shell
jupyter lab
```
and run `main.ipynb`. Note that this notebook downloads trained networks from this [url](https://www.creatis.insa-lyon.fr/~ducros/spyritexamples/2020_ISBI_CNet/2020_ISBI_CNet.zip). 

## Re-training the networks

1. Alternatively, we provide `train.py` to train the different variants of the network. In a terminal:
```shell
python train 
python train --net_arch 1
python train --net_arch 2
```
Note that 
* the models are saved in the default folder `.\models\`. To save them at another location consider
```
python train --model_root myfolder
```
* The defaults training parameters can be changed. For instance, run 
```
python train --num_epochs 10 --batch_size 512 --lr 1e-8
```
to train your network for 10 epochs, with a batch size of 512, and a learning rate of 1e-8. 
* you can keep `Average_64x64.npy`, `Cov_64x64.npy` and `Var_64x64.npy` in `.\stats\`, if you have downloaded them, to avoid their computation that can be time-consumming.

