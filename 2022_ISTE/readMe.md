# Single-pixel reconstruction (ISTE 2021)
We provide the code that produces the results that we report in 

> Nicolas Ducros. Book title. Chapter Title. 2022 
> ⟨doi⟩.
> [Download PDF](https://hal.archives-ouvertes.fr/hal-XXXXXX/document/).

*Contact:* nicolas.ducros@insa-lyon.fr, CREATIS Laboratory, University of Lyon, France.



## Running the scripts
* Install SPyRiT and all dependencies. On Windows, you need first to install [torch](https://pytorch.org/get-started/locally/) first (see the SPyRiT [installation guide](https://github.com/openspyrit/spyrit)).

```shell
	pip install -e spyrit .
```
NB: The scripts were tested with the commit [0035901](https://github.com/openspyrit/spyrit/commit/00359010cb3f8c5ca46a50dab1d0c3fbe2dbc41a).

* Launch Spyder from the current folder

```shell
spyder
```
and run `main_*.py`. 

### STL-10 database

[optional] If you already have the STL-10 dataset on your computer, create a symbolic link.  Otherwise the STL-10 dataset will be downloaded.

* Linux:

```shell
ln -s <stl-10 parent folder> /data/ 
```

* Windows Powershell:

``` powershell
New-Item -ItemType SymbolicLink -Name \data\ -Target <stl-10 parent folder>
```

### Covariance matrix

Most of the scripts require a covariance matrix and mean image

   * Download and save them both in `./stats/`:

        https://www.creatis.insa-lyon.fr/~ducros/spyritexamples/2021_ISTE/Average_64x64.npy

        https://www.creatis.insa-lyon.fr/~ducros/spyritexamples/2021_ISTE/Cov_64x64.npy

   * Or (alternative likely to take longer without GPU) compute them

``` python 
from spyrit.misc.statistics import stat_walsh_stl10
stat_walsh_stl10()
```

## Trained networks (not tested yet)

You can download the networks from this [url](https://www.creatis.insa-lyon.fr/~ducros/spyritexamples/2020_ISBI_CNet/2020_ISBI_CNet.zip). 

We also  provide `train.py` to train the different variants of the network, from a single command line.

1. Completion network

   ``` shell
   python train.py
   ```

2. Pseudo inverse network

   ```shell
   python train.py --net_arch 2
   ```

3. Free network

   ```shell
   python train.py --net_arch 3
   ```

Note that 

* the models are saved in the default folder `.\models\`. To save them at another location consider
```
python train.py --model_root myfolder
```
* The defaults training parameters can be changed. For instance, run 
```
python train.py --num_epochs 10 --batch_size 512 --lr 1e-8
```
to train your network for 10 epochs, with a batch size of 512, and a learning rate of 1e-8. 
* you can keep `Average_64x64.npy`, `Cov_64x64.npy` and `Var_64x64.npy` in `.\stats\`, to avoid re-computing them, which can be time-consuming.

