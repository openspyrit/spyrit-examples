
# Labex PRIMES workshop on reproductibility

This code generates the figures reported in the presentation entitled "Robustness of Deep Reconstruction Methods" (in French "Robustesse des méthodes de reconstruction par deep learning") given at Labex PRIMES [workshop on reproductibility](https://reprod-primes.sciencesconf.org/) on 08 December 2022.

*Authors:* N Ducros

*Contact:* nicolas.ducros@insa-lyon.fr, CREATIS Laboratory, University of Lyon, France.

## Install the dependencies

1. We recommend creating a virtual (e.g., conda) environment first.

    ```shell
    # conda install
    conda create --name new-env
    conda activate new-env
    conda install spyder
    conda install -c conda-forge matplotlib
    conda install -c conda-forge jupyterlab
    conda install -c anaconda scikit-image
    conda install -c anaconda h5py 
    conda install pytorch torchvision torchaudio cudatoolkit=11.6 -c pytorch -c conda-forge
    ```

    Alternatively, you can clone an existing environment with `conda create --name new-env --clone existing-env`

1. Clone the spyrit package, and install the version in the  `towards_v2_fadoua` branch

    ```shell
    git clone https://github.com/openspyrit/spyrit.git
    cd spyrit
    git checkout towards_v2
    pip install -e .
    ```
    
1. Clone the spas package: 

    ```shell
    git clone https://github.com/openspyrit/spas.git
    cd spas
    pip install -e .
    ```

## Get the scripts and networks

1.  Get source code and navigate to the `/2022_reproductibility/` folder
    ```shell
    git clone https://github.com/openspyrit/spyrit-examples.git
    cd spyrit-examples/2022_reproductibility/ 
    ```
2. Download the models from this [link](https://pilot-warehouse.creatis.insa-lyon.fr/#collection/6140ba6929e3fc10d47dbe3e/folder/639355234d15dd536f0483c4) 

3. Download the average and covariance matrices from this [link](https://pilot-warehouse.creatis.insa-lyon.fr/#collection/6140ba6929e3fc10d47dbe3e/folder/639359a14d15dd536f04847a)

The directory structure should be

```
|---model_v2
|   |---dc-net_unet_stl10
|   |   |---reprod
|   |   |   |---dc-net_unet_stl10_rect_N0_2_N_64_M_1024_*.pth
|   |   |   |---
|---spyrit-examples
|   |---2022_reproductibility
|   |   |---main_net_std.py
|   |   |---
|---stat
|   |---stl10
|   |   |---Average_64x64.npy
|   |   |---Cov_64x64.npy
```


## Train the networks from scratch
For a given random seed (`seed`) and image intensity `N0` (in photons)

```powershell
python ./train.py --seed 4 --N0 10
```

For all cases considered in the study

```powershell
./train.ps1
```