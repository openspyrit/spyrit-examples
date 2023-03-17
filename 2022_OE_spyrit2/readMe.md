# OpenSpyrit: an Ecosystem for Reproducible Single-Pixel Hyperspectral Imaging 

This repository contains the code that produces the results reported in

> G. Beneti-Martin, L Mahieu-Williame, T Baudier, N Ducros, "OpenSpyrit: an Ecosystem for Reproducible Single-Pixel Hyperspectral Imaging," Preprint, (2022). [HAL (open access PDF)](https://hal.science/hal-03910077).

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

1. Clone the spyrit package, and install the version in the  `towards_v2` branch: 

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
> Todo: Upate spas package to remove unwanted imports of ``reconstruct_nn``  

## Get the scripts, networks and raw data

1.  Get source code and navigate to the `/2022_OE_spyrit2/` folder

    ```shell
    git clone https://github.com/openspyrit/spyrit-examples.git
    cd spyrit-examples/2022_OE_spyrit2/ 
    ```
    
2. Download 

* the models from this [link](https://pilot-warehouse.creatis.insa-lyon.fr/#collection/6140ba6929e3fc10d47dbe3e/folder/638630794d15dd536f04831e) 

* the covariance matrices from this [link](https://pilot-warehouse.creatis.insa-lyon.fr/#collection/6140ba6929e3fc10d47dbe3e/folder/63d7f3620386da2747641e1b) 

* the raw data from this [link](https://pilot-warehouse.creatis.insa-lyon.fr/#collection/6140ba6929e3fc10d47dbe3e/folder/6149c3ce29e3fc10d47dbffb). You can download the full folder with all subfolder or navigate to select a few of them.


Unzip the folders into `/model/`, `/stat/`, and `/data/`, respectively. The directory structure should be

```
|---spyrit-examples
|   |---2022_OE_spyrit2
|   |   |---data
|   |   |   |---
|   |   |---model
|   |   |   |---
|   |   |---stat
|   |   |   |---
|   |   |---fig8_recon.py
|   |   |---fig8_mask.py
|   |   |---
```


## To train the network from scratch
```powershell
./train.py --M 2048 --img_size 128 --batch_size 256
```