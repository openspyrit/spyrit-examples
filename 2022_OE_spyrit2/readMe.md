# OpenSpyrit: an Ecosystem for Reproducible Single-Pixel Hyperspectral Imaging 

The code in the current folder allows to reproduce the results that are reported in

> G. Beneti-Martin, L Mahieu-Williame, T Baudier, N Ducros, "OpenSpyrit: an Ecosystem for Reproducible Single-Pixel Hyperspectral Imaging," Optics Express, Vol. 31, No. 10, (2023). 

*DOI (open access):* https://doi.org/10.1364/OE.483937

*Preprint (main PDF):* https://hal.science/hal-03910077

*Preprint (supplemental document):* https://hal.science/hal-XXXXXXX 

*Contact:* nicolas.ducros@insa-lyon.fr, CREATIS Laboratory, University of Lyon, France.

## Install the dependencies

1. We recommend using a virtual (e.g., conda) environment.

    ```shell
    # conda install
    conda create --name new-env
    conda activate new-env
    ```

    Alternatively, you can clone an existing environment with `conda create --name new-env --clone existing-env`

1. Install the spyrit package (more details [here](https://github.com/openspyrit/spyrit)). Tested with spyrit 2.1.

    ```shell
    pip install spyrit==2.1
    ```
    
1. Clone and install the spas package (more details [here](https://github.com/openspyrit/spas)). Tested with spas 1.2. 

    ```shell
    pip install -e https://github.com/openspyrit/spas.git@1.2
    ```

## Get the scripts, networks and raw data

1.  Get source code and navigate to the `/2022_OE_spyrit2/` folder

    ```shell
    git clone https://github.com/openspyrit/spyrit-examples.git
    cd spyrit-examples/2022_OE_spyrit2/ 
    ```
    
2. Download models, statistics, and raw data

Get all files by running `download_data.py`, e.g.,
```shell
python3 download_data.py
```

Otherwise, select

* the models from this [link](https://pilot-warehouse.creatis.insa-lyon.fr/#collection/6140ba6929e3fc10d47dbe3e/folder/638630794d15dd536f04831e) 

* the covariance matrices from this [link](https://pilot-warehouse.creatis.insa-lyon.fr/#collection/6140ba6929e3fc10d47dbe3e/folder/63d7f3620386da2747641e1b) 

* the raw data from this [link](https://pilot-warehouse.creatis.insa-lyon.fr/#collection/6140ba6929e3fc10d47dbe3e/folder/6149c3ce29e3fc10d47dbffb).


The directory structure should be

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
|   |   |---recon
```


## Train the network from scratch
```powershell
./train.py --M 2048 --img_size 128 --batch_size 256
```