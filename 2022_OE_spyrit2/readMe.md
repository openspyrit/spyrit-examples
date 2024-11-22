# OpenSpyrit: an Ecosystem for Reproducible Single-Pixel Hyperspectral Imaging 

The code in the current folder allows to reproduce the results that are reported in

> G. Beneti-Martin, L Mahieu-Williame, T Baudier, N Ducros, "OpenSpyrit: an Ecosystem for Reproducible Single-Pixel Hyperspectral Imaging," Optics Express, Vol. 31, No. 10, (2023). 

* DOI (open access): [10.1364/OE.483937](https://doi.org/10.1364/OE.483937)

* Preprint: [main document (PDF)](https://hal.science/hal-03910077v2/preview/OpenSpyrit.pdf) 

* Preprint: [supplemental document (PDF)](https://hal.science/hal-03910077v2/preview/revised.pdf)

*Contact:* nicolas.ducros@insa-lyon.fr, CREATIS Laboratory, University of Lyon, France.


## Get the code from Github

There are two options:

1. You can clone the entire `spyrit-examples` repository, which contains code for some other papers.
```shell
git clone https://github.com/openspyrit/spyrit-examples.git
```

2. Or you can use the `sparse-checkout` command to get only the code corresponding to this paper.
```shell
git clone -n --depth=1 --filter=tree:0 https://github.com/openspyrit/spyrit-examples
cd spyrit-examples
git sparse-checkout set 2022_OE_spyrit2
git checkout
```

## Installation

1. Create a conda environment
    ```shell
    conda create --name spyrit_OE_2022
    conda activate spyrit_OE_2022
    ```

1. Install pytorch. E.g.,
    ```shell
    conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia
    ```
    Visit https://pytorch.org/get-started/locally/ if you need a different installation.

1. Install SPyRiT and a few more packages
    ```shell
    pip install spyrit==2.3.3
    pip install girder-client
    pip install spyder-kernels
    ```
    
4. Install spas (more details [here](https://github.com/openspyrit/spas))

    ```shell
    cd ..
    git clone https://github.com/openspyrit/spas.git@df6cdd01dc4f25b2ff44cdfc82717f868317c7d9
    cd spas
    pip install -e .
    ```

## Get the scripts, networks and raw data

1. We recommend to run the `download_data.py` script from the `/2022_OE_spyrit2/` folder. E.g.,

    ```shell
    cd spyrit-examples/2022_OE_spyrit2/ 
    python download_data.py
    ```

2. Otherwise (not recommended), manually download

* the models from this [link](https://pilot-warehouse.creatis.insa-lyon.fr/#collection/6140ba6929e3fc10d47dbe3e/folder/638630794d15dd536f04831e),

* the covariance matrices from this [link](https://pilot-warehouse.creatis.insa-lyon.fr/#collection/6140ba6929e3fc10d47dbe3e/folder/63935a034d15dd536f048487),

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
## How to reproduce the results of the paper?
1. Run `fig8_mask.py` and `fig8_recon.py` to reproduce the sampling masks and reconstructed images in Figure 8, respectively. 

4. Run `supplemental.py` to reproduce all the figures in the supplemental document. 

## Train the network from scratch
```powershell
./train.py --M 2048 --img_size 128 --batch_size 256
```