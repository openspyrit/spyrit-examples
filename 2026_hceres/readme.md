# HCERES Demonstration 

*Authors:* Jérémy Cohen, Nicolas Ducros, Laurent Mahieu-Williame 

*Contact:* nicolas.ducros@insa-lyon.fr, CREATIS Laboratory, University of Lyon, France.

## Get the code from Github

There are two options:

1. Clone the entire `spyrit-examples` repository, which contains code for some other papers.
    ```shell
    git clone https://github.com/openspyrit/spyrit-examples.git
    ```

2. Or use the `sparse-checkout` command to get only the code corresponding to this paper.
    ```shell
    git clone -n --depth=1 --filter=tree:0 https://github.com/openspyrit/spyrit-examples
    cd spyrit-examples
    git sparse-checkout set 2026_hceres
    git checkout
    ```
## Installation

1. Create a conda environment
    ```shell
    conda create --name hceres
    conda activate hceres
    ```

1. Install pytorch. E.g.,
    ```shell
    pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu126
    ```
    Visit https://pytorch.org/get-started/locally/ if you need a different installation.

1. Install SPyRiT v3 and a few more packages
    ```shell
    pip install spyrit==3.0.2
    pip install girder-client
    pip install scikit-image
    ```

## Download the models and data [Needs update!]

*Todo: Update `download_data.py`!*  

Run the `download_data.py` script from the `2026_hceres` subfolder
```shell
cd spyrit-examples/2025_spyrit_v3/ 
python download_data.py
```


The ImageNet (ILSVRC2012) test and validation sets can be downloaded from [this url](https://image-net.org/challenges/LSVRC/2012/2012-downloads.php). They must saved in `./data/ILSVRC2012/test/all` and `./data/ILSVRC2012/val/all`. 

If the images are already available on your computer, just create a symbolic link

* Linux:

    ```shell
    ln -s <imagenet folder> /data/ILSVRC2012/ 
    ```

* Windows Powershell (run as Administrator):

    ```shell
    New-Item -ItemType SymbolicLink -Name \data\ILSVRC2012\ -Target <imagenet folder>
    ```
*Todo: Update!* 

The directory structure should be as follows:

```
|---spyrit-examples
|   |---2025_spyrit_v3
|   |   |---data
|   |   |   |---ILSVRC2012
|   |   |   |   |---test
|   |   |   |   |   |---all
|   |   |   |   |   |   |---ILSVRC2012_test_00000001.JPEG
|   |   |   |   |   |   |---
|   |   |   |   |---val
|   |   |   |   |   |---all
|   |   |   |   |   |   |---ILSVRC2012_val_00000001.JPEG
|   |   |   |   |   |   |---
|   |   |   |---?????_spectraldata.npz
|   |   |   |---images
|   |   |   |---
|   |   |---model
|   |   |   |---
|   |   |---stat
|   |   |   |---
|   |   |---recon
|   |   |   |---
|   |   |---aux_functions.py
|   |   |---download_data.py
|   |   |---figure_2.py
|   |   |---figure_3.py
|   |   |---figure_4.py
|   |   |---supplemental_figure_S1.py
|   |   |---
|   |   |---table_1.py
|   |   |---train.py
|   |   |---utility_dpgd.py
```

## How to reproduce the results of the paper?