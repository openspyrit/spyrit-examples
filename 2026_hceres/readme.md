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

1. Install latest SPyRiT (and a few more packages ?)
    ```shell
    pip install pip install git+https://github.com/openspyrit/spyrit.git@master
    pip install girder-client
    pip install scikit-image
    ```

## Download the models and data [Check if this works]

*Todo: Update `download_data.py`!*  

Run the `download_data.py` script from the `2026_hceres` subfolder
```shell
cd spyrit-examples/2026_hceres/ 
python download_data.py
```

If measurements are already available on your computer, just create a symbolic link

* Linux:

    ```shell
    ln -s <imagenet folder> /data/ILSVRC2012/ 
    ```

* Windows Powershell (run as Administrator):

    ```shell
    New-Item -ItemType SymbolicLink -Name \data\2025-11-10_test_HCERES\ -Target <measurement folder>
    ```
The directory structure should be as follows:

```
|---spyrit-examples
|   |---2025_spyrit_v3
|   |   |---data
|   |   |   |---2025-11-10_test_HCERES
|   |   |   |--- obj_Cat_bicolor_thin_overlap_...
|   |   |   |   |---obj_Cat_bicolor_thin_overlap_..._metadata.json
|   |   |   |   |---obj_Cat_bicolor_thin_overlap_..._had_reco.npz
|   |   |   |   |---obj_Cat_bicolor_thin_overlap_..._spectraldata.npz
|   |   |   |---
|   |   |---model
|   |   |   |---
|   |   |---stat
|   |   |   |---
|   |   |---aux_functions.py
|   |   |---download_data.py
```