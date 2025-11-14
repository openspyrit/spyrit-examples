# Freeform Hadamard imaging: Back to the roots of computational optics

We provide the code to reproduce the results reported in

> N Ducros, J Cohen, L Mahieu-Williame, "Freeform Hadamard imaging: Back to the roots of computational optics," preprint (2025). 
* [PDF](https://hal.science/hal-xxxxxxx/document)

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
    git sparse-checkout set 2025_freeform
    git checkout
    ```

# Installation

1. Create a conda environment
    ``` shell
    conda create --name freeform
    conda activate freeform
    ```

1. Install PyTorch. Our scripts were tested with torch 2.8.0 and cuda 12.6.
    ``` shell
    pip ...
    ```

1. Install spyrit. Our scripts were tested with spyrit 3.0.2.

    (User mode, preferred)
    ``` shell
    pip install spyrit==3.0.2
    ```

    (Developper mode)
    ``` shell
    git clone https://github.com/openspyrit/spyrit.git
    cd spyrit
    pip install -e .
    ```

1. And a few more packages
    ``` shell
    pip install conda-forge::girder-client # for downloading expe data
    pip install spyder-kernels
    ```
    
# Download data
There are two options to download the raw measurements from our warehouse

1. Run `download_data.py` (preferred option)

    ```shell
    cd spyrit-examples/2025_spyrit_v3/ 
    python download_data.py
    ```
2. Otherwise, use this [direct link](https://pilot-warehouse.creatis.insa-lyon.fr/#collection/6140ba6929e3fc10d47dbe3e/folder/68d5069cc68404167c562973) and use the web interface.

The directory structure should be as follows:

```
|---spyrit-examples
|   |---2025_freeform
|   |   |---data
|   |   |   |---2025-09-25_freeform_publication
|   |   |   |   |---obj_Nothing_source_white_LED_black_4096_...
|   |   |   |   |---...
|   |   |   |   |---obj_StarSector_source_white_LED_black_4096_...
|   |   |   |   |---...
|   |   |   |   |---obj_StarSector_source_white_LED_hadam1d_cat_8192_...
|   |   |   |   |   |---overview
|   |   |   |   |   |---obj_StarSector_source_..._metadata_IDScam_before_acq.npy
|   |   |   |   |   |---obj_StarSector_source_..._metadata.json
|   |   |   |   |   |---obj_StarSector_source_..._spectraldata.npz
|   |   |   |   |---...
|   |   |---figures
|   |   |---cat_roi.png
|   |   |---dmd_patterns.py
|   |   |---download_data.py
|   |   |---figure_1.py
|   |   |---...
|   |   |---table_1.py
```

## How to reproduce the results of the paper?
1. Run `Figure_xx.py` for `xx` in `{2, 4, 6, 7, 8, 9}` to reproduce the corresponding figure in the paper.

1. Run `table_1.py` to evaluate numerically the expression of the trace reported in Table 1.