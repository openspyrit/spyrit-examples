# Dual-arm motion-compensated single-pixel imaging

This folder contains the code that generated the results of the followinp paper:

> T. Maitre, E. Bretin, L. Mahieu-Williame, R. Phan, M. Sdika, N. Ducros, "Dual-arm motion-compensated single-pixel imaging". IEEE Transactions on Image Processing, 35, 9730-9743 (2026). 
> [DOI](https://dx.doi.org/10.1109/TIP.2026.3730764), 
> [Main PDF](https://hal.science/hal-05068181/document), 
> [Supplementary PDF](https://hal.science/hal-05068181/preview/supplementary_v3.pdf).

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
    git sparse-checkout set 2025_dynamic_TIP
    git checkout
    ```

## Environment and dependencies

1. Create a conda environment
```shell
conda create --name dynamic
conda activate dynamic
```

2. Install SPyRiT. E.g., 

    * For CPU computation, run:
    ```shell
    conda install pip
    pip install spyrit
    ```

    * To use CUDA/GPU, install PyTorch first:
    ```shell
    conda install pip
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu132
    pip install spyrit
    ```

    Visit https://pytorch.org/get-started/locally/ if you need a different installation.

3. Install a few more useful packages
```shell
pip install ipykernel
pip install opencv-python
pip install git+https://github.com/openspyrit/spas.git
pip install nibabel
```

Code was tested using spyrit 3.1.1. For more detail, see `environment.yaml`

## Reproducibility of the results

Run `fig_XX.py` to reproduce the results associated to the Fig. XX of the paper.

> All script should be run from the `2025_dynamic_TIP` folder

> Some of the scripts should be run in increasing order to generate intermediate results.

## Contact
- For reproducibility issues, feel free to open an issue in our GitHub repository.

