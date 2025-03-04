# SPyRiT: an open source package for single-pixel imaging based on deep learning

We provide here the code to reproduce the results reported in

> JFJP Abascal, T Baudier, R Phan, A Repetti, N Ducros, "SPyRiT: an open source package for single-pixel imaging based on deep learning," Preprint (2024). 

*Preprint view (main PDF + supplemental document):* https://hal.science/hal-04662876

*Preprint download (main PDF):* https://hal.science/hal-04662876/document

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
    git sparse-checkout set 2024_spyrit
    git checkout
    ```
## Installation

1. Create a conda environment
    ```shell
    conda create --name spyrit_2024
    conda activate spyrit_2024
    ```

1. Install pytorch using conda. E.g.,
    ```shell
    conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia
    ```
    Visit https://pytorch.org/get-started/locally/ if you need a different installation.

1. Install SPyRiT and a few more packages (until release of v3, checkout the `spyrit_dev` branch):
    ```shell
    pip install spyrit
    pip install girder-client
    pip install scikit-image
    ```

## Download the models and data

Run the `download_data.py` script from the `2024_spyrit` subfolder
```shell
cd spyrit-examples/2024_spyrit/ 
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
The directory structure should be as follows:

```
|---spyrit-examples
|   |---2024_spyrit
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
|   |   |   |---tomato_slice_2_zoomx2_spectraldata.npz
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
1. To reproduce the sampling masks, acquisition matrices, measurements, and images in Figure 2, run `figure_2.py`. 

2. To reproduce the reconstructions in Figures 3 and 4, run `figure_3.py` and `figure_4.py`, respectively. All images will be saved in `\2024_spyrit\recon\`

3. Run `table_1.py` to reproduce the metrics in Table 1.

4. Run `supplemental_figure_Sxx.py` for `xx` in `{1, ..., 8}` to reproduce all the figures in the supplemental document.