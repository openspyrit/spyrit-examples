# SPyRiT: an open source package for single-pixel imaging based on deep learning

We provide here the code to reproduce the results reported in

> JFJP Abascal, T Baudier, R Phan, A Repetti, N Ducros, "SPyRiT: an open source package for single-pixel imaging based on deep learning," Preprint (2024). 

*Preprint view (main PDF + supplemental document):* https://hal.science/hal-04662876v1

*Preprint download (main PDF):* https://hal.science/hal-04662876v1/file/submitted.pdf

*Preprint download (supplemental document):* https://hal.science/hal-04662876v1/file/supplemental.pdf

*Contact:* nicolas.ducros@insa-lyon.fr, CREATIS Laboratory, University of Lyon, France.

## Get the code from Github

You can only clone the code corresponding to this paper using the  `sparse-checkout` git command.
```shell
git clone -n --depth=1 --filter=tree:0 https://github.com/openspyrit/spyrit-examples
cd spyrit-examples
git sparse-checkout set 2024_Optics_Express
git checkout
```

Alternatively, you can clone the whole `spyrit-examples` repository (which includes code for some other papers):
```shell
git clone https://github.com/openspyrit/spyrit-examples.git
```

## Download the models and data

Run the `download_data.py` script from the `2024_Optics_Express` subfolder
```shell
cd spyrit-examples/2024_Optics_Express/ 
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
|   |---2024_Optics_Express
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
|   |   |---table_1.py
|   |   |---train.py
|   |   |---utility_dpgd.py
```

## Installation

### Method 1 (preferred, NOT WORKING): using environment.yml
Using a environment manager (e.g. conda), create an environment using the `environment.yml` file.
```shell
conda env create -f environment.yml
conda activate spyrit_optics_express_2024
```

### Method 2: install each module independently
1. Create a conda environment
    ```shell
    conda create --name spyrit_optics_express_2024
    conda activate spyrit_optics_express_2024
    ```

1. Install pytorch using conda. E.g.,
    ```shell
    conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia
    ```
    Visit https://pytorch.org/get-started/locally/ if you need a different installation.

1. Install SPyRiT and a few more packages:
    ```shell
    pip install spyrit==2.3.3
    pip install girder-client
    pip install scikit-image
    ```

## How to reproduce the paper's results?
1. To reproduce the sampling masks, acquisition matrices, measurements, and images in Figure 2, run `figure_2.py`. 

2. To reproduce the reconstructions in Figure 3 and 4, run `figure_3.py` and `figure_4.py`, respectively. All images are saved in `\2024_Optics_Express\recon\`

3. Run `table_1.py` to reproduce the metrics in Table 1. To limit the computation time, the code runs only on a subset of the ImageNet validation set. For this reason, the obtained metrics are close to but not equal to those reported in the paper. 

## Training the networks from scratch (NOT TESTED)

```powershell
./train.py --M 2048 --img_size 128 --batch_size 256
```