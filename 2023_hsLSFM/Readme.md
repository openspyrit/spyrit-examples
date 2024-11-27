# Quantitative hyperspectral microscopy using encoded illumination and neural networks with physics priors

This code generates the figures for the paper draft and allow to study the data used.

*Authors:* S Crombez, N Ducros

*Contact:* nicolas.ducros@insa-lyon.fr, CREATIS Laboratory, University of Lyon, France.

## Install the dependencies

* `main_recon_*.py` requires the `hspim` branch of the `spyrit` package.


## Get the notebooks and the data

1.  Get source code and navigate to the `/2023_Optica/` folder
    ```shell
    git clone https://github.com/openspyrit/spyrit-examples.git
    cd spyrit-examples/2023_Optica/ 
    ```

2. Download 

* The raw measurements can be found [here](https://pilot-warehouse.creatis.insa-lyon.fr/#collection/63caa9497bef31845d991351/folder/64218b0d0386da2747699efc).
* The neural networks used for reconstruction (i.e., by Tikhonov-Net) can be can be found [here](https://pilot-warehouse.creatis.insa-lyon.fr/#collection/63caa9497bef31845d991351/folder/6464d5f585f48d3da071893c).


Unzip the folders into `/data/`, respectively. The directory structure should be

```
|---spyrit-examples
|   |---2023_Optica
|   |   |---data
|   |   |   |---2023_02_28_mRFP_DsRed_3D
|   |   |   |	|---Analyse_out
|   |   |   |	|---Fig_images
|   |   |   |	|---Raw_data_chSPSIM_and_SPIM
|   |   |   |	|   |---data_2023_02_28
|   |   |   |	|   |   |---RUN0001
|   |   |   |	|   |   |---RUN0002
|   |   |   |	|   |   |---
|   |   |   |---2023_03_07_mRFP_DsRed_can_vs_had
|   |   |   |	|--- 
|   |   |   |	|   |Reference_spectra
|   |   |---fonction
|   |   |---Notebooks
|   |   |---env.yaml
|   |   |---Readme.md
|   |   |---model
|   |   |   |---tikho-net_unet_imagenet_ph_50_exp_N_512_M_128_epo_20_lr_0.001_sss_10_sdr_0.5_bs_20_reg_1e-07.pth
```
