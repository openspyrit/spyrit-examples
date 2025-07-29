# Hyperspectral LSFM with DMD-only Shaping and Neural Network Reconstruction

We provide the code to reproduce the results reported in

> Sébastien Crombez, Cédric Ray, Chloé Exbrayat-Héritier,Florence Ruggiero, Nicolas Ducros, "*Hyperspectral LSFM with DMD-only Shaping and Neural Network Reconstruction*," (2025). 

<!---
* [DOI (open access)](https://doi.org/10.1364/OE.559227)
-->
* [Main PDF](https://hal.science/hal-04824372/document)
* [Supplementary PDF](https://hal.science/hal-04824372/preview/hspim_nature_supp.pdf)

*Contact:* nicolas.ducros@insa-lyon.fr, CREATIS Laboratory, University of Lyon, France.

## Installation
1. Create a conda environment
    ```shell
    conda create --name hsflsm
    conda activate hsflsm
    ```

1. Install pytorch using conda. E.g.,
    ```shell
    conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia
    conda install pip
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
    ```
    Visit https://pytorch.org/get-started/locally/ if you need a different installation.

1. Install SPyRiT **v2.4.1** and a few more packages:

    ```shell
    pip install spyrit==2.4.1
    pip install spyder-kernels
    pip install opencv-python
    pip install pysptools
    pip install girder-client
    pip install natsort
    ```

> opencv-python is required for spectral registration. It is not only necessary to run `main_spectal_registration.py`.

> pysptools is required for spectral unmixing. It is not only necessary to run `main_unmix_filter_EGFP-DsRed_14.py` and `main_unmix_filter_mRFP-DsRed.py`.

> natsoft is required to sort the files. It is not only necessary to run `main_create_gif.py`.

## Code and data

1.  Get source code and navigate to the `/2025_hLSFM/` folder
    ```shell
    git clone https://github.com/openspyrit/spyrit-examples.git
    cd spyrit-examples/2025_hLSFM/ 
    ```

2. Download

* The raw measurements can be found [here](https://pilot-warehouse.creatis.insa-lyon.fr/#collection/63caa9497bef31845d991351/folder/63caaa937bef31845d991353).
* The neural networks used for reconstruction can be found [here](https://pilot-warehouse.creatis.insa-lyon.fr/#collection/63caa9497bef31845d991351/folder/6464d5f585f48d3da071893c). The covariance matrix used by the Tikhonov network can be found [here](https://pilot-warehouse.creatis.insa-lyon.fr/#collection/63caa9497bef31845d991351/folder/6464d58285f48d3da0718935).
* The reference spectra used for unmixing can be found [here](https://pilot-warehouse.creatis.insa-lyon.fr/#collection/63caa9497bef31845d991351/folder/64a56bad655dd021c0b08ef6).


The directory structure should be

```
|---spyrit-examples
|   |---2025_hLSFM
|   |   |---data
|   |   |   |---2023_02_28_mRFP_DsRed_3D
|   |   |   |	|---Raw_data_chSPSIM_and_SPIM
|   |   |   |	|   |---data_2023_02_28
|   |   |   |	|   |   |---RUN0001
|   |   |   |	|   |   |---RUN0002
|   |   |   |	|   |   |---
|   |   |   |---2023_03_07_mRFP_DsRed_can_vs_had
|   |   |   |	|--- 
|   |   |   |---2023_03_13_2023_03_14_eGFP_DsRed_3D
|   |   |   |	|---
|   |   |   |---Reference_spectra
|   |   |---fonction
|   |   |---model
|   |   |   |---tikho-net_unet_imagenet_ph_50_exp_N_512_M_128_epo_20_lr_0.001_sss_10_sdr_0.5_bs_20_reg_1e-07.pth
|   |   |   |---
|   |   |---stat
|   |   |   |---Cov_1_512x512.npy
|   |   |---main_colorize_EGFP-DsRed_14.py
|   |   |---main_colorize_mRFP-DsRed.py
|   |   |---
```

## Visualisation of the data

* [Dataset 1](https://pilot-warehouse.creatis.insa-lyon.fr/api/v1/file/6752fdfa74c908c83b35e030/download?contentDisposition=inline) shows the raw measurements for one slice of the Tg(fli1:EGFP;olig2:DsRed) sample.
* [Dataset 2](https://pilot-warehouse.creatis.insa-lyon.fr/api/v1/item/6753224074c908c83b35e39c/download?contentDisposition=inline) shows the abundance of EGFP (in green) and DsRed (in red) for all of the 25 slices of the Tg(fli1:EGFP;olig2:DsRed) sample. Many more visualizations of the abundance maps can be found [here](https://pilot-warehouse.creatis.insa-lyon.fr/#collection/63caa9497bef31845d991351/folder/675321a374c908c83b35e03c).
* [Dataset 3](https://pilot-warehouse.creatis.insa-lyon.fr/api/v1/item/67532a0274c908c83b35e8e1/download?contentDisposition=inline) shows the conventional LSFM images for 21 slices of the Tg(fli1:EGFP;olig2:DsRed) sample. Other conventional LSFM images of the sample can be found [here](https://pilot-warehouse.creatis.insa-lyon.fr/#collection/63caa9497bef31845d991351/folder/672e1f280e9f151150f4234c).
* [Dataset 4](https://pilot-warehouse.creatis.insa-lyon.fr/api/v1/file/675311ac74c908c83b35e036/download?contentDisposition=inline) shows the abundance of mRFP (in cyan) and DsRed (in red) for all of the 20 slices of the Tg(sox10:mRFP;olig2:DsRed) sample. Many more visualizations of the abundance maps can be found [here](https://pilot-warehouse.creatis.insa-lyon.fr/#collection/63caa9497bef31845d991351/folder/6750832874c908c83b353457).
* [Dataset 5](https://pilot-warehouse.creatis.insa-lyon.fr/api/v1/item/67533f5074c908c83b35e8ec/download?contentDisposition=inline) shows the conventional LSFM images for all slices of the Tg(sox10:mRFP;olig2:DsRed) sample. Other conventional LSFM images of the sample can be found [here](https://pilot-warehouse.creatis.insa-lyon.fr/#collection/63caa9497bef31845d991351/folder/672e1edb0e9f151150f4230e).


## How to reproduce the results of the paper?

### Figure 3: Fellgett's advantage and Tikhonov-Net

> All the results are saved in `.\data\2023_03_07_mRFP_DsRed_can_vs_had\`. They can also be found on our warehouse (see [here](https://pilot-warehouse.creatis.insa-lyon.fr/#collection/63caa9497bef31845d991351/folder/642d24900386da274769abd4)). 

1. Run `main_preprocess_Fellgett.py` to generate the preprocessed measurements that will be saved in the subfolder `.\Preprocess\`
1. Run `figure_3.py` to reconstruct hypercubes that will be saved in the subfolder `.\Reconstruction\hypercube\` and generate Fig. 3 of the paper.

### Figure 4: EGFP-DsRed sample

> All the results are saved in `.\data\2023_03_13_2023_03_14_eGFP_DsRed_3D\`. They can also be found on our warehouse (see [here](https://pilot-warehouse.creatis.insa-lyon.fr/#collection/63caa9497bef31845d991351/folder/6708d7990e9f151150f3c100)). 

1. Run `main_preprocess_EGFP-DsRed.py` to generate the preprocessed measurements that will be saved in the subfolder `.\Preprocess\`
1. Run `main_recon_net_EGFP-DsRed_14_all_slices.py` to reconstruct the hypercubes that will be saved in the subfolder `.\Reconstruction\hypercube\`.
1. Run (the first sections of) `main_spectal_registration.py` to compensate for a spectral shift. The resulting hypercubes will be saved in the subfolder `.\Reconstruction\hypercube\*_shift\`.
1. Run `main_unmix_filter_EGFP-DsRed_14.py` to estimate the map of the different components (i.e., DsRed, EGFP, and autofluorescence) in the sample by both spectral filtering and spectral unmixing.
    * The maps obtained by spectral filtering will be saved in the subfolder `.\Filtering_shift\`.
    * The quantitative abundance maps obtained by spectral unmixing will be saved in the subfolder `.\Unmixing_shift\`.
1. Run `main_visual_EGFP-DsRed_14.py` to visualise the filter maps and quantitative maps in color. The resulting visualisations will be saved in the subfolder `.\Visualisation_shift\`.

### Figure 5: DsRed-mRFP sample

> All the results are saved in `.\data\2023_02_28_mRFP_DsRed_3D\`. They can also be found on our warehouse (see [here](https://pilot-warehouse.creatis.insa-lyon.fr/#collection/63caa9497bef31845d991351/folder/66ff9c49ae27f5ad8259f38a)). 

1. Run `main_preprocess_mRFP-DsRed.py` to generate the preprocessed measurements that will be saved in the subfolder `.\Preprocess\`
1. Run `main_recon_net_mRFP-DsRed_14_all_slices.py` to reconstruct the hypercubes that will be saved in the subfolder `.\Reconstruction\hypercube\`.
1. Run (the last sections of) `main_spectal_registration.py` to compensate for a spectral shift. The resulting hypercubes will be saved in the subfolder `.\Reconstruction\hypercube\*_shift\`.
1. Run `main_unmix_filter_EGFP-DsRed_14.py` to estimate the map of the different components (i.e., DsRed, EGFP, and autofluorescence) in the sample by both spectral filtering and spectral unmixing.
    * The maps obtained by spectral filtering will be saved in the subfolder `.\Filtering_shift\`.
    * The quantitative abundance maps obtained by spectral unmixing will be saved in the subfolder `.\Unmixing_shift\`.
1. Run `main_unmix_filter_mRFP-DsRed.py` to visualise the filter maps and quantitative maps in color. The resulting visualisations will be saved in the subfolder `.\Visualisation_shift\`.



### Miscellaneous

1. The script `main_pattern_full_visu.py` reproduces the illumination patterns of Fig. S6 of the supplementary document.
1. The scripts  `main_colorize_*.py` plot in color ("rainbow colors") 
    * the raw measurements in `./Preprocess/Run*/`,
    * the reconstructed hypercubes in `./Reconstruction/hypercube/tikhonet50_div1.5/RUN*/`. 
1. The script  `main_stat.py` shows examples to compute mean and covariance matrices "in 2D". To compute mean and covariance "in 1D", as required by the Tikhonov step of the reconstruction, see `stat_1.py` in `spyrit.misc.statistics`.
1. Training:
   * See `train.sh` for a typical shell script.
   * `train.py` is the main Python file (called in `train.sh`).
   * `train_plot.py` load and plot the training loss.