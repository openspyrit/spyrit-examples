# Optics Express paper, in preparation
*Authors:* N Ducros, A Lorente Mur, G. Beneti-Martin, L Mahieu-Williame
*Contact:* nicolas.ducros@insa-lyon.fr, CREATIS Laboratory, University of Lyon, France.

## Preliminary notice
The scripts `*spyrit_v2.ipynb` relies on spyrit v2 (still under develoment, see the `New_Fadoua` branch)
The other scripts relies on the `ieee_tci_branch` branch of spyrit v1

## Install the dependencies
1. We recommend creating a virtual (e.g., conda) environment first.

    ```shell
    # conda (or pip) install
    conda create --name new-env
    conda activate new-env
    conda install -c anaconda spyder
    conda install -c conda-forge jupyterlab
    conda install -c anaconda scikit-image
    conda install -c anaconda h5py 
    conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch # for windows
    ```

    Alternatively, you can clone an existing environment with `conda create --name new-env --clone existing-env`

1. Clone the spyrit package, and install the version in the `ieee_tci_branch` (or `New_Fadoua`) branch: 

    ```shell
    git clone https://github.com/openspyrit/spyrit.git
    git checkout ieee_tci_branch
    pip install -e ./spyrit
    ```

## Get the scripts, networks and raw data
1.  Get source code and navigate to the `/2022_OE/` folder

    ```shell
    git clone https://github.com/openspyrit/spyrit-examples.git
    cd spyrit-examples/2022_OE/ 
    ```
    
2. Download the trained EM-Net models and raw data at this [url](https://www.creatis.insa-lyon.fr/~ducros/spyritexamples/2022_SPIE_OE/2022_SPIE_OE.zip) and unzip the folder.
* Windows PowerShell

```powershell
wget https://www.creatis.insa-lyon.fr/~ducros/spyritexamples/2022_SPIE_OE/2022_SPIE_OE.zip -outfile data.zip
tar xvf data.zip 
```

The directory structure should be
```
|---spyrit-examples
|   |---2022_OE
|   |   |---data
|   |   |   |---
|   |   |---models
|   |   |   |---
|   |   |---expe_data_analysis.py
|   |   |---
```

NB: We use the following datasets in `2022_OE/recon_check_pinv_mmse.ipynb`:
* cat 
* horse 
* star_sector
* tomato_slice

We use the following datasets in `recon_resolution_targets.ipynb`:
* star_sector_x12 
* star_sector_x2
* usaf_x12
* usaf_x2

We use the following datasets in `recon_tomato.ipynb`:
* tomato_slice_x1
* tomato_slice_x12
* tomato_slice_x2
* tomato_slice_x6

We do **NOT** use the following datasets:
* cat_linear
* star_sector_linear