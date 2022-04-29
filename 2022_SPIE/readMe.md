# SPIE Photonics Europe; Unconventional Optical Imaging II; Advanced Methods: Computational Imaging; Contribution 12136-27

*Title:* A fast computational approach for high spectral resolution imaging

*Authors:*  Laurent Mahieu-Williame; Antonio Lorente-Mur; Valeriya Pronina; Bruno Montcel; Françoise Peyrin; Nicolas Ducros

*Contact:* nicolas.ducros@insa-lyon.fr, CREATIS Laboratory, University of Lyon, France.

## Install the dependencies

1. We recommend creating a virtual (e.g., conda) environment first.

    ```shell
    # conda (or pip) install
    conda create --name new-env
    conda activate new-env
    conda install -c anaconda spydery
    conda install -c conda-forge jupyterlab
    conda install -c anaconda scikit-image
    conda install -c anaconda h5py 
    conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch # for windows
    ```

    Alternatively, you can clone an existing environment with `conda create --name new-env --clone existing-env`

1. Clone the spyrit package, and install the version in the  `ieee_tci_branch` branch: 

    ```shell
    git clone https://github.com/openspyrit/spyrit.git
    git checkout ieee_tci_branch
    pip install -e ./spyrit
    ```

## Get the scripts, networks and raw data

1.  Get source code and navigate to the `/2022_SPIE/` folder

    ```shell
    git clone https://github.com/openspyrit/spyrit-examples.git
    cd spyrit-examples/2022_SPIE/ 
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
|   |---2022_SPIE
|   |   |---data
|   |   |   |---
|   |   |---models
|   |   |   |---
|   |   |---SPIE_europe_figures.py
|   |   |---
```

NB: We only use the following datasets in `2022_SPIE`:

* cat_linear
* star_sector_linear
