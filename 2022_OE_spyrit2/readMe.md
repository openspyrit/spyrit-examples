# OpenSpyrit: an Ecosystem for Reproducible Single-Pixel Hyperspectral Imaging 

The code in the current folder allows to reproduce the results that are reported in

> G. Beneti-Martin, L Mahieu-Williame, T Baudier, N Ducros, "OpenSpyrit: an Ecosystem for Reproducible Single-Pixel Hyperspectral Imaging," Optics Express, Vol. 31, No. 10, (2023). 

*DOI (open access):* https://doi.org/10.1364/OE.483937

*Preprint (main PDF):* https://hal.science/hal-03910077

*Preprint (supplemental document):* https://hal.science/hal-XXXXXXX 

*Contact:* nicolas.ducros@insa-lyon.fr, CREATIS Laboratory, University of Lyon, France.

Spyrit v2.3.1 is used in this code.

## Install the dependencies

To ensure reproducibility and compatibility, we recommend creating a new environment using the environment.yml file included in this folder. With conda for example, move to the folder containing the environment.yml file and run this command:

    ```shell
    conda env create -f environment.yml
    conda activate OpenSpyritPaper2022
    ```

The name of this new environment is `OpenSpyritPaper2022`.

If this doesn't work, follow these step-by-step instructions:

1. We recommend using a virtual (e.g., conda) environment.

    ```shell
    # conda install
    conda create --name OpenSpyritPaper2022_manual
    conda activate OpenSpyritPaper2022_manual
    ```

    Alternatively, you can clone an existing environment with `conda create --name new-env --clone existing-env`

2. If you have a cuda-compatible GPU, install torch [here](https://pytorch.org/get-started/locally/) or by using the following line:

    ```shell
    conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
    ```

3. Install the spyrit package (more details [here](https://github.com/openspyrit/spyrit)). Tested with spyrit 2.3.1

    ```shell
    conda install pip
    pip install spyrit==2.3.1
    ```
    
4. Move to your local folder where your environment is installed and clone and install the spas package (more details [here](https://github.com/openspyrit/spas)). Tested with spas v1.4.

    ```shell
    # get your environment path
    echo %CONDA_PREFIX%  # C:\Users\[name]\.conda\envs\OpenSpyritPaper
    # move to that location
    cd C:\Users\[name]\.conda\envs\OpenSpyritPaper2022_manual
    # move to the package installation folder
    cd Lib/site-packages
    pip install -e git+https://github.com/openspyrit/spas.git@v1.4#egg=spas
    ```

The spas package will be installed in: C:\Users\[name]\.conda\envs\OpenSpyritPaper\Lib\site-packages\src\

5. You may need to install OpenCV. When running the code, if you get

    ```shell
    ModuleNotFoundError: No module named 'cv2'
    ```
    
    Then try in your environment:
    
    ```shell
    pip install opencv-python 
    ```

## Get the scripts, networks and raw data

1.  Get source code and navigate to the `/2022_OE_spyrit2/` folder

    ```shell
    git clone https://github.com/openspyrit/spyrit-examples.git
    cd spyrit-examples/2022_OE_spyrit2/ 
    ```
    
2. Download models, statistics, and raw data

Get all files by running `download_data.py`, e.g.,
```shell
python3 download_data.py
```

Otherwise, select

* the models from this [link](https://pilot-warehouse.creatis.insa-lyon.fr/#collection/6140ba6929e3fc10d47dbe3e/folder/638630794d15dd536f04831e) 

* the covariance matrices from this [link](https://pilot-warehouse.creatis.insa-lyon.fr/#collection/6140ba6929e3fc10d47dbe3e/folder/63d7f3620386da2747641e1b) 

* the raw data from this [link](https://pilot-warehouse.creatis.insa-lyon.fr/#collection/6140ba6929e3fc10d47dbe3e/folder/6149c3ce29e3fc10d47dbffb).


The directory structure should be

```
|---spyrit-examples
|   |---2022_OE_spyrit2
|   |   |---data
|   |   |   |---
|   |   |---model
|   |   |   |---
|   |   |---stat
|   |   |   |---
|   |   |---fig8_recon.py
|   |   |---fig8_mask.py
|   |   |---
|   |   |---recon
```


## Train the network from scratch
```powershell
./train.py --M 2048 --img_size 128 --batch_size 256
```
