# Optica 2023 pulibcation : Quantitative hyperspectral microscopy using encoded illumination and neural networks with physics priors

This code generates the figures in the article and allow to study the data used.

*Authors:* S Crombez,N Ducros

*Contact:* nicolas.ducros@insa-lyon.fr, CREATIS Laboratory, University of Lyon, France.

## Install the dependencies

1. We recommend creating a virtual (e.g., conda) environment first.

    ```shell
    # conda install
    conda create --name new-env
    conda activate new-env
    conda install spyder
    conda install -c conda-forge matplotlib
    conda install -c conda-forge jupyterlab
    conda install -c anaconda scikit-image
    conda install -c anaconda h5py 
    conda install pytorch torchvision torchaudio cudatoolkit=11.6 -c pytorch -c conda-forge
    ```

    Alternatively, you can clone an existing environment with `conda create --name new-env --clone existing-env`

1. Clone the spyrit package, and install the version in the  `towards_v2_fadoua` branch

    ```shell
    git clone https://github.com/openspyrit/spyrit.git
    cd spyrit
    git checkout towards_v2
    pip install -e .
    ```
    
1. Clone the spas package: 

    ```shell
    git clone https://github.com/openspyrit/spas.git
    cd spas
    pip install -e .
    ```

## Get the scripts and networks

1.  Get source code and navigate to the `/2023_Opyica/` folder
    ```shell
    git clone https://github.com/openspyrit/spyrit-examples.git
    cd spyrit-examples/2023_Opyica/ 
    ```
2. Download the models from this [link](https://pilot-warehouse.creatis.insa-lyon.fr/#collection/Achanger) 

3. Download the average and covariance matrices from this [link](https://pilot-warehouse.creatis.insa-lyon.fr/#collection/Achanger)

The directory structure should be



