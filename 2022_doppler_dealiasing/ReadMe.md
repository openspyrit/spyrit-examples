## Install the dependencies

1. We recommend creating a virtual (e.g., conda) environment first.

    ```shell
    # conda install
    conda create --name new-env
    conda activate new-env
    conda install spyder
    conda install -c conda-forge jupyterlab
    conda install -c anaconda scikit-image
    conda install -c anaconda h5py 
    conda install pytorch torchvision torchaudio cudatoolkit=11.6 -c pytorch -c conda-forge
    ```

    Alternatively, you can clone an existing environment with `conda create --name new-env --clone existing-env`

1. Clone the spyrit package, and install the version in the  `towards_v2` branch: 

    ```shell
    git clone https://github.com/openspyrit/spyrit.git
    cd spyrit
    git checkout towards_v2
    pip install -e .
    ```