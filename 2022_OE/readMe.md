# OE/OL paper, in preparation

*Authors:* N Ducros, G. Beneti-Martin, L Mahieu-Williame

*Contact:* nicolas.ducros@insa-lyon.fr, CREATIS Laboratory, University of Lyon, France.

# Install the dependencies

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

    Alternatively, you can clone an existing environment with `conda create --name new-env --clone existing-env `

    Our scripts primarily relies on the [SPyRiT ](https://github.com/openspyrit/spyrit) package that can be installed via `pip`.  NB: On Windows, you need to install [torch](https://pytorch.org/get-started/locally/) before SPyRiT

    ```shell
    # pip install
    pip install spyrit # tested with spyrit==1.2.0
	pip install spas   # tested with spas== ??1.1.0
    ```

# Get the scripts and data

1. Get source code from GitHub
   
        git clone https://github.com/openspyrit/spyrit-examples.git        
    
4. Go into `spyrit-examples/2022_OE/`     

    ```
    cd spyrit-examples/2022_OE/    
    ```

3. Download the measurements and trained model at this [url](https://www.creatis.insa-lyon.fr/~ducros/spyritexamples/2021_DLMIS_Hands-on/data.zip) and extract its content

    * Windows PowerShell

    ```powershell
    wget https://www.creatis.insa-lyon.fr/~ducros/spyritexamples/2022_OE/2022_OE.zip -outfile data.zip
    tar xvf data.zip 
    ```

    The directory structure should be

        |---spyrit-examples
        |   |---2022_OE
        |   |   |---data
        |   |   |   |---
        |   |   |---models
        |   |   |   |---
        |   |   |---stats
        |   |   |   |---
        |   |   |---analysis.py
        |   |   |---

