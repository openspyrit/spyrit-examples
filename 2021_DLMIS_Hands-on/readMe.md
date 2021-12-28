# Hands-on Session 3.1: Image Reconstruction using the PyTorch and Spyrit Packages

This code was used during a hands-on session given at the [Deep Learning for Medical Imaging School 2021](https://deepimaging2021.sciencesconf.org/).

The session was a practical introduction to image reconstruction, considering the limited-angle computed tomography problem. Participants were invited to run the cells, answer the questions, and fill in blanks in the code of `main.ipynb`. All answers and the solution code are given in `main_with_answers.ipynb`

The hands-on session followed a presentation. Check the [slides](https://www.creatis.insa-lyon.fr/~ducros/hands_on/2021_Ducros_DLMIS.pdf) or watch the [video](https://www.youtube.com/watch?v=Q5s5P3luqOE).

*Authors:* N Ducros, T Leuliet, A Lorente Mur, L Friot--Giroux, T. Grenier

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
    pip install spyrit # tested with spyrit==1.1.0
    ```

# Get the scripts and data

1. Get source code from GitHub
   
        git clone https://github.com/openspyrit/spyrit-examples.git        
    
4. Go into `spyrit-examples/2021_DLMIS_Hands-on/`     

    ```
    cd spyrit-examples/2021_DLMIS_Hands-on/    
    ```

3. Download the image database at this [url](https://www.creatis.insa-lyon.fr/~ducros/spyritexamples/2021_DLMIS_Hands-on/data.zip) and extract its content

    **Windows PowerShell**

    ```powershell
    wget https://www.creatis.insa-lyon.fr/~ducros/spyritexamples/2021_DLMIS_Hands-on/data.zip -outfile data.zip
    tar xvf data.zip 
    ```

    The directory structure should be

        |---spyrit-examples
        |   |---2021_DLMIS_Hands-on
        |	|	|---data
        |	|	|   |---
        |	|	|---main.ipynb
        |	|	|---main_with_answers.ipynb
        |	|	|---train.py

    

4. Open JupyterLab environment and select the kernel corresponding to your environment (e.g., dlmis21)

        ipython kernel install --user --name=dlmis21
        jupyter lab

# Training a model from scratch

