# Hands-on Session 4: Image Reconstruction using the PyTorch and SPyRiT Packages

This code was used during a hands-on session given at the [Deep Learning for Medical Imaging School 2023](https://deepimaging2023.sciencesconf.org/).

The session was a practical introduction to image reconstruction, considering the limited-angle computed tomography problem. Participants were invited to run the cells, answer the questions, and fill in blanks in the code of `main.ipynb`. All answers and the solution code are given in `main_with_answers.ipynb`

> To be updated!! 
The hands-on session followed a presentation. Check the [slides](https://www.creatis.insa-lyon.fr/~ducros/hands_on/2021_Ducros_DLMIS.pdf) or watch the [video](https://www.youtube.com/watch?v=Q5s5P3luqOE).

*Authors:* N Ducros, T Leuliet, A Lorente Mur, L Friot-Giroux, T. Grenier

*Contact:* nicolas.ducros@insa-lyon.fr, CREATIS Laboratory, University of Lyon, France.

# Install the dependencies

> NB: we should check this carefully!

1. We recommend creating a virtual (e.g., conda) environment first. 


```shell
# conda (or pip) install
conda create --name new-env
conda activate new-env
```
Our scripts primarily relies on the [SPyRiT](https://github.com/openspyrit/spyrit) package that can be installed via `pip`.  NB: On Windows, you may need to install [torch](https://pytorch.org/get-started/locally/) before SPyRiT

```shell
# pip install
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install spyrit # tested with spyrit==2.0.0
python -m pip install -U scikit-image
pip install h5py
pip install jupyterlab
```

# Get the scripts and data

1. Get source code from GitHub
   
        git clone https://github.com/openspyrit/spyrit-examples.git        
    
4. Go into `spyrit-examples/2023_DLMIS/`     

    ```
    cd spyrit-examples/2023_DLMIS/    
    ```

3. Download the image database at this [url](https://www.creatis.insa-lyon.fr/~ducros/spyritexamples/2023_DLMIS/data.zip) and extract its content

    * Windows PowerShell

    ```powershell
    wget https://www.creatis.insa-lyon.fr/~ducros/spyritexamples/2021_DLMIS_Hands-on/data.zip -outfile data.zip
    tar xvf data.zip 
    ```

    The directory structure should be

        |---spyrit-examples
        |   |---2021_DLMIS_Hands-on
        |   |   |---data
        |   |   |   |---
        |   |   |---main.ipynb
        |   |   |---main_with_answers.ipynb
        |   |   |---train.py

4. Open JupyterLab environment and create a kernel (e.g., dlmis21) corresponding to your current conda environment 

        ipython kernel install --user --name=dlmis21
        jupyter lab

# Training a model from scratch

We provide `train.py` to train a network from a single command line

```shell
python train.py
```

By default, all networks are trained for 60 view angles during 20 epochs. For other values (e.g., 40 angles and 100 epochs), consider

```shell
python train.py --angle_nb 40 --num_epochs 100
```

To specify training parameters such as the batch size or learning rate, and for other options, type ` python train.py --help`

