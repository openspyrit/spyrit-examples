# Hands-on Session XX: Image Reconstruction using the PyTorch and SPyRiT Packages

This code was used during a hands-on session given at the [Deep Learning for Medical Imaging School 2025](https://deepimaging2025.sciencesconf.org/).

The session was a practical introduction to image reconstruction, considering the limited-angle computed tomography problem. Participants were invited to run the cells, answer the questions, and fill in blanks in the code of `main.ipynb`. All answers and the solution code are given in `main_with_answers.ipynb`

The hands-on session followed a lecture on the topic. 
* [Slides](https://www.creatis.insa-lyon.fr/~ducros/hands_on/2023_DLMIS_ducros.pdf) 
* [Video](https://www.youtube.com/)

*Authors:* (version 2025) N Ducros, S Hariga, T Kaprelian, Lorena Leon, T Maitre.  
*Authors:* (version 2023) L Amador, E Chen, N Ducros, H-J Ling, K Mom, J Puig, T Grenier, E Saillard,      
*Authors:* (version 2021): N Ducros, T Leuliet, A Lorente Mur, Louise Friot-Giroux

*Contact:* nicolas.ducros@insa-lyon.fr, CREATIS Laboratory, University of Lyon, France.

# Install the dependencies
We recommend using a virtual (e.g., conda) environment.
```shell
conda create --name new-env
conda activate new-env
```
Our notebook relies on the [SPyRiT](https://github.com/openspyrit/spyrit) package that can be installed via `pip`.  The notebook was tested with version 3.0.
```shell
conda install pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install spyrit==3.0
pip install ipykernel
pip install scikit-image
pip install h5py
pip install opencv-python
```

# Get the scripts and data
1. Get source code from GitHub
   
        git clone https://github.com/openspyrit/spyrit-examples.git        
    
2. Go into `spyrit-examples/2025_DLMIS/`     

    ```
    cd spyrit-examples/2025_DLMIS/    
    ```

3. Download the image database at this [url](https://www.creatis.insa-lyon.fr/~ducros/spyritexamples/2023_DLMIS/data.zip) and extract its content

    * Windows PowerShell

    ```powershell
    wget https://www.creatis.insa-lyon.fr/~ducros/spyritexamples/2023_DLMIS/data.zip -outfile data.zip
    tar xvf data.zip 
    ```

    The directory structure should be

        |---spyrit-examples
        |   |---2025_DLMIS
        |   |   |---data
        |   |   |   |---
        |   |   |---main.ipynb
        |   |   |---main_with_answers.ipynb
        |   |   |---train.py

# At this this point you ready to go!
Try to complete `main.ipynb` or run `main_with_answers.ipynb`


# To go further
We provide `train.py` to train a network from a single command line

```shell
python train.py
```

By default, all networks are trained for 60 view angles during 20 epochs. For other values (e.g., 40 angles and 100 epochs), consider

```shell
python train.py --angle_nb 40 --num_epochs 100
```

To specify training parameters such as the batch size or learning rate, and for other options, type ` python train.py --help`