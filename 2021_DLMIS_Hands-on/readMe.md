# Hands-on session 3.1: image reconstruction using the PyTorch and Spyrit packages

This code was used for a hands-on session given at the [Deep Learning for Medical Imaging School 2021](https://deepimaging2021.sciencesconf.org/).

The purpose of the session was to practice image reconstruction, considering the limited-angle computed tomography problem. Participants were invited to run the cells, answer the questions, and fill in blanks in the code of `main.ipynb`. Answers and solution code are given in `main_with_answers.ipynb`

The hands-on session followed a scientific presentation. Check the [slides](https://www.creatis.insa-lyon.fr/~ducros/hands_on/2021_Ducros_DLMIS.pdf) or watch the [video](https://www.youtube.com/watch?v=Q5s5P3luqOE).

*Authors:* N Ducros, T Leuliet, A Lorente Mur, L Friot--Giroux, T. Grenier

*Contact:* nicolas.ducros@insa-lyon.fr, CREATIS Laboratory, University of Lyon, France.

## Running the code on floyhub
1. Create a new workspace in your personal project:
* Press the ’Create Workspace’ button
* Select ’Start from scratch’
* Set ’Environment’ to ’Pytorch 1.7’ and ’Machine’ to ’GPU’
* Press the ’Create environment’ button

2. Enter in the new workspace (Warning: this step can take several minutes)

3. Add dataset using the right-hand panel:
* Type `dlmis21_dataset`
* Click on the ’Attach dataset’ button

4. In a ’New Launcher’ (directly accessible, if not, via File > New Launcher) choose’Terminal’ and run the following command lines

        wget https://www.creatis.insa-lyon.fr/~ducros/hands_on/start.sh
        bash ./start.sh
        rm start.sh

    NB: The downloads and installation typically take two minutes

5. Launch the jupyter notebook     `/floyd/home/spyritexamples-master/2021_DLMIS_Hands-on/main.ipyn`

## Running the code on a local installation (being tested)
1. We recommend you to use virtual environment.

1. Install spyrit and a requirement
        
        pip install spyrit
        pip install h5py

1. Retrieve spyrit source code via git :
        
        git clone https://github.com/openspyrit/spyrit-examples.git        
        
1. Go into `spyrit-examples/2021_DLMIS_Hands-on/`     

6. Open jupyter notebook    

        pip install notebook
        jupyter notebook
        
1. Create a `data` folder and go into it.

3. Download dataset https://www.creatis.insa-lyon.fr/~ducros/hands_on/datasets-dlmis21.tar

1. Extract the dataset (there is an error in `tar: Unexpected EOF in archive`, even if the files appear with the good size (1.5Go = matrices 860Mo + nets 586Mo)) :    

        tar xvf datasets-dlmis21.tar 

1. Update the dataset path in the notebook `main.ipynb`: look for the two occurences of `data_root =` and replace the quoted value with the path to your data folder, containing matrices and nets subfolders.
