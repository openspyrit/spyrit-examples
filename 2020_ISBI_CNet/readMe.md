# A Completion Network for Reconstruction from Compressed Acquisition (ISBI 2020)
We provide the code that produces the results that we report in 

> Nicolas Ducros, A Lorente Mur, F. Peyrin. A Completion Network for Reconstruction from Compressed Acquisition. 2020 IEEE 17th International Symposium on Biomedical Imaging (ISBI), Apr 2020, Iowa City, United States, pp.619-623, ⟨10.1109/ISBI45749.2020.9098390⟩.
> [Download PDF](https://hal.archives-ouvertes.fr/hal-02342766/document/).

*License:* The SPIHIM datasets are distributed under the Creative Commons Attribution 4.0 International license ([CC-BY 4.0](https://creativecommons.org/licenses/by/4.0/))

*Contact:* nicolas.ducros@insa-lyon.fr, CREATIS Laboratory, University of Lyon, France.

## Running the code
0. Create a virtual environment apt to run this code :
```
shell $ python3 -m venv spyrit-env
shell $ source spyrit-env/bin/activate
```
1. Install spyrit (with all it's dependencies) in that virtual environment :
```
(spyrit-env) shell $ pip install -e spyrit
```
```
pip install -r extra_requirements.txt
```
2. Configure JupyterLab on the virtual environement : 
Install ipykernel which provides the IPython kernel for Jupyter
```
(spyrit-env) shell $ python -m pip install ipykernel
```
Next you can add your virtual environment to Jupyter :
```
(spyrit-env) shell $ ipython kernel install --user --name=spyrit-env
```

3. Create a symbolic link to a repository containing stl-10
```
(spyrit-env) shell $ ln -s <name of parent folder of stl-10> /data/
```
If you do not have stl-10 downloaded in your computer, then the dataset will automatically be downloaded. 

4. Simply launch JupiterLab from the current folder
```
(spyrit-env) shell $ jupyter Lab
```
and run `main.ipynb` on the kernel named `spyrit-env`. Note that this notebook downloads trained networks from this [url](https://www.creatis.insa-lyon.fr/~ducros/spyritexamples/2020_ISBI_CNet/2020_ISBI_CNet.zip). 

5. Alternatively, we provide `train.py` to train the different variants of the network. In a terminal:
```
(spyrit-env) shell $ train 
(spyrit-env) shell $ train --net_arch 1
(spyrit-env) shell $ train --net_arch 2
```
Note that 
* the models are saved in the default folder `.\models\`. To save them at another location consider
```
(spyrit-env) shell $ python train --model_root myfolder
```
* The defaults training parameters can be changed. For instance, run 
```
(spyrit-env) shell $ train --num_epochs 10 --batch_size 512 --lr 1e-8
```
to train your network for 10 epochs, with a batch size of 512, and a learning rate of 1e-8. 
* you can keep `Average_64x64.npy`, `Cov_64x64.npy` and `Var_64x64.npy` in `.\stats\`, if you have downloaded them, to avoid their computation that can be time-consumming.
