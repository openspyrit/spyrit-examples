# List of tutorial examples
The following tutorials show how to simulate data, perform image reconstruction and train a reconstruction network with spyrit toolbox for 2D single-pixel imaging using different image reconstruction methods. These can be run direclty in colab and present more specific cases than the main [spyrit tutorials](https://spyrit.readthedocs.io/en/latest/gallery/index.html) (refer to the main tutorials for an introduction to spyrit toolbox). 

Note: Notebooks may not open directly on github, but they can be downloaded and run locally or run directly on colab (this requires a google account).

### tuto_core_2d_drunet.ipynb : Data completion with DRUNet denoising  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/openspyrit/spyrit-examples/blob/tutorials/tutorial/tuto_core_2d_drunet.ipynb)     
Image reconstruction with different methods for single pixel imaging. We compare the following methods: 
- PinvNet : Pseudoinverse
- DCNet : Data completion network with UNet denoising
- DCDRUNet: Data completion network with pretrained DRUNet denoising. DRUNet allows to provide the noise level as input, as it is trained for all possible noise levels ([0, 255]) in a wide range of datasets. For more details on DRUNet, please refer to [Deep Plug-and-Play Image Restoration](https://github.com/cszn/DPIR). 


### train_gen_meas.py: Function to train a reconstruction network
This function allows to parse different data types and network and reconstruction parameters. Thus, it permits to use the same function to train a wide range of scenarios. You can select a wide range of parameters:

- Measurement type:
- Acquisition
- Network and training
- Optimisation
- Tensorboard

It is called by the tutorials below.

### tuto_train_lin_meas_colab.ipynb: Train reconstruction network for different measurement operators (eg. linear) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/openspyrit/spyrit-examples/blob/tutorials/tutorial/tuto_train_lin_meas_colab.ipynb)

This notebook shows to train in colab a reconstruction network for linear measurements. We take the example of Hadamard positive matrix, but this can be changed by the desired matrix. It set the requirements, install spyrit and spyrit-examples in colab, define all the parameters and calls *train_gen_meas.py*. 


### Train reconstruction network for split measurements [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/openspyrit/spyrit-examples/blob/tutorials/tutorial/tuto_train_split_meas_colab.ipynb)

Shows how to train in colab a reconstruction network selecting acquisition, network and training parameters, for the case of split measurements. This notebook set the requirements, install spyrit and spyrit-examples, define all the parameters and calls *train_split_meas.py*. 

Current networks: 
- 'dc-net': Denoised Completion Network (DCNet). 
- 'pinv-net': Pseudo Inverse Network (PinvNet).
- 'upgd':  Unrolled proximal gradient descent (UPGD). 
