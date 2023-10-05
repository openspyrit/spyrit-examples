# List of colab tutorials
The following tutorials show how to simulate data, perform image reconstruction and train a reconstruction network with spyrit toolbox for 2D single pixel images using different image reconstruction methods. They can be run directly in colab and present more advanced cases than the main [spyrit tutorials](https://spyrit.readthedocs.io/en/latest/gallery/index.html) (see the main tutorials for an introduction to the spyrit toolbox).

> [!WARNING]
> Notebooks may not open on github, but they can be run locally directly on colab (this requires a google account).

## 1. Data simulation and image reconstruction for linear measurements using pinvNet with a CNN denoiser (`tuto_pinvnet_cnn.ipynb`) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/openspyrit/spyrit-examples/blob/master/tutorial/tuto_pinvnet_cnn.ipynb)     

We simulate data for a linear forward operator and perform image reconstruction using PinvNet with a CNN denoising layer. The forward operator is chosen as the positive part of a Hadamard matrix, yet any linear operator can be used instead. The forward operator, the pseudo-inverse reconstruction and the denoiser are layers of PinvNet, where only the last layer has learnable parameters. 

This tutorial is a colab version of the documentation tutorial [Pseudoinverse solution + CNN denoising](https://spyrit.readthedocs.io/en/master/gallery/tuto_pseudoinverse_cnn_linear.html#sphx-glr-gallery-tuto-pseudoinverse-cnn-linear-py). 

For training the network, refer to tuto_train_lin_meas_colab.ipynb.

## 2. Comparison of single-pixel image reconstructions (`tuto_core_2d_drunet.ipynb`) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/openspyrit/spyrit-examples/blob/master/tutorial/tuto_core_2d_drunet.ipynb)     

We compare several single-pixel image reconstruction methods: 
- *PinvNet*: Pseudoinverse
- *DCNet*: Denoised completion network with UNet denoising (see [ here](https://doi.org/10.1364/OE.424228))
- *DCDRUNet*: Denoised completion network with pretrained DRUNet denoising. DRUNet allows to provide the noise level as input, as it is trained for all possible noise levels ([0, 255]) in a wide range of datasets. For more details on DRUNet, please refer to [Deep Plug-and-Play Image Restoration](https://github.com/cszn/DPIR). 

## 3. Training a reconstruction network for linear measurements (`tuto_train_lin_meas_colab.ipynb`) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/openspyrit/spyrit-examples/blob/master/tutorial/tuto_train_lin_meas_colab.ipynb)

This notebook shows how to train a reconstruction network for linear measurements. We take the Hadamard positive matrix as an example, but any measurement matrix can be considered. It sets up the requirements, installs spyrit and spyrit-examples in colab, defines all parameters and calls `train_gen_meas.py`. 

The `train_gen_meas.py` function allows to parse different data types, networks and reconstruction parameters. This allows a wide range of scenarios to be considered:
- Measurement operator
- Network and training
- Optimisation
- Use of Tensorboard

## 4. Training a reconstruction network for linear split (pos/neg) measurements (`tuto_train_split_meas_colab.ipynb`) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/openspyrit/spyrit-examples/blob/master/tutorial/tuto_train_split_meas_colab.ipynb)

This notebook shows how to train a reconstruction network in the case of a linear forward operator split into its positive and negative components, as often encountered in single pixel imaging.

The following methods are considered
- *PinvNet*: Pseudo-inverse network
- *DCNet*: Denoised Completion Network (see [here](https://doi.org/10.1364/OE.424228))
- *UPGD*: Unrolled Proximal Gradient Descent

The notebook is mainly based on `train_split_meas.py`.

## 5. Launch tensorboard (`launch_tensorboard_colab.ipynb`) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/openspyrit/spyrit-examples/blob/master/tutorial/launch_tensorboard_colab.ipynb)
Tutorial to launch tensorboard simultaneously to training in another notebook. 

