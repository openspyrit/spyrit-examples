# List of tutorial examples

Tutorial examples for different reconstruction algorithms. 

Note: Notebooks may not open directly on github, but they can be downloaded and run locally or run directly on colab (this requires a google account).

## Image reconstruction tutorials
The following tutorials show how to simulate data and perform image reconstruction with spyrit toolbox for 2D single-pixel imaging for different image reconstruction methods. 

### DC DRUNet : Data completion with DRUNet denoising   [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/openspyrit/spyrit-examples/blob/tutorials/tutorial/tuto_core_2d_short_drunet.ipynb)

Image reconstruction with data completion and pretrained DRUNet denoising (DCDRUNet). DRUNet allows to provide the noise level as input, as it is trained for all possible noise levels ([0, 255]) in a wide range of datasets. For more details on DRUNet, please refer to [Deep Plug-and-Play Image Restoration](https://github.com/cszn/DPIR). 
