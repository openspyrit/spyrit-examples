# -*- coding: utf-8 -*-

from pathlib import Path
from PIL import Image
from natsort import natsorted

def animated_gif_from_folder(gif_path, folder_path, pattern='*.png', duration=0.1):
    """
    Loads all images from a given folder and creates an animated GIF.

    Args:
        folder_path (str or Path): The path to the folder containing the images.
        
        gif_path (str or Path): The path to save the GIF file.
        
        duration (float, optional): Duration in seconds for each frame. Defaults to 0.1.
        
    Example:
        >>> folder_path = "/path/to/your/images"
        >>> gif_path = "animated.gif"
        >>> animated_gif_from_folder(gif_path, folder_path)
        
    """

    folder_path = Path(folder_path)
    gif_path = Path(gif_path)

    images = []
    for image_path in natsorted(folder_path.glob(pattern)):
        try:
            image = Image.open(image_path)
            pImage = image
            #pImage = image.quantize(colors=256, method=Image.FASTOCTREE, dither=0)
            images.append(pImage)
        except:
            print(f'ERROR: Unable to open {image_path}')

    images[0].save(gif_path, 
                   save_all=True, 
                   append_images=images[1:], 
                   duration=duration*1000,
                   loop=0)
    
#%%
from pathlib import Path
from misc_dev import animated_gif_from_folder

folder_main = Path(r"D:\Creatis\Programmes\openspyrit\spyrit-examples\2023_Optica\data\2023_03_13_2023_03_14_eGFP_DsRed_3D\Reconstruction\hypercube\tikhonet50_div1.5_shift")
folder_path = "T15_RUN0010_2023_03_14_color"
gif_path = "T15_RUN0010_2023_03_14_color.gif"
animated_gif_from_folder(folder_main/gif_path, folder_main/folder_path)

#%%
from pathlib import Path
from misc_dev import animated_gif_from_folder

folder_main = Path(r"D:\Creatis\Programmes\openspyrit\spyrit-examples\2023_Optica\data\2023_03_13_2023_03_14_eGFP_DsRed_3D\Raw_data_chSPSIM_and_SPIM")
folder_path = "spim"
gif_path = "spim.gif"
animated_gif_from_folder(folder_main/gif_path, 
                         folder_main/folder_path, 
                         duration=0.25)

#%%
from pathlib import Path
from misc_dev import animated_gif_from_folder

folder_main = Path(r"D:\Creatis\Programmes\openspyrit\spyrit-examples\2023_Optica\data\2023_02_28_mRFP_DsRed_3D\Raw_data_chSPSIM_and_SPIM")
folder_path = "spim"
gif_path = "spim.gif"
animated_gif_from_folder(folder_main/gif_path, 
                         folder_main/folder_path, 
                         duration=0.25)


#%%
from pathlib import Path
from misc_dev import animated_gif_from_folder

folder_main = Path(r"D:\Creatis\Programmes\openspyrit\spyrit-examples\2023_Optica\data\2023_03_13_2023_03_14_eGFP_DsRed_3D\Visualisation_shift\tikhonet50_div1.5_shift_NNLS")
folder_path = "qmap"
gif_path = "qmap_0.gif"
animated_gif_from_folder(folder_main/gif_path, 
                         folder_main/folder_path,
                         pattern = 'qmap_0_*.png',
                         duration=0.25)

gif_path = "qmap_01.gif"
animated_gif_from_folder(folder_main/gif_path, 
                         folder_main/folder_path,
                         pattern = 'qmap_01_*.png',
                         duration=0.25)