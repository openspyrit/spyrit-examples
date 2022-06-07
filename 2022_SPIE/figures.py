# -*- coding: utf-8 -*-

# %%
from pathlib import Path
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

def crop_make_gif(filename, frame_folder=Path('.'), extention='*jpg', duration=1000, loop=0):
    crop_area = (813, 213, 2709, 1842)
    frames = [Image.open(image).convert('L').crop(crop_area) for image in frame_folder.glob(extention)]
    frame_one = frames[0]
    frame_one.save(filename+".gif", format="GIF", append_images=frames,
                   save_all=True, duration=duration, loop=loop)

def make_gif(filename, frame_folder=Path('.'), extention='*png', duration=1000, loop=0):
    frames = [Image.open(image).convert('L') for image in frame_folder.glob(extention)]
    frame_one = frames[0]
    frame_one.save(filename+".gif", format="GIF", append_images=frames,
                   save_all=True, duration=duration, loop=loop)
    
# %% Robustness reconstrcution
p = Path('robustness')
crop_make_gif('test_crop', p)

# %% Ground-truth
p = Path('ground-truth')
make_gif('ground-truth_png', p, '*png', 1000, 0)

# %% cat
p = Path('reconstruction')
target = 'cat' # 'cat' 'star_sec'
method = 'mmse'     # among 'em_net', 'mmse'
M = 2048
#
filename = f'{target}_{method}_{M}.npy'
data = np.load(p / filename)
data = np.flip(data, 1)
#
fig, axs = plt.subplots(1, 2)
fig.suptitle(filename)
im = axs[0].imshow(data[400,:,:])
fig.colorbar(im, ax=axs[0])

im = axs[1].imshow(data[0:900,:,:].sum(axis=0))
fig.colorbar(im, ax=axs[1])


# %% Color montage for all reconstructions 
p = Path('reconstruction')
target = 'cat' # 'cat' 'star_sec'

if target=='cat':
    prefix = 'Cat_LinearColoredFilter'
    
elif target=='star_sec':
    prefix = 'SeimensStar_whiteLamp_linear_color_filter'

filename = prefix + '_had_reco.npz'
    
data = np.load(p / filename)
data = data['arr_0']
data = np.moveaxis(data,-1,0)
#data = np.flip(data, 1)
data = np.flip(data, 2)

#-- Plot --#
fig, axs = plt.subplots(1, 2)
fig.suptitle(filename)
im = axs[0].imshow(data[400,:,:])
fig.colorbar(im, ax=axs[0])

im = axs[1].imshow(data[0:900,:,:].sum(axis=0))
fig.colorbar(im, ax=axs[1])

# %% Color montage for the reconstructions
from spas import spectral_slicing, read_metadata, plot_color

#-user-defined
p = Path('reconstruction')
lambda_min = 520
lambda_max = 630
n_channel_plot = 12

for target in ["cat","star_sec"]:
    for method in ['mmse','em_net']:
        for M in [2048, 1024, 512]:
            #- load metadata
            if target=='cat':
                prefix_meta = 'Cat_LinearColoredFilter'
                
            elif target=='star_sec':
                prefix_meta = 'SeimensStar_whiteLamp_linear_color_filter'
                
            _, acquisition_parameters, _, _ = read_metadata(p / (prefix_meta + '_metadata.json'))
            
            #- load recon
            filename = f'{target}_{method}_{M}.npy'
            data = np.load(p / filename)
            data = np.rot90(data, -1, axes=(1, 2))
            print(filename)
            
            #-- Plot --#
            plt.figure()
            F_bin, wavelengths_bin, _ = spectral_slicing(data, 
                 acquisition_parameters.wavelengths, lambda_min, lambda_max, 
                 n_channel_plot)
            plot_color(F_bin, wavelengths_bin)
            plt.savefig(filename[0:-4]+'_color.png',dpi=300)
            plt.clf()

# %% Color montage for the ground-truths
from spas import spectral_slicing, read_metadata, plot_color

p = Path('reconstruction')
target = 'star_sec' # 'cat' 'star_sec'

if target=='cat':
    prefix = 'Cat_LinearColoredFilter'
    
elif target=='star_sec':
    prefix = 'SeimensStar_whiteLamp_linear_color_filter'

filename = prefix + '_had_reco.npz'
_, acquisition_parameters, _, _ = read_metadata(p / (prefix + '_metadata.json'))
    
data = np.load(p / filename)
data = data['arr_0']
data = np.moveaxis(data,-1,0)
#data = np.flip(data, 1)
#data = np.flip(data, 2)
data = np.rot90(data, axes=(1, 2))

#-- Plot --#
fig = plt.figure()
F_bin, wavelengths_bin, _ = spectral_slicing(data, acquisition_parameters.wavelengths, lambda_min, lambda_max, n_channel_plot)
plot_color(F_bin, wavelengths_bin)
fig.savefig(target+'_4096_color.png',dpi=300)


# %% 
from spas import generate_colormap, ListedColormap, spectral_slicing, read_metadata, plot_color
import imageio

p = Path('reconstruction')
gif_path = Path('gif2')

lambda_min = 520
lambda_max = 660

ims = []

for filename in list(p.glob('*.npy')):
#for filename in [p / 'cat_em_net_2048.npy']:
   
    print(filename)
    #-- metadata
    if 'cat' in filename.stem:
        prefix = 'Cat_LinearColoredFilter'
        
    elif 'star_sec' in filename.stem:
        prefix = 'SeimensStar_whiteLamp_linear_color_filter'

    _, acquisition_parameters, _, _ = read_metadata(p / (prefix + '_metadata.json'))

    #-- data
    cube = np.load(filename)
    cube = np.rot90(cube, -1, axes=(1,2))
    #cube = np.rot90(cube, axes=(1,2))

    F_bin_cube, wavelengths_bin, bin_width = spectral_slicing(cube, acquisition_parameters.wavelengths, lambda_min, lambda_max, 256)
    
    bin_ = F_bin_cube.shape[0]
    img_size = cube.shape[2]
    gamma = 0.8
    
    Path(gif_path/'tmp').mkdir(parents=True, exist_ok=True)
    
    for bin_num in range(bin_):
        color = generate_colormap(wavelengths_bin[bin_num], img_size, gamma)
        colormap = ListedColormap(color)
        
        plt.figure
        plt.imshow(F_bin_cube[bin_num,:,:], cmap = colormap)
        plt.axis('off')
        plt.title('\u03BB' + '=' + str(round(wavelengths_bin[bin_num])) + 'nm')
        #
        im = Path(gif_path/'tmp')/f'{bin_num}.png'
        ims.append(im)
        plt.savefig(im)
        plt.close()
        
    fps = 50
    with imageio.get_writer(gif_path / (filename.stem +'.gif'), mode='I', fps = fps) as writer:
        for im in ims:
            image = imageio.imread(im)
            writer.append_data(image)

    #_ remove temp folder and its content
    pth_tmp = gif_path/'tmp'
    for child in pth_tmp.glob('*.png'):
            child.unlink()
    pth_tmp.rmdir()