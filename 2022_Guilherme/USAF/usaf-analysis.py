import numpy as np
import spyrit.misc.walsh_hadamard as wh
from matplotlib import pyplot as plt
from scipy.signal import find_peaks

from spas import read_metadata, reconstruction_hadamard
from spas import ReconstructionParameters, setup_reconstruction, load_noise, reconstruct

H = wh.walsh2_matrix(64)/64

f = np.load('./data/2021-10-22_magnification_usaf_WhiteLED/zoom_x1_usaf_group1/zoom_x1_usaf_group1_spectraldata.npz')

spectral_data = f['spectral_data']

metadata, acquisition_metadata, spectrometer_parameters, dmd_parameters = \
    read_metadata('./data/2021-10-22_magnification_usaf_WhiteLED/zoom_x1_usaf_group1/zoom_x1_usaf_group1_metadata.json')

recon = reconstruction_hadamard(acquisition_metadata.patterns, 'walsh', H, spectral_data)

recon_sum = np.sum(recon,axis=2) # Summing all the wavelength components

#%%

plt.imshow(recon_sum, cmap='gray')
y = np.arange(29,37)
x = np.ones(len(y), dtype=int)*50
plt.scatter(x=x,y=y,s=6,color='red')

#%%
def usaf_analysis(img, x, y):
    
    values = img[x,y]
    maxima,_ = find_peaks(values)
    minima,_ = find_peaks(-values)
    
    plt.figure()
    plt.plot(values)
    plt.scatter(minima,values[minima])
    plt.scatter(maxima,values[maxima])
    
    contrast = np.zeros(3)

    Imax = values[maxima]
    Imin = values[minima]

    contrast[0] = (Imax[0] - Imin[0])/(Imax[0] + Imin[0])
    contrast[1] = (Imax[1] - Imin[0])/(Imax[1] + Imin[0])
    contrast[2] = (Imax[2] - Imin[1])/(Imax[2] + Imin[1])

    if contrast.all() > 0.1:
        print(True)
    else:
        print(False)

#%%
for i in range(48,52+1):
    x = np.ones(len(y), dtype=int)*i
    usaf_analysis(recon_sum,y,x)

#%%

plt.imshow(recon_sum, cmap='gray')
x = np.arange(41,49)
y = np.ones(len(x), dtype=int)*30
plt.scatter(x=x,y=y,s=6,color='red')
usaf_analysis(recon_sum,y,x)
