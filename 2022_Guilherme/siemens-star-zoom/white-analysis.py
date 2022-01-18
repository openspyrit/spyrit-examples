import numpy as np
import spyrit.misc.walsh_hadamard as wh
from matplotlib import pyplot as plt

from spas import read_metadata, reconstruction_hadamard

H = wh.walsh2_matrix(64)/64

f = np.load('./data/zoom_x1_white/zoom_x1_white_spectraldata.npz')

spectral_data = f['spectral_data']

metadata, acquisition_metadata, spectrometer_parameters, dmd_parameters = \
    read_metadata('./data/zoom_x1_white/zoom_x1_white_metadata.json')

recon = reconstruction_hadamard(acquisition_metadata.patterns, 'walsh', H, spectral_data)

#%% Analysis

top_right = np.mean(recon[8:24,40:56,:], axis=(0,1))
bot_left = np.mean(recon[40:56,8:24,:], axis=(0,1))
mid = np.mean(recon[24:40,24:40,:], axis=(0,1))

plt.figure()
plt.plot(acquisition_metadata.wavelengths, top_right/top_right.max(), label='top right')
plt.plot(acquisition_metadata.wavelengths, bot_left/bot_left.max(), label='bottom left')
plt.plot(acquisition_metadata.wavelengths, mid/mid.max(), label='center')
plt.xlabel('Wavelengths (nm)')
plt.ylabel('Normalized Intensity')
plt.legend()