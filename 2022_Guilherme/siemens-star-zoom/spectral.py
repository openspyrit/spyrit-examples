import numpy as np
import spyrit.misc.walsh_hadamard as wh
from matplotlib import pyplot as plt
from scipy.signal import find_peaks

from spas import read_metadata, reconstruction_hadamard
from spas import ReconstructionParameters, setup_reconstruction

H = wh.walsh2_matrix(64)/64

f = np.load('./data/2021-10-22_spectral_resolution_HgAr_Lamp/HgAr_Lamp_center/HgAr_Lamp_center_spectraldata.npz')

spectral_data = f['spectral_data']

metadata, acquisition_metadata, spectrometer_parameters, dmd_parameters = \
    read_metadata('./data/2021-10-22_spectral_resolution_HgAr_Lamp/HgAr_Lamp_center/HgAr_Lamp_center_metadata.json')

recon = reconstruction_hadamard(acquisition_metadata.patterns, 'walsh', H, spectral_data)

recon_sum = np.sum(recon,axis=2) # Summing all the wavelength components

ind_opt = acquisition_metadata.patterns[0::2]
ind_opt = np.array(ind_opt)/2
ind_opt = ind_opt.astype('int')
M = spectral_data
M_breve = M[0::2,:] - M[1::2,:]
spectrum = np.mean(M_breve, axis=0)