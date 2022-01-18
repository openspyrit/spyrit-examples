import numpy as np
import spyrit.misc.walsh_hadamard as wh
from matplotlib import pyplot as plt
from scipy.interpolate import CubicSpline, interp1d
from scipy.optimize import curve_fit

from spas import read_metadata, reconstruction_hadamard
from spas import ReconstructionParameters, setup_reconstruction, spectral_binning, plot_color

H = wh.walsh2_matrix(64)

#%%

def mean_spectrum(acquisition_metadata):
    
    ind_opt = acquisition_metadata.patterns[0::2]
    ind_opt = np.array(ind_opt)/2
    ind_opt = ind_opt.astype('int')
    M = spectral_data
    M_breve = M[0::2,:] + M[1::2,:]
    spectrum = np.mean(M_breve, axis=0)
    
    return spectrum

#%%

f = np.load('./data/2021-10-22_spectral_resolution_HgAr_Lamp/HgAr_Lamp_center/HgAr_Lamp_center_spectraldata.npz')

spectral_data = f['spectral_data']

metadata, acquisition_metadata, spectrometer_parameters, dmd_parameters = \
    read_metadata('./data/2021-10-22_spectral_resolution_HgAr_Lamp/HgAr_Lamp_center/HgAr_Lamp_center_metadata.json')

recon_center = reconstruction_hadamard(acquisition_metadata.patterns, 'walsh', H, spectral_data)

spectrum_center = mean_spectrum(acquisition_metadata)

#%%

plt.imshow(np.sum(recon_center, axis=2))

#%%

F_bin, wavelengths_bin, bin_width = spectral_binning(recon_center.T, acquisition_metadata.wavelengths, 530, 560, 8)
plot_color(F_bin, wavelengths_bin, fontsize=16)

#%%

plt.figure()
plt.plot(acquisition_metadata.wavelengths, recon_center[30,36,:],label='1')
plt.xlim([542, 550])

#%%

from scipy.interpolate import UnivariateSpline

ind = np.where((acquisition_metadata.wavelengths > 542) & (acquisition_metadata.wavelengths < 550))

spline = UnivariateSpline(acquisition_metadata.wavelengths[ind], recon_center[30,36,ind], s=0)
r1, r2 = spline.roots()
plt.plot(acquisition_metadata.wavelengths, recon_center[30,36,:],label='1')
plt.xlim([542, 550])
plt.axvspan(r1, r2, facecolor='g', alpha=0.5)

