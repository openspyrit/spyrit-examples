#%% 
import numpy as np
from pathlib import Path
from tools import read_acquisitionData, binArray


#%% load raw data that has been downloaded from PILOT using download_data.py

data_folder = Path("data/opticalTuningCat3")
#Path("data/lisaCat") #Path("data/USAF2") #Path("data/USAF") # Path("data/opticalTuningCat3") # Path("data/opticalTuningCat2")

# spectral data
spectral_data = np.load(data_folder / f"spectral_data.npz", allow_pickle=True)["spectral_data"]

# metadata
meta_path = data_folder / "metadata.json"
meta = read_acquisitionData(meta_path)
acquisition_parameters = meta['acquisition_params']


wavelengths = acquisition_parameters.wavelengths # list of wavelengths measured

w = acquisition_parameters.pattern_dimension_x   # width of pattern
h = acquisition_parameters.pattern_dimension_y   # height of pattern
P = acquisition_parameters.pattern_amount        # number of (split) patterns applied

order = acquisition_parameters.patterns # order of acquisition patterns
ind = np.array(order) # corresponding acquisition rows


print(f"\n\nSpectral data shape: {spectral_data.shape}")
print(f"Spectral data dtype: {spectral_data.dtype}")
print(f"Wavelength range: {wavelengths.min():.2f} - {wavelengths.max():.2f} nm "
      f"({len(wavelengths)} points)")
print(f"Pattern dimensions: {h} x {w}")
print(f"Number of patterns applied: {P}")


#%% format raw data - will need to edit depending on what data was downloaded
spectral_data = spectral_data.squeeze() #np.moveaxis(spectral_data, 0,1)


# for opticalTuningCat, crop from 165 to 128:
spectral_data = spectral_data[20:148, :,:]

N,L,P = spectral_data.shape
K = P//2 

m_reordered = np.zeros((N,L,P)) # create space first
m_reordered[:,:,ind]  = spectral_data # place rows of m_unsplit following order of ind :)

# if already in asceneding order, should be zeros:
m_reordered - spectral_data

np.save(data_folder / 'formatted_spectral_data.npy', m_reordered)

# %%
# need to bin N dimension of m - or crop, depending on data
m_binned = m_reordered # binArray(m_reordered, 0, 3, 3, np.sum)
np.save(data_folder/'m_binned.npy', m_binned)

# %%
import matplotlib.pyplot as plt
plt.imshow(spectral_data[:,:,0])
plt.title("First Measure")
plt.colorbar()
# %%


