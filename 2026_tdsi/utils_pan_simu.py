import torch
import numpy as np
import scipy
import pandas as pd

from spyrit.misc.disp import torch2numpy
from spyrit.misc.dual_arm import recalibrate


def simulate_hs_data(img: torch.tensor, n: int, homography: torch.tensor, R: torch.tensor, spectrum: torch.tensor):
    HS_ref = img.moveaxis(1, -1) @ spectrum
    HS_ref = HS_ref.moveaxis(-1, 1)

    P = HS_ref.moveaxis(1, -1) @ R
    P = P.moveaxis(-1, 1)

    homography_inv = torch.linalg.inv(homography)

    HS = recalibrate(HS_ref, (n, n), homography_inv)

    return HS_ref, HS, P


def gaussian(x, mu=0., sigma=1.):
    normalization = 1 #torch.sqrt(2 * torch.tensor(np.pi)) * sigma
    gauss = torch.exp(- ((x - mu) / sigma) ** 2 / 2) / normalization
    return  gauss


def simulate_spectrum_RGB(wavelengths, CIE=True):
    """
    Simulate spectral responses for RGB discretized with n_wav points
    :param wavelengths: array of shape (n_wav,) representing the wavelengths in which we discretize the spectrum
    :return: array of shape (3, n_wav)
    """

    n_wav = wavelengths.shape[0]

    if CIE:
        cie_data = pd.read_csv('../data/pan_data/spec_rep/CIE_xyz_1931_2deg.csv')  
        cie_wav = cie_data.iloc[:, 0]
        x_bar = cie_data.iloc[:, 1] 
        y_bar = cie_data.iloc[:, 2]  
        z_bar = cie_data.iloc[:, 3] 

        spectrum = np.zeros((3, n_wav))
        x_bar_interp = scipy.interpolate.interp1d(cie_wav, x_bar, kind='linear', bounds_error=False, fill_value=0)
        spectrum[0] = x_bar_interp(torch2numpy(wavelengths))
        y_bar_interp = scipy.interpolate.interp1d(cie_wav, y_bar, kind='linear', bounds_error=False, fill_value=0)
        spectrum[1] = y_bar_interp(torch2numpy(wavelengths))
        z_bar_interp = scipy.interpolate.interp1d(cie_wav, z_bar, kind='linear', bounds_error=False, fill_value=0)
        spectrum[2] = z_bar_interp(torch2numpy(wavelengths))
        spectrum = torch.from_numpy(spectrum).to(dtype=wavelengths.dtype)

        spectrum = spectrum / spectrum.sum()

    else:
        mu_blue = 465
        mu_green = 532
        mu_red = 630

        spectrum = torch.zeros((3, n_wav), dtype=wavelengths.dtype)

        spectrum[0] = gaussian(wavelengths, mu_red, 20) / 3
        spectrum[1] = gaussian(wavelengths, mu_green, 25) / 3
        spectrum[2] = gaussian(wavelengths, mu_blue, 10) / 3

    return spectrum