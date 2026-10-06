import json  
from datetime import datetime
from enum import IntEnum
from dataclasses import dataclass, InitVar, field
from typing import Optional, Union, List, Tuple, Optional
from pathlib import Path
import os
from dataclasses_json import dataclass_json
import numpy as np
import ctypes as ct
import pickle

import re
from types import SimpleNamespace

import json
import ast
import numpy as np
from pathlib import Path
import imageio.v3 as iio
import matplotlib.pyplot as plt
from PIL import Image
import torch


def binArray(data, axis, binstep, binsize, func=np.nanmean):

    data = np.array(data)
    dims = np.array(data.shape)
    argdims = np.arange(data.ndim)
    argdims[0], argdims[axis]= argdims[axis], argdims[0]
    data = data.transpose(argdims)
    data = [func(np.take(data,np.arange(int(i*binstep),int(i*binstep+binsize)),0),0) for i in np.arange(dims[axis]//binstep)]
    data = np.array(data).transpose(argdims)
    return data

# def hadamard(M, w):

    
#     pattern_dir = Path(f"C:/Users/ceidigh/Documents/Obair/data/Patterns/Walsh_{M}x{M}")
#     fname = pattern_dir / f"Walsh_{M}x{M}_0.png"
#     png = np.array(Image.open(fname).convert("L"))
#     px,py = png.shape

#     A_png = np.zeros((2*M, px, px)) # want square - patterns have border in y

#     for i in range(2*M):
#             fname = pattern_dir / f"Walsh_{M}x{M}_{i}.png"
#             png = np.array(Image.open(fname).convert("L"))
#             png = (png > 0).astype(float)
            
#             A_png[i] = png[:, ((py-px)//2):(px + ((py-px)//2))]

#     px,py = A_png[0].shape
#     pat_pos = np.zeros((M,px)) # half measures are positive (1s)
#     pat_neg = np.zeros((M,px)) # other half need to be minused (represent the -1s that the DMD can't do)

#     for i in range(0,M*2,2):
#         pat_pos[i//2,:] =  A_png[i, 400, :] # just select a row - not measured data so all are identical and already in binary
#         pat_neg[i//2,:] =  A_png[i+1, 400, :]# just select a row - not measured data so all are identical and already in binary 

#         if((A_png[i, 400, :] - pat_pos[i//2,:]).sum() != 0 ):
#             print("Check failed at i = ",i, ":", A_png[i, 400, :] - pat_pos[i//2,:])

#         if((A_png[i+1, 400, :] - pat_neg[i//2,:]).sum() != 0 ):
#             print("Check failed at i = ",i, ":", A_png[i, 400, :] - pat_pos[i//2,:])


#     # make square
#     pat_pos = bin_cols(pat_pos,  (px // (w))) 
#     pat_neg = bin_cols(pat_neg,  (px // (w)))
#     H = pat_pos-pat_neg

#     #normalise
#     H = H//(px // (w))

#     return pat_pos, pat_neg, H


def read_acquisitionData(file_path: str):
    """Reads acquisition data from a JSON file

    """
    with open(file_path, 'r') as file:
        data = json.load(file)

    result = {}

    for entry in data:
        desc = entry.get('class_description')

        if desc == 'Acquisition parameters':
            acq = AcquisitionParameters.from_dict(entry)
            acq.undo_readable_pattern_order()
            result['acquisition_params'] = acq

        # else:
        #     print(f"Warning: unrecognized class_description '{desc}'")

    return result

# taken from spyrit git
@dataclass_json
@dataclass
class AcquisitionParameters:
    """Class containing acquisition specifications and timing results.

    This class is adapted to be reconstructed from a JSON file.

    Attributes:
        pattern_compression (float):
            Percentage of total available patterns to be present in an
            acquisition sequence.
        pattern_dimension_x (int):
            Length of reconstructed image that defines pattern length.
        pattern_dimension_y (int):
            Width of reconstructed image that defines pattern width.
        zoom (int):
            numerical zoom of the patterns
        xw_offset (int):
            offset of the pattern in the DMD for zoom > 1 in the width (x) direction
        yh_offset (int):
            offset of the pattern in the DMD for zoom > 1 in the heihgt (y) direction   
        mask_index (Union[np.ndarray, str], optional):
            Array of `int` type corresponding to the index of the mask vector where
            the value is egal to 1
        x_mask_coord (Union[np.ndarray, str], optional):
            coordinates of the mask in the x direction x[0] and x[1] are the first
            and last points respectively
        y_mask_coord (Union[np.ndarray, str], optional):
            coordinates of the mask in the y direction y[0] and y[1] are the first
            and last points respectively    
        pattern_amount (int, optional):
            Quantity of patterns sent to DMD for an acquisition. This value is
            calculated by an external function. Default in None.
        acquired_spectra (int, optional):
            Amount of spectra actually read from the spectrometer. This value is
            calculated by an external function. Default in None.
        mean_callback_acquisition_time_ms (float, optional):
            Mean time between 2 callback executions during an acquisition. This 
            value is calculated by an external function. Default in None.
        total_callback_acquisition_time_s (float, optional):
            Total time of callback executions during an acquisition. This value
            is calculated by an external function. Default in None.
        mean_spectrometer_acquisition_time_ms (float, optional):
            Mean time between 2 spectrometer measurements during an acquisition
            based on its own internal clock. This value is calculated by an
            external function. Default in None.
        total_spectrometer_acquisition_time_s (float, optional):
            Total time of spectrometer measurements during an acquisition
            based on its own internal clock. This value is calculated by an
            external function. Default in None.
        saturation_detected (bool, optional):
            Boolean incating if saturation was detected during acquisition.
            Default is None.
        patterns (Union[List[int],str], optional) = None
            List `int` or `str` containing all patterns sent to the DMD for an
            acquisition sequence. This value is set by an external function and
            its type can be modified by multiple functions during object
            creation, manipulation, when dumping to a JSON file or
            when reconstructing an AcquisitionParameters object from a JSON
            file. It is intended to be of type List[int] most of the execution
            List[int]time. Default is None.
        wavelengths (Union[np.ndarray, str], optional):
            Array of `float` type corresponding to the wavelengths associated
            with spectrometer's start and stop pixels.
        timestamps (Union[List[float], str], optional):
            List of `float` type elapsed time between each measurement
            made by the spectrometer based on its internal clock. Units in 
            milliseconds. Default is None.
        measurement_time (Union[List[float], str], optional):
            List of `float` type elapsed times between each callback. Units in
            milliseconds. Default is None.
        class_description (str):
            Class description used to improve redability when dumped to JSON
            file. Default is 'Acquisition parameters'.
    """

    pattern_compression: float
    pattern_dimension_x: int
    pattern_dimension_y: int
    zoom: Optional[int] = field(default=None) 
    xw_offset: Optional[int] = field(default=None) 
    yh_offset: Optional[int] = field(default=None) 
    mask_index: Optional[Union[np.ndarray, str]] = field(default=None, 
                                                        repr=False)
    x_mask_coord: Optional[Union[np.ndarray, str]] = field(default=None, 
                                                        repr=False)
    y_mask_coord: Optional[Union[np.ndarray, str]] = field(default=None, 
                                                        repr=False)
    
    pattern_amount: Optional[int] = None
    acquired_spectra: Optional[int] = None

    mean_callback_acquisition_time_ms: Optional[float] = None
    total_callback_acquisition_time_s: Optional[float] = None
    mean_spectrometer_acquisition_time_ms: Optional[float] = None
    total_spectrometer_acquisition_time_s: Optional[float] = None

    saturation_detected: Optional[bool] = None

    patterns: Optional[Union[List[int], str]] = field(default=None, repr=False)
    patterns_wp: Optional[Union[List[int], str]] = field(default=None, repr=False)
    wavelengths: Optional[Union[np.ndarray, str]] = field(default=None, 
                                                        repr=False)
    timestamps: Optional[Union[List[float], str]] = field(default=None, 
                                                        repr=False)
    measurement_time: Optional[Union[List[float], str]] = field(default=None,
                                                            repr=False)

    class_description: str = 'Acquisition parameters'


    def undo_readable_pattern_order(self) -> None:
        """Changes the patterns attribute from `str` to `List` of `int`."""

        def to_float(str_arr):
            arr = []
            for s in str_arr:
                try:
                    num = float(s)
                    arr.append(num)
                except ValueError:
                    pass
            return arr

        def parse_pattern_list(s):
            """Handles plain ints ('[0, 1, 2]') and numpy-repr ints
            ('[np.uint16(0), np.uint16(1)]')."""
            if s is None or s == 'None':
                return None

            s = s.strip()
            if s.startswith('[') and s.endswith(']'):
                s = s[1:-1]
            s = s.strip()
            if not s:
                return []

            items = s.split(', ')
            result = []
            for item in items:
                item = item.strip()
                match = re.search(r'\((-?\d+)\)', item)
                if match:
                    result.append(int(match.group(1)))
                else:
                    result.append(int(item))
            return result

        self.patterns = parse_pattern_list(self.patterns)
        self.patterns_wp = parse_pattern_list(self.patterns_wp)
        if self.wavelengths:
            self.wavelengths = (
                self.wavelengths.strip('[').strip(']').split(', '))
            self.wavelengths = to_float(self.wavelengths)
            self.wavelengths = np.asarray(self.wavelengths)
        else:
            print('wavelenghts not present in metadata.'
            ' Reading data in legacy mode.')

        if self.timestamps:
            self.timestamps = self.timestamps.strip('[').strip(']').split(', ')
            self.timestamps = to_float(self.timestamps)
        else:
            print('timestamps not present in metadata.'
            ' Reading data in legacy mode.')

        if self.measurement_time:
            self.measurement_time = (
                self.measurement_time.strip('[').strip(']').split(', '))
            self.measurement_time = to_float(self.measurement_time)
        else:
            print('measurement_time not present in metadata.'
            ' Reading data in legacy mode.')

        if self.mask_index:
            self.mask_index = (
                self.mask_index.strip('[').strip(']').split(', '))
            self.mask_index = to_float(self.mask_index)
            self.mask_index = np.asarray(self.mask_index)
        else:
            print('mask_index not present in metadata.'
            ' Reading data in legacy mode.')
        
        if self.x_mask_coord:
            self.x_mask_coord = (
                self.x_mask_coord.strip('[').strip(']').split(', '))
            self.x_mask_coord = to_float(self.x_mask_coord)
            self.x_mask_coord = np.asarray(self.x_mask_coord)
        else:
            print('x_mask_coord not present in metadata.'
            ' Reading data in legacy mode.')
        
        if self.y_mask_coord:
            self.y_mask_coord = (
                self.y_mask_coord.strip('[').strip(']').split(', '))
            self.y_mask_coord = to_float(self.y_mask_coord)
            self.y_mask_coord = np.asarray(self.y_mask_coord)
        else:
            print('y_mask_coord not present in metadata.'
            ' Reading data in legacy mode.')
        
    @staticmethod
    def readable_pattern_order(acquisition_params_dict: dict) -> dict:
        """Turns list of patterns into a string.

        Turns the list of pattern attributes from an AcquisitionParameters 
        object (turned into a dictionary) into a string that will improve
        readability once all metadata is dumped into a JSON file.
        This function must be called before dumping.

        Args:
            acquisition_params_dict (dict): Dictionary obtained from converting 
            an AcquisitionParameters object.

        Returns:
            [dict]: Modified dictionary with acquisition parameters metadata.
        """

        def _hard_coded_conversion(data):
            s = '['
            for value in data:
                s += f'{value:.4f}, '
            s = s[:-2]
            s += ']'

            return s

        readable_dict = acquisition_params_dict
        readable_dict['patterns'] = str(readable_dict['patterns'])
        readable_dict['patterns_wp'] = str(readable_dict['patterns_wp'])
        
        readable_dict['wavelengths'] = _hard_coded_conversion(
            readable_dict['wavelengths'])
    
        readable_dict['timestamps'] = _hard_coded_conversion(
            readable_dict['timestamps'])

        readable_dict['measurement_time'] = _hard_coded_conversion(
            readable_dict['measurement_time'])
        
        readable_dict['mask_index'] = _hard_coded_conversion(
            readable_dict['mask_index'])
        
        readable_dict['x_mask_coord'] = _hard_coded_conversion(
            readable_dict['x_mask_coord'])
        
        readable_dict['y_mask_coord'] = _hard_coded_conversion(
            readable_dict['y_mask_coord'])

        return readable_dict


    def update_timings(self, timestamps: np.ndarray, 
                       measurement_time: np.ndarray):
        """Updates acquisition timings.

        Args:
            timestamps (ndarray): 
                Array of `float` type elapsed time between each measurement made
                by the spectrometer based on its internal clock. Units in 
                milliseconds.
            measurement_time (ndarray):
                Array of `float` type elapsed times between each callback. Units
                in milliseconds.
        """
        self.mean_callback_acquisition_time_ms = np.mean(measurement_time)
        self.total_callback_acquisition_time_s = np.sum(measurement_time) / 1000
        self.mean_spectrometer_acquisition_time_ms = np.mean(
            timestamps, dtype=float)
        self.total_spectrometer_acquisition_time_s = np.sum(timestamps) / 1000

        self.timestamps = timestamps
        self.measurement_time = measurement_time


        import json
import ast
import numpy as np
from pathlib import Path
import imageio.v3 as iio
import matplotlib.pyplot as plt

def binArray(data, axis, binstep, binsize, func=np.nanmean):

    data = np.array(data)
    dims = np.array(data.shape)
    argdims = np.arange(data.ndim)
    argdims[0], argdims[axis]= argdims[axis], argdims[0]
    data = data.transpose(argdims)
    data = [func(np.take(data,np.arange(int(i*binstep),int(i*binstep+binsize)),0),0) for i in np.arange(dims[axis]//binstep)]
    data = np.array(data).transpose(argdims)
    return data

def load_experiment(exp_dir: str | Path, gr: int, lc: str, file_prefix: str = "spectral") -> dict:
    exp_dir = Path(exp_dir)
    meta_path = exp_dir / "metadata.json"
    raw_dir = exp_dir / "raw_data"
    overview_dir = exp_dir / "overview"

    with open(meta_path, "r") as f:
        metadata = json.load(f)

    acq = metadata[-1]
    dmd = metadata[0]

    wavelengths = np.array(ast.literal_eval(acq["wavelengths"]), dtype=float)
    Lc = np.array(ast.literal_eval(acq["Lc"]), dtype=float)[0][0]

    patterns = acq["patterns"]
    p_x = acq["pattern_dimension_x"]
    p_y = acq["pattern_dimension_y"]
    M = int(dmd["patterns"])

    # build filenames in guaranteed numeric order
    files = [
        raw_dir / f"{file_prefix}_NR_0_Gr_{gr}_Lc_{lc}nm_NA_0_NS_{k}.npz"
        for k in range(M)
    ]

    missing = [f for f in files if not f.exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing {len(missing)} files, first missing: {missing[0]}"
        )

    first = np.load(files[0], allow_pickle=True)["arr_0"].astype(np.float32)
    N, L = first.shape

    if L != len(wavelengths):
        print(f"Warning: file has L={L} channels, metadata has {len(wavelengths)} wavelengths")

    y = np.empty((M, N, L), dtype=np.float32)
    y[0] = first

    for k, f in enumerate(files[1:], start=1):
        arr = np.load(f, allow_pickle=True)["arr_0"].astype(np.float32)
        if arr.shape != (N, L):
            raise ValueError(f"Shape mismatch in {f.name}: got {arr.shape}, expected {(N, L)}")
        y[k] = arr

    
    raw_data = np.moveaxis(y, 0, -1) # N,L,P
    

    bin_fact =  N / (M//2)

    if bin_fact == 1:
        spectral_data_all = raw_data.astype(float, copy=True)
    else:
        spectral_data_all = binArray(raw_data, 0, bin_fact, bin_fact)
    

  #  spectral_data_all = (spectral_data_all - spectral_data_all.min()) / (spectral_data_all.max() - spectral_data_all.min())
    f = raw_dir / f"spatial_NR_0_Gr_{gr}_Lc_{lc}nm_NA_0_NS_1.npz"
    data = np.load(f)

    spatial_data = data[data.files[0]]


    # bin_img   = iio.imread(overview_dir / "spectral_BIN_IMAGE_had_reco.png")
    # gray_img  = iio.imread(overview_dir / "spectral_GRAY_IMAGE_had_reco.png")
    # rgb_img   = iio.imread(overview_dir / "spectral_RGB_IMAGE_had_reco.png")
    # slice_img = iio.imread(overview_dir / "spectral_SLICE_IMAGE_had_reco.png")
    # spectra   = iio.imread(overview_dir / "spectral_SPECTRA_PLOT_had_reco.png")

    return {
        "dir": exp_dir,
        "metadata": metadata,
        "acq": acq,
        "wavelengths": wavelengths,
        "Lc": Lc,
        "patterns": patterns,
        "pattern_dimension_x": p_x,
        "pattern_dimension_y": p_y,
        "M": M,
        "N": N,
        "L": L,
        "y": y,
        "raw_data": raw_data,
        "spectral_data_all": spectral_data_all,
        # "bin_img": bin_img,
        # "gray_img": gray_img,
        # "rgb_img": rgb_img,
        # "slice_img": slice_img,
        # "spectra": spectra,
        "spatial_data": spatial_data,
    }

