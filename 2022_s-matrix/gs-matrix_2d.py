# -*- coding: utf-8 -*-
"""
Created on Tue May 24 08:44:31 2022

@author: ducros
"""
#%% 2D inverse S-matrix from spyrit
saveTag = False
n = 31

#- compute
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import spyrit.misc.walsh_hadamard as wh
from skimage import data, transform



def add_colorbar(mappable):
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    import matplotlib.pyplot as plt
    last_axes = plt.gca()
    ax = mappable.axes
    fig = ax.figure
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = fig.colorbar(mappable, cax=cax)
    plt.sca(last_axes)
    return cbar

#%% 2D Walsh-ordered from spyrit
import numpy as np
import spyrit.misc.walsh_hadamard as wh
import matplotlib.pyplot as plt

img_size = 4
H = wh.walsh2_matrix(img_size)
print(H)

f, axs = plt.subplots(4, 4, figsize=(8,8),  dpi= 100) 
for ii in range(4):
    for jj in range(4):
        ind = 4*ii + jj
        axs[ii,jj].imshow(np.reshape(H[ind,:],(img_size, img_size)), cmap='gray');
        #axs[ii,jj].set_title(f'{ind}');
        axs[ii,jj].get_xaxis().set_visible(False)
        axs[ii,jj].get_yaxis().set_visible(False)
        axs[ii,jj].set_aspect('equal')
        #axs[ii,jj].set_xticklabels([])
        #axs[ii,jj].set_yticklabels([])
f.subplots_adjust(wspace=0, hspace=0.05)

#%% Walsh S-matrix in 1D
saveTag = True
n = 15

#- compute
import matplotlib.pyplot as plt
import spyrit.misc.walsh_hadamard as wh

S = wh.walsh_S_matrix(n)
print(H)

#- plot    
f, a = plt.subplots(1,1, figsize=(8,8),  dpi= 100) 

a.imshow(S, cmap='gray');
a.get_xaxis().set_visible(False)
a.get_yaxis().set_visible(False)
a.set_aspect('equal')
f.subplots_adjust(wspace=0, hspace=0.05)

if saveTag:
    plt.savefig(f"Walsh_S_matrix_{n}.png", bbox_inches=0)

plt.title(f"Walsh S-matrix of order {n}")

#%% Walsh S-matrix in 2D
saveTag = True
n = 8

import numpy as np
import spyrit.misc.walsh_hadamard as wh
import matplotlib.pyplot as plt


S = wh.walsh2_S_matrix(n)

print(S)

f, axs = plt.subplots(n, n, figsize=(8,8),  dpi= 100) 
ind = 0
for ii in range(n):
    for jj in range(n):
        if ii==0 and jj==0:
            axs[0,0].axis('off')
            continue
        X = wh.walsh2_S_fold(S[ind,:])
        
        print(ii,jj,ind)
        print(X)
        
        axs[ii,jj].imshow(X, cmap='gray')
        #axs[ii,jj].set_title(f'{ind}');
        axs[ii,jj].get_xaxis().set_visible(False)
        axs[ii,jj].get_yaxis().set_visible(False)
        axs[ii,jj].set_aspect('equal')
        #axs[ii,jj].set_xticklabels([])
        #axs[ii,jj].set_yticklabels([])
        ind += 1
        
f.subplots_adjust(wspace=0, hspace=0.05)

if saveTag:
    f.savefig(f"Walsh2_S_matrix_{n}x{n}.png", bbox_inches=0)