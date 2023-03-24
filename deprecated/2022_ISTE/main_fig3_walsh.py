# -*- coding: utf-8 -*-
"""
Created on Tue Jun 22 18:53:39 2021

@author: ducros
"""

#%% 1D Hadamard from scipy
import matplotlib.pyplot as plt
from scipy.linalg import hadamard
H = hadamard(8)

for i in range(8):
    print(H[i,:])
    
f, a = plt.subplots(1,1, figsize=(8,8),  dpi= 100) 

a.imshow(H, cmap='gray');
a.get_xaxis().set_visible(False)
a.get_yaxis().set_visible(False)
a.set_aspect('equal')
f.subplots_adjust(wspace=0, hspace=0.05)
plt.savefig("Hadamard_natural_8.png", bbox_inches=0)


#%% 1D Walsh-ordered from spyrit
import spyrit.misc.walsh_hadamard as wh
H = wh.walsh_matrix(8)

for i in range(8):
    print(H[i,:])
    

f, a = plt.subplots(1,1, figsize=(8,8),  dpi= 100) 

a.imshow(H, cmap='gray');
a.get_xaxis().set_visible(False)
a.get_yaxis().set_visible(False)
a.set_aspect('equal')
f.subplots_adjust(wspace=0, hspace=0.05)
plt.savefig("Hadamard_Walsh_8.png", bbox_inches=0)


#%% 2D Walsh-ordered from spyrit
import numpy as np
img_size = 4
H = wh.walsh2_matrix(img_size)
print(H[0,0])

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
f.savefig("Hadamard_Walsh_4x4.png", bbox_inches=0)