# -*- coding: utf-8 -*-
"""
Created on Mon May 16 15:26:43 2022

@author: ducros
"""

#%% 1D Walsh-Hadamard matrix from spyrit
saveTag = False
n = 16

#- compute
import matplotlib.pyplot as plt
import spyrit.misc.walsh_hadamard as wh

H = wh.walsh_matrix(n)

print(H)

#- plot    
f, a = plt.subplots(1,1, figsize=(8,8),  dpi= 100) 

a.imshow(H, cmap='gray');
a.get_xaxis().set_visible(False)
a.get_yaxis().set_visible(False)
a.set_aspect('equal')
f.subplots_adjust(wspace=0, hspace=0.05)

if saveTag:
    plt.savefig(f"Walsh_{n}.png", bbox_inches=0)
    
plt.title(f"Hadamard matrix of order {n}")
    
#%% 1D Walsh-Hadamard G-matrix from spyrit
saveTag = True
n = 15

#- compute
import matplotlib.pyplot as plt
import spyrit.misc.walsh_hadamard as wh

H = wh.walsh_G_matrix(n)
print(H)

#- plot    
f, a = plt.subplots(1,1, figsize=(8,8),  dpi= 100) 

a.imshow(H, cmap='gray');
a.get_xaxis().set_visible(False)
a.get_yaxis().set_visible(False)
a.set_aspect('equal')
f.subplots_adjust(wspace=0, hspace=0.05)

if saveTag:
    plt.savefig(f"Walsh_G_matrix_{n}.png", bbox_inches=0)

plt.title(f"Walsh G-matrix of order {n}")
    
#%% 1D S-matrix from spyrit
saveTag = True
n = 15

#- compute
import matplotlib.pyplot as plt
import spyrit.misc.walsh_hadamard as wh

H = wh.walsh_S_matrix(n)
print(H)

#- plot    
f, a = plt.subplots(1,1, figsize=(8,8),  dpi= 100) 

a.imshow(H, cmap='gray');
a.get_xaxis().set_visible(False)
a.get_yaxis().set_visible(False)
a.set_aspect('equal')
f.subplots_adjust(wspace=0, hspace=0.05)

if saveTag:
    plt.savefig(f"Walsh_S_matrix_{n}.png", bbox_inches=0)

plt.title(f"Walsh S-matrix of order {n}")

#%% 1D inverse S-matrix from spyrit
saveTag = False
n = 15

#- compute
import matplotlib.pyplot as plt
import spyrit.misc.walsh_hadamard as wh

H = wh.iwalsh_S_matrix(n)
print(H)

#- Check the inverse
print(H @ wh.walsh_S_matrix(n))
print(wh.walsh_S_matrix(n) @ H)

#- plot    
f, a = plt.subplots(1,1, figsize=(8,8),  dpi= 100) 

a.imshow(H, cmap='gray');
a.get_xaxis().set_visible(False)
a.get_yaxis().set_visible(False)
a.set_aspect('equal')
f.subplots_adjust(wspace=0, hspace=0.05)

if saveTag:
    plt.savefig(f"Walsh_inv_S_matrix_{n}.png", bbox_inches=0)

plt.title(f"Inverse Walsh S-matrix of order {n}")
