# -*- coding: utf-8 -*-

import numpy as np
from spyrit.misc.statistics import Cov2Var, img2mask
from matplotlib import pyplot as plt
from spyrit.misc.disp import add_colorbar

#%%
M = [1024,2048,4096]

C = np.load('../../stat/ILSVRC2012_v10102019/Cov_8_128x128.npy')
Ord = Cov2Var(C)
mask_0 = img2mask(Ord, M[0])
mask_1 = img2mask(Ord, M[1])
mask_2 = img2mask(Ord, M[2])

fig , axs = plt.subplots(1,3)
#noaxis(axs)
#
im = axs[0].imshow(mask_0, cmap='gray')
axs[0].set_title(f"M = {M[0]}")
add_colorbar(im)
#
im = axs[1].imshow(mask_1, cmap='gray')
axs[1].set_title(f"M = {M[1]}")
add_colorbar(im)
#
im = axs[2].imshow(mask_2, cmap='gray')
axs[2].set_title(f"M = {M[2]}")
add_colorbar(im)