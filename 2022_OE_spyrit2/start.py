# -*- coding: utf-8 -*-

#%% Compute covariance matrix
from spyrit.misc.statistics import stat_walsh_stl10
#stat_walsh_stl10()

#%% Compare "experimental" and "simulated" covariances
import numpy as np
C_sim = np.load('./stats/Cov_64x64.npy')
C_exp = np.load('./data_online/Cov_64x64.npy')
C_exp2= np.load('./models_online/Cov_64x64.npy')
err = np.linalg.norm(C_sim - C_exp)/np.linalg.norm(C_exp)
print(f'err = {err}')

err = np.linalg.norm(C_exp - C_exp2)/np.linalg.norm(C_exp2)
print(f'err = {err}')


#%% Compare "experimental" and "simulated" covariances
i1 = 3426
i2 = 876
print(C_exp[i1, i2])
print(C_sim[i1, i2])

#%%