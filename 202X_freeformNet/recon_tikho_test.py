import torch
%matplotlib qt6 
# The above line enables interactive plotting in a separate window

# %% Parameters
# B is batch size, C is number of channels, H is height and width of the image,
# P is number of pixels in the freeform region, M is number of measurements, 
# N is number of pixels in the image.

B, C, H, P, M = 85, 10, 8, 9, 17
N = H**2

#%% Forward operator
from spyrit.core.meas import FreeformLinear

mask   = torch.randint(H,(2,P))
A = torch.randn(M, P)
meas  = FreeformLinear(A, meas_shape=(H,H), index_mask=mask)

# (random) images
x = torch.rand(B, C, H, H)

# measurements
y = meas(x)

#%% Tikhonov reconstruction
from spyrit.core.inverse import Tikhonov
sigma = torch.rand(P, P)    # covariance of the freeform image  
gamma = torch.rand(M, M)    # covariance of the noise

recon = Tikhonov(meas, sigma)
x_hat = recon(y, gamma)

#%% plot
import matplotlib.pyplot as plt

x_plot = torch.zeros_like(x)
x_plot[:,:,mask[0], mask[1]] = x[:,:,mask[0], mask[1]]

plt.subplot(1, 2, 1)
plt.imshow(x_plot[0, 0], cmap='gray')
plt.title('Ground-truth')
plt.colorbar(location='bottom')

plt.subplot(1, 2, 2)
plt.imshow(x_hat[0, 0], cmap='gray')
plt.title('Reconstructed')
plt.colorbar(location='bottom')

plt.show()  # This will open the plot in a separate window
