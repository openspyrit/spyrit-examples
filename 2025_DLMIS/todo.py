# -*- coding: utf-8 -*-

#from skimage import data
import matplotlib.pyplot as plt
import numpy as np
from skimage.transform import radon, iradon, resize
from pathlib import Path

#%%
class Radon:
    def __init__(self, angle, img_shape=(128,128)):
        self.angle = angle
        self.img_shape = img_shape
        sinog = radon(np.zeros(img_shape), circle=False, theta=angle)
        self.pixel = sinog.shape[0]
    
    def forward(self, f):
        m = radon(f, self.angle, circle=False)
        return m
    
    def adjoint(self, m):
        f = iradon(m, self.angle, circle=False, filter_name=None)
        return f
    
    def pinv(self, m):
        f = iradon(m, self.angle, circle=False, filter_name='ramp')
        return f
    
def ml_gaussian(A, m, n_ite=20):
    """
    Maximum likelihood under Gaussian noise.
    
    It iteratively solves the linear system :math:`m = Af` that has been corrupted by gaussian noise.
       
    Args:
        A (class): Forward model :math:`A`. The class computes :math:`Af` from :math:`f` using the `forward()` method and :math:`A^T m` from :math:`m` using the `adjoint()` method.
        
        m (ndarray): Measurement :math:`m` of shape (n_pixel, n_angle).
        
        n_ite (int, optional): Number of iterations. Defaults to 20.

    Returns:
        f (ndarray): Unknown image :math:`f` of shape (n_1, n_2).
    """
    # Init
    f = np.ones(A.img_shape)
    
    # Main loop
    for kk in range(n_ite):
        print(f'iteration: {kk}')
        AtAf =  A.adjoint(A.forward(f))
        AtAf[AtAf==0] = 1e-6
        f = f / AtAf * A.adjoint(m)
    return f

#%% Load image
image_file = Path('./data/test/1/653_150_T2-S.png')

imag = plt.imread(image_file)

# rescale
img_size = 64           # image assumed to be square
imag = (imag - imag.min())/(imag.max() - imag.min())
imag = resize(imag, (img_size, img_size))

# plot
plt.figure()
plt.imshow(imag, cmap='gray')
plt.xlabel(r'$x_1$ (in pixels)')
plt.ylabel(r'$x_2$ (in pixels)')
plt.colorbar()

#%% Compute radon transform using skimage
n_angle = 100
theta = np.linspace(0.0, 180.0, n_angle)
sinog = radon(imag, circle=False, theta=theta)

n_detec = sinog.shape[0]

# plot | todo: specify axis and units
plt.figure()
plt.imshow(sinog.T, cmap='gray')
plt.ylabel(r'$\theta$ (in degrees)')
plt.xlabel(r'$\rho$ (in pixels)')
plt.colorbar()


prct = 0.05 # noise percentage
filter_name = 'ramp' # E.g., 'ramp', 'shepp-logan', 'cosine', 'hamming', 'hann', None

#%% Add noise and reconstruct
sinog_noise = sinog + prct*sinog.max()*np.random.standard_normal(size=sinog.shape)
recon = iradon(sinog_noise, circle=False, theta=theta, filter_name=filter_name)

# plot | todo: specify axis and units
plt.figure()
plt.imshow(sinog_noise.T, cmap='gray')
plt.ylabel(r'$\theta$ (in degrees)')
plt.xlabel(r'$\rho$ (in pixels)')
plt.title('noisy sinogram')
plt.colorbar()

# plot | todo: specify axis and units
plt.figure()
plt.imshow(recon, cmap='gray')
plt.xlabel(r'$x_1$ (in pixels)')
plt.ylabel(r'$x_2$ (in pixels)')
plt.title('filtered back projection')
plt.colorbar()

# Reconstruction
n_ite = 40
A = Radon(theta, (img_size,img_size))
f_rec = ml_gaussian(A, sinog_noise, n_ite=n_ite)
f_rec  = f_rec.reshape((img_size,img_size))

# Display results
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 4.5), layout='compressed')
ax1.set_title(r"Ground Truth")
im = ax1.imshow(imag, cmap=plt.cm.Greys_r)
fig.colorbar(im)

ax2.set_title(f"ML ({n_ite} iterations)")
im = ax2.imshow(f_rec, cmap=plt.cm.Greys_r)
fig.colorbar(im)

ax3.set_title(f"Diff ({n_ite} iterations)")
im = ax3.imshow(imag-f_rec, cmap=plt.cm.Greys_r)
fig.colorbar(im)