import torch
import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as sio
import h5py
from PIL import Image, ImageOps

#Ce programme permet de montrer un exemple de complétion statistique de sinogramme

#On définit d'abord un nombre d'angles consécutif d'acquisition commençant par 0°
nbAngles = 100

#Importer les matrices de mesure et reconstruction
radon_matrix_path = '../../models/radon/Q64_D64.mat'
H = h5py.File(radon_matrix_path)
A = H.get("A")
A = np.array(A)
A = torch.from_numpy(A)
A = A.t()
A = A.type(torch.FloatTensor)

'''
radon_matrix_path = '../../models/radon/pinv/pinv_Q64_D64.mat'
H = h5py.File(radon_matrix_path)
pinvA = H.get("A_pinv")
pinvA = np.array(pinvA)
pinvA = torch.from_numpy(pinvA)
pinvA = pinvA.t()
pinvA = pinvA.type(torch.FloatTensor)
'''

regu = 1e2;
pinvA = np.dot(sio.inv(np.dot(np.transpose(A),A)+regu*np.eye(64*64)),np.transpose(A))
pinvA = torch.from_numpy(pinvA)
pinvA = pinvA.type(torch.FloatTensor)

#Importer les matrices de moyenne et de covariance
Mean_radon = torch.load('../../models/radon/Mean_Q64D64.pt', map_location='cpu')
Cov_radon = torch.load('../../models/radon/Cov_Q64D64.pt', map_location='cpu')

Mean_radon_view = Mean_radon.view(181,64)
Mean_radon_view = Mean_radon_view.t()
Mean_radon_array = Mean_radon_view.numpy()
#plt.matshow(Mean_radon_array)
#plt.colorbar()
#plt.show()

#Importer une image
im = Image.open("../../models/radon/test2.png")
im = ImageOps.grayscale(im)
im_array = np.asarray(im)
im_array = im_array.astype(np.float32)
im_array = 2*(im_array)/255 - np.ones([64,64])
plt.matshow(im_array)
plt.colorbar()
plt.gray()
f = torch.from_numpy(im_array)
f = f.view(1,64*64);
f = f.t()
f = f.type(torch.FloatTensor)


#Effectuer une mesure
m = torch.mv(A,f[:,0])

f_o = torch.mv(pinvA,m)
f_o = f_o.view(64,64)
f_oarray = f_o.numpy()
plt.matshow(f_oarray)
plt.colorbar()
plt.gray()


m_view = m.view(181,64)
m_view = m_view.t()
m_array = m_view.numpy()
plt.matshow(m_array)
plt.colorbar()
plt.gray()

#Tronquer le sinogramme après un certain nombre d'angles d'acquisition
mreduced = m[:nbAngles*64]

#Séparation des moyennes des pixels. mu1 étant l'acquisition et mu2 les pixel à boucher
mu1 = Mean_radon[:nbAngles*64,0]
mu2 = Mean_radon[nbAngles*64:,0]

#Séparation des covariances des pixels. Simga1 covariance de l'acquisition, Sigma21 la covariance des pixels acquis et non acquis
Sigma1 = Cov_radon[:64*nbAngles,:64*nbAngles]
Sigma21 = Cov_radon[64*nbAngles:,:64*nbAngles]


#Complétion statistique
diff = mreduced-mu1

sigma1_np = Sigma1.numpy()

pinvs = sio.pinv(sigma1_np)

pinvSigma1 = torch.from_numpy(pinvs)

sMix = torch.matmul(Sigma21,pinvSigma1)

y2 = mu2 + torch.mv(torch.matmul(Sigma21,pinvSigma1),diff)
#y2 = 0

B = torch.mv(sMix,mreduced)
B_np = B.numpy()
W = mu2 - torch.mv(sMix,mu1)
W_np = W.numpy()

mcomplete = torch.zeros(64*181)
mcomplete[:64*nbAngles] = mreduced
mcomplete[64*nbAngles:] = y2

mcomplete_view = mcomplete.view(181,64)
mcomplete_view = mcomplete_view.t()
mcomplete_array = mcomplete_view.numpy()
plt.matshow(mcomplete_array)
plt.colorbar()
plt.gray()

f_reconstructed = torch.mv(pinvA,mcomplete)


f_reconstructed_view = f_reconstructed.view(64,64)
f_reconstructed_array = f_reconstructed_view.numpy()
plt.matshow(f_reconstructed_array)
plt.colorbar()
plt.gray()


plt.show()

