import torch
import numpy as np
import matplotlib.pyplot as plt
import h5py
import scipy.linalg as sio
from PIL import Image, ImageOps

def generateActivationAngles(nbAngles):
    listOfAngles = np.floor(np.linspace(0,180,nbAngles,endpoint=True))
    listOfAnglesInt = listOfAngles.astype(int)
    angles = np.zeros(181)
    c = 0
    for i in range(0,181):
        if listOfAnglesInt[c] == i:
            angles[i] = 1
            c = c+1

    return angles;

def separateMu(mu, angles, nbAngles):
    mu = np.reshape(mu,[181, 64])
    mu = np.transpose(mu)

    mu1 = np.zeros([64,nbAngles], dtype="float32")
    mu2 = np.zeros([64, 181-nbAngles], dtype="float32")
    cmu1 = 0
    cmu2 = 0
    for i in range(0,181):
        if angles[i]==1:
            mu1[:,cmu1] = mu[:,i]
            cmu1 = cmu1 + 1
        else:
            mu2[:,cmu2] = mu[:,i]
            cmu2 = cmu2 + 1

    #mu1 = np.resize(mu1, [nbAngles*64,1])
    #mu2 = np.resize(mu2, [(181-nbAngles) * 64, 1])

    mu1 = np.transpose(mu1)
    mu1 = np.resize(mu1, [nbAngles*64,1])
    mu2 = np.transpose(mu2)
    mu2 = np.resize(mu2, [(181-nbAngles) * 64, 1])


    return mu1, mu2


def separateSigma(sigma, angles, nbAngles):
    sigma1 = np.zeros([64 * nbAngles, 64 * nbAngles], dtype="float32")
    sigma21 = np.zeros([64*(181-nbAngles), 64 * nbAngles], dtype="float32")
    csigma1_i = 0
    csigma1_j = 0
    csigma21_i = 0
    csigma21_j = 0
    for i in range(0,181):
        if (angles[i] == 1):
            csigma1_j = 0
            for j in range(0, 181):
                if (angles[j] == 1):
                    sigma1[csigma1_i*64:csigma1_i*64+64,csigma1_j*64:csigma1_j*64+64] = sigma[i*64:i*64+64,j*64:j*64+64]
                    csigma1_j = csigma1_j + 1
            csigma1_i = csigma1_i + 1

        if (angles[i] == 0):
            csigma21_j = 0
            for j in range(0, 181):
                if (angles[j] == 1):
                    sigma21[csigma21_i*64:csigma21_i*64 + 64, csigma21_j*64:csigma21_j*64 + 64] = sigma[i * 64:i * 64 + 64, j * 64:j * 64 + 64]
                    csigma21_j = csigma21_j + 1
            csigma21_i = csigma21_i + 1

    return sigma1, sigma21

def computeUnknownData(mu1,mu2,sigma1,sigma21,angles,nbAngles):
    w = np.zeros([181*64, nbAngles*64], dtype="float32")
    b = np.zeros([181*64,1], dtype="float32")

    sigmaMix = np.dot(sigma21, sio.pinv(sigma1))
    #sigmaMix = np.dot(sigma21,np.transpose(sigma1))
    bMix = mu2 - np.dot(sigmaMix, mu1)
    #bMix = mu2

    eyeMix = np.eye(nbAngles*64)
    cm1 = 0
    cm2 = 0

    for i in range(0,181):
        if angles[i]==1:
            w[i*64:i*64+64,cm1*64:cm1*64+64] = eyeMix[cm1*64:cm1*64+64,cm1*64:cm1*64+64]
            cm1 = cm1+1
        if angles[i] == 0:
            w[i*64:i*64+64,:] = sigmaMix[cm2*64:cm2*64+64,:]
            b[i*64:i*64+64,:] = bMix[cm2*64:cm2*64+64,:]
            cm2 = cm2+1

    return w, b[:,0]

def radonSpecifyAngles(A,angles):
    nbAngles = angles.shape[0]
    D = (int)(A.shape[0]/181)
    QQ = A.shape[1]

    Areduced = torch.zeros([ D*nbAngles, QQ ])
    for theta in range(0, nbAngles):
        Areduced[theta * D : theta * D + D - 1, :] = A[(int)(angles[theta] * D) : (int)(angles[theta] * D + D - 1), :]
    return Areduced

def generateAngles(nbAngles):
    angles = np.zeros([nbAngles])
    for i in range(0, nbAngles):
        angles[i] = (int)(180 * i / nbAngles)
    return angles


#Nombre d'angles d'acquisition
nbAngles = 60

#Importer les matrices de mesure et reconstruction
radon_matrix_path = '../../models/radon/Q64_D64.mat'
H = h5py.File(radon_matrix_path)
A = H.get("A")
A = np.array(A)
A = torch.from_numpy(A)
A = A.t()
A = A.type(torch.FloatTensor)
Areduced = radonSpecifyAngles(A, generateAngles(nbAngles))

'''
radon_matrix_path = '../../models/radon/pinv/pinv_Q64_D64.mat'
H = h5py.File(radon_matrix_path)
pinvA = H.get("A_pinv")
pinvA = np.array(pinvA)
pinvA = torch.from_numpy(pinvA)
pinvA = pinvA.t()
pinvA = pinvA.type(torch.FloatTensor)
'''
regu = 1e1;
pinvA = np.dot(sio.inv(np.dot(np.transpose(A), A) + regu * np.eye(64 * 64)), np.transpose(A))
pinvA = torch.from_numpy(pinvA)
pinvA = pinvA.type(torch.FloatTensor)

#Importer les matrices de moyenne et de covariance
Mean_radon = torch.load('../../models/radon/Mean_Q64D64.pt', map_location='cpu')
Cov_radon = torch.load('../../models/radon/Cov_Q64D64.pt', map_location='cpu')
mu = Mean_radon.numpy()
sigma = Cov_radon.numpy()
#mu = np.linspace(0,nbAngles,nbAngles)
#sigma = np.linspace(0,181*64*181*64,181*64*181*64)
#sigma = np.resize(sigma,[181*64, 181*64])
#mu = np.zeros(181*64)
#sigma = np.zeros([181*64,181*64])

#Importer une image
im = Image.open("../../models/radon/test2.png")
im = ImageOps.grayscale(im)
im_array = np.asarray(im)
im_array = im_array.astype(np.float32)
im_array = 2*(im_array)/255 - np.ones([64,64])
f = torch.from_numpy(im_array)
f = f.view(1,64*64);
f = f.t()
f = f.type(torch.FloatTensor)

#Effectuer une mesure
m = torch.mv(Areduced,f[:,0])
m_perfect = torch.mv(A,f[:,0])

#afficher le sinogrames plein
m_perfect_affichage = m_perfect.view(181,64)
m_perfect_affichage = m_perfect_affichage.t()
m_perfect_array = m_perfect_affichage.numpy()


#Faire des trous
angles = generateActivationAngles(nbAngles)
m_perfect_holes = torch.zeros(nbAngles*64)
m_unperfect = torch.zeros(181*64)
count = 0
for i in range(0,181):
    if (angles[i] == 1):
        m_perfect_holes[count*64:count*64+64] = m_perfect[i*64:i*64+64]
        m_unperfect[i * 64:i * 64 + 64] = m_perfect[i*64:i*64+64]
        count = count + 1

#Boucher les trous
muTuple = separateMu(mu,angles,nbAngles)
sigmaTuple = separateSigma(sigma,angles,nbAngles)
networkTuple = computeUnknownData(muTuple[0], muTuple[1], sigmaTuple[0], sigmaTuple[1],angles,nbAngles)


W = torch.from_numpy(networkTuple[0])
W = W.type(torch.FloatTensor)
B = torch.from_numpy(networkTuple[1])
B = B.type(torch.FloatTensor)

#m_filled = torch.mv(W,m_perfect_holes) + B
m_filled = torch.mv(W,m_perfect_holes)
m_filled_bias = torch.mv(W,m_perfect_holes) + B
m_bias = m_unperfect + B


#afficher le sinogrames bouché
m_filled_affichage = m_filled.view(181,64)
m_filled_affichage = m_filled_affichage.t()
m_filled_array = m_filled_affichage.numpy()

#afficher le sinogrames bouché avec Biais
m_filled_bias_affichage = m_filled_bias.view(181,64)
m_filled_bias_affichage = m_filled_bias_affichage.t()
m_filled_bias_array = m_filled_bias_affichage.numpy()

#afficher le sinogrames Biais
m_bias_affichage = m_bias.view(181,64)
m_bias_affichage = m_bias_affichage.t()
m_bias_array = m_bias_affichage.numpy()

#afficher le sinogrames troué
m_unperfect_affichage = m_unperfect.view(181,64)
m_unperfect_affichage = m_unperfect_affichage.t()
m_unperfect_array = m_unperfect_affichage.numpy()


#reconstruire l'image
f_reconstructed = torch.mv(pinvA,m_filled)
f_reconstructed_view = f_reconstructed.view(64,64)
f_reconstructed_array = f_reconstructed_view.numpy()

f_poor = np.dot(sio.pinv(Areduced.numpy()),m.numpy())
f_poor = np.resize(f_poor,[64,64])

f_perfect = torch.mv(pinvA,m_perfect)
f_perfect_view = f_perfect.view(64,64)
f_perfect_array = f_perfect_view.numpy()

#Afficher les résultats

plt.matshow(im_array, cmap='gray')
plt.colorbar()

plt.matshow(m_perfect_array, cmap='gray')
plt.colorbar()

#plt.matshow(m_filled_array, cmap='gray')
#plt.colorbar()

plt.matshow(m_filled_bias_array, cmap='gray')
plt.colorbar()

#plt.matshow(m_bias_array, cmap='gray')
#plt.colorbar()

#plt.matshow(m_unperfect_array, cmap='gray')
#plt.colorbar()

plt.matshow(f_reconstructed_array, cmap='gray')
plt.colorbar()

#plt.matshow(f_poor, cmap='gray')
#plt.colorbar()

#plt.matshow(f_perfect_array, cmap='gray')
#plt.colorbar()

plt.show()


