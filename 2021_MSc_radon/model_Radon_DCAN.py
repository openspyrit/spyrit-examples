#from __future__ import print_function, division
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
from PIL import Image, ImageOps
from collections import OrderedDict
import cv2
from scipy.stats import rankdata
#from ..misc.disp import *
from itertools import cycle;
import scipy.linalg as lin


#######################################################################
# 1.1 Determine the important Radon Coefficients
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# Helps determining the statistical best 
# Radon patterns for a given image size
#

def matrix2conv(Matrix):
    """
        Returns Convolution filter where each kernel corresponds to a line of the reshaped
        Matrix
    """
    M = Matrix.shape[0];
    N = Matrix.shape[1];
    img_size =int(round(np.sqrt(N)));

    P = nn.Conv2d(1,M,kernel_size=img_size, stride=1, padding=0);
    P.bias.data=torch.zeros(M);
    for i in range(M):
        pattern = Matrix[i,:].view(img_size, img_size);
        P.weight.data[i,0,:,:] = pattern;
    P.bias.requires_grad = False;
    P.weight.requires_grad=False;
    return P

def compute_radon(img,A):
    Q = img.shape[0];
    D = int(A.shape[0] / 181);
    alpha = np.zeros((Q*Q));
    for i in range(0, Q):
        alpha[i * Q:(i + 1) * Q] = img[i, :];
    m = np.dot(A, alpha);

    R = np.zeros((D, 181))
    for angle in range(0, 181):
        R[:, angle] = m[D * angle:D * (angle + 1)];
    return R

def Stat_radon(device, dataloader, root, A, Q, D):
    """ 
        Computes Mean radon Image over the whole dataset +
        Covariance Matrix Amongst the coefficients
    """
    with torch.no_grad():
        if device == "cuda:0":
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
        else:
            torch.set_default_tensor_type('torch.FloatTensor')

        inputs, classes = next(iter(dataloader))
        #inputs = inputs.cpu().detach().numpy();
        (batch_size, channels, nx, ny) = inputs.shape
        inputs = inputs.view([batch_size, A.shape[1], 1])

        if device == "cuda:0":
            inputs = inputs.type(torch.cuda.FloatTensor)
            A = A.type(torch.cuda.FloatTensor)
        else:
            inputs = inputs.type(torch.FloatTensor)
            A = A.type(torch.FloatTensor)

        tot_num = dataloader.dataset.data.shape[0]
        img_radon = torch.zeros([batch_size, D * 181, 1])
        Mean_radon = torch.zeros([D * 181, 1])
        for inputs,labels in dataloader:
            inputs = inputs.view([inputs.shape[0], A.shape[1], 1])

            if device == "cuda:0":
                inputs = inputs.type(torch.cuda.FloatTensor)
            else:
                inputs = inputs.type(torch.FloatTensor)


            img_radon = torch.matmul(A, inputs)
            Mean_radon += torch.sum(img_radon, 0)

            mat_test = img_radon[25,:,:]
            mat_test = mat_test.view(181, D)
            mat_test = mat_test.t()
            mat_test = mat_test.numpy()
            plt.matshow(mat_test)
            plt.colorbar()
            plt.show()

        Mean_radon = Mean_radon/tot_num
    
        torch.save(Mean_radon, root + 'Mean_Q{}D{}'.format(Q, D) + '.pt')
        Mean_radon_view = Mean_radon.view(181, D)
        Mean_radon_view = Mean_radon_view.t()
        Mean_radon_view = Mean_radon_view.numpy()

        plt.matshow(Mean_radon_view)
        plt.colorbar()
        plt.gray()
        plt.savefig(root + 'Mean_Q{}D{}'.format(Q, D) + '.png')
        #plt.show()
    
        Cov_radon = torch.zeros([D * 181, D * 181])
        for inputs,labels in dataloader:
            inputs = inputs.view([inputs.shape[0], A.shape[1], 1])

            if device == "cuda:0":
                inputs = inputs.type(torch.cuda.FloatTensor)
            else:
                inputs = inputs.type(torch.FloatTensor)

            img_radon = torch.matmul(A, inputs)
            Norm_Variable = img_radon - Mean_radon.repeat(inputs.shape[0], 1, 1)
            for index in range(inputs.shape[0]):
                Cov_radon += Norm_Variable[index, :, :] * torch.transpose(Norm_Variable[index, :, :], 0, 1)
            #Cov_radon += torch.sum(torch.mul(Norm_Variable , torch.transpose(Norm_Variable,1,2)), 0)
        Cov_radon = Cov_radon/(tot_num-1)
    
        torch.save(Cov_radon, root + 'Cov_Q{}D{}'.format(Q, D) + '.pt')
        Cov_array = Cov_radon.numpy()
        plt.matshow(Cov_array)
        plt.colorbar()
        plt.savefig(root + 'Cov_Q{}D{}'.format(Q, D) + '.png')
        return Mean_radon, Cov_radon


def Cov2Var(Cov,D,nbAngles):
    Var = torch.diag(Cov)
    Var = Var.view(nbAngles, D)
    return Var.t()

def radonSpecifyAngles(A,angles):
    nbAngles = angles.shape[0];
    D = (int)(A.shape[0]/181);
    QQ = A.shape[1];

    Areduced = torch.zeros([ D*nbAngles , QQ ]);
    #Areduced = torch.zeros([ D*181 , QQ ]);

    for theta in range(0, nbAngles):
        Areduced[theta * D : theta * D + D - 1, :] = A[(int)(angles[theta] * D) : (int)(angles[theta] * D + D - 1), :]
        #Areduced[(int)(angles[theta] * D):(int)(angles[theta] * D + D - 1), :] = A[(int)(angles[theta] * D):(int)(
            #angles[theta] * D + D - 1), :]
    return Areduced;

def generateAngles(nbAngles):
    angles = np.zeros([nbAngles]);
    for i in range(0, nbAngles):
        angles[i] = (int)(180 * i / nbAngles);
    return angles;

def vector2matrix(vector, d):
    vector_affichage = vector.view(d[0], d[1])
    vector_affichage = vector_affichage.t()
    matrix = vector_affichage.numpy()
    return matrix

#######################################################################
# 1.2 Important functions for statistical completion
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# Helps determining the statistical best
# Radon patterns for a given image size
#

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

    sigmaMix = np.dot(sigma21, lin.pinv(sigma1))
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


###############################################################################
# 2. NEW Convolutional Neural Network
###############################################################################
#==============================================================================
# A. NO NOISE
#==============================================================================    
class compNet(nn.Module):
    def __init__(self, Q, D, nbAngles, A = None, pinvA = None, Mean = None, Cov = None, variant=0):
        super(compNet, self).__init__()
        
        self.Q = Q;
        self.D = D;
        self.A = A;
        self.nbAngles = nbAngles;

        if type(Mean)==type(None):
            Mean = torch.zeros([D * 181, 1])

        if type(Cov)==type(None):
            Cov = torch.zeros([D * 181, D * 181])

        if type(A)==type(None):
            A = torch.zeros([D * nbAngles, Q*Q])

        if type(pinvA)==type(None):
            pinvA = torch.zeros([Q*Q, D * nbAngles])

        self.pinvA = pinvA; # (A'A + lambda*I)^-1*A' le top  = apprendre lambda ! (passer par `matmul`)
        # sinon prendre lambda = 1e-1*trace(A'A)


        ''''
        #-- Denoising parameters 
        Sigma = np.dot(Perm,np.dot(Cov,np.transpose(Perm)));
        diag_index = np.diag_indices(n**2);
        Sigma = Sigma[diag_index];
        Sigma = 1/4*Sigma[:M]; #(A = A donc Cov = Cov)!
        #Sigma = Sigma[:M];
        Sigma = torch.Tensor(Sigma);
        self.sigma = Sigma.view(1,1,M);
        

        P1 = np.zeros((n**2,1));
        P1[0] = n**2;
        mean = n*np.reshape(Mean,(self.n**2,1))+P1;
        mu = (1/2)*np.dot(Perm, mean);
        #mu = np.dot(Perm, np.reshape(Mean, (n**2,1)))
        mu1 = torch.Tensor(mu[:M]);
        self.mu_1 = mu1.view(1,1,M);
        '''

        #-- Measurement preprocessing
        self.Patt = matrix2conv(A);
        self.Patt.bias.requires_grad = False;
        self.Patt.weight.requires_grad = False;

        #-- Pseudo-inverse to determine levels of noise.
        self.Pinv = nn.Linear(D * nbAngles, Q*Q, False)
        self.Pinv.weight.data= pinvA;
        self.Pinv.weight.data = self.Pinv.weight.data.float();
        self.Pinv.weight.requires_grad=False;


        #-- Measurement to image domain
        if variant==0:
            ''''
            #--- Statistical Matrix completion (no mean)
            print("Measurement to image domain: statistical completion (no mean)")
            
            self.fc1 = nn.Linear(M,n**2, False)
            
            W, b, mu1 = stat_completion_matrices(Perm, A, Cov, Mean, M)
            W = (1/n**2)*W; 

            self.fc1.weight.data=torch.from_numpy(W);
            self.fc1.weight.data=self.fc1.weight.data.float();
            self.fc1.weight.requires_grad=False;
            '''
        if variant==1:
            '''
            #--- Statistical Matrix completion
            print("Measurement to image domain: statistical completion")
            
            self.fc1 = nn.Linear(M,n**2)
            
            W, b, mu1 = stat_completion_matrices(Perm, A, Cov, Mean, M)
            W = (1/n**2)*W; 
            b = (1/n**2)*b;
            b = b - np.dot(W,mu1);
            self.fc1.bias.data=torch.from_numpy(b[:,0]);
            self.fc1.bias.data=self.fc1.bias.data.float();
            self.fc1.bias.requires_grad = False;
            self.fc1.weight.data=torch.from_numpy(W);
            self.fc1.weight.data=self.fc1.weight.data.float();
            self.fc1.weight.requires_grad=False;
            '''
        elif variant==2:
            #--- Pseudo-inverse
            print("Measurement to image domain: pseudo inverse")

            self.fc1 = self.Pinv;
       
        elif variant==3:
            #--- FC is learnt
            print("Measurement to image domain: free")
            
            self.fc1 = nn.Linear(D * nbAngles, Q*Q)
            
        #-- Image correction
        self.recon = nn.Sequential(OrderedDict([
          ('conv1', nn.Conv2d(1,64,kernel_size=9, stride=1, padding=4)),
          ('relu1', nn.ReLU()),
          ('conv2', nn.Conv2d(64,32,kernel_size=1, stride=1, padding=0)),
          ('relu2', nn.ReLU()),
          ('conv3', nn.Conv2d(32,1,kernel_size=5, stride=1, padding=2))
        ]));

    def forward(self, x):
        b,c,h,w = x.shape;
        x = self.forward_acquire(x, b, c, h, w);
        x = self.forward_reconstruct(x, b, c, h, w);
        return x
    #--------------------------------------------------------------------------
    # Forward functions (with grad)
    #--------------------------------------------------------------------------
    def forward_acquire(self, x, b, c, h, w):
        #--Acquisition
        x = self.Patt(x);
        return x
    
    def forward_maptoimage(self, x, b, c, h, w):
        '''
        #- Pre-processing (use batch norm to avoid division by N0 ?)
        x = self.T(x);
        x = 2*x-torch.reshape(self.Patt(torch.ones(b*c,1, h,w).to(x.device)),(b*c,1,self.M));
        '''
        #--Projection to the image domain
        x = self.fc1(x[:,:,0,0]);

        x = x.view(b,1,h,w)

        return x
    
    def forward_postprocess(self, x, b, c, h, w):
        x = self.recon(x)
        return x
    
    def forward_reconstruct(self, x, b, c, h, w):
        x = self.forward_maptoimage(x, b, c, h, w)
        x = self.forward_postprocess(x, b, c, h, w)
        return x
     
    
    #--------------------------------------------------------------------------
    # Evaluation functions (no grad)
    #--------------------------------------------------------------------------
    def acquire(self, x, b, c, h, w):
        with torch.no_grad():
            b,c,h,w = x.shape
            x = self.forward_acquire(x, b, c, h, w)
        return x
    
    def evaluate_fcl(self, x):
        with torch.no_grad():
           b,c,h,w = x.shape
           x = self.forward_acquire(x, b, c, h, w)
           x = self.forward_maptoimage(x, b, c, h, w)
        return x
     
    def evaluate_Pinv(self, x):
        with torch.no_grad():
           b,c,h,w = x.shape
           x = self.forward_Pinv(x, b, c, h, w)
        return x
    
    def evaluate(self, x):
        with torch.no_grad():
           x = self.forward(x)
        return x
    
    def reconstruct(self, x, b, c, h, w):
        with torch.no_grad():
            b,c,h,w = x.shape
            x = self.forward_reconstruct(x, b, c, h, w)
        return x
   
#==============================================================================    
# B. NOISY MEASUREMENTS (NOISE LEVEL IS VARYING)
#==============================================================================
class noiCompNet(compNet):
    def __init__(self, n, M, Mean, Cov, variant, N0, sig = 0.1, H=None):
        super().__init__(n, M, Mean, Cov, variant, H)
        self.N0 = N0;
        self.sig = sig;
        self.max = nn.MaxPool2d(kernel_size = n);
        print("Varying N0 = {:g} +/- {:g}".format(N0,sig*N0))
        
    def forward_acquire(self, x, b, c, h, w):
        #--Scale input image
        x = (self.N0*(1+self.sig*torch.randn_like(x)))*(x+1)/2;
        #--Acquisition
        x = x.view(b*c, 1, h, w);
        x = self.P(x);
        x = F.relu(x);     # x[:,:,1] = -1/N0 ????
        x = x.view(b*c,1, 2*self.M); # x[:,:,1] < 0??? 
        #--Measurement noise (Gaussian approximation of Poisson)
        x = x + torch.sqrt(x)*torch.randn_like(x);  
        return x
    
    def forward_maptoimage(self, x, b, c, h, w):
        #-- Pre-processing (use batch norm to avoid division by N0 ?)
        x = self.T(x);
        x = 2/self.N0*x-torch.reshape(self.Patt(torch.ones(b*c,1, h,w).to(x.device)),(b*c,1,self.M)); 
        #--Projection to the image domain
        x = self.fc1(x);
        x = x.view(b*c,1,h,w) 
        return x
         

    def forward_Pinv(self, x, b, c, h, w):
        #-- Pre-processing (use batch norm to avoid division by N0 ?)
        x = self.T(x);
        x = 2/self.N0*x-torch.reshape(self.Patt(torch.ones(b*c,1, h,w).to(x.device)),(b*c,1,self.M)); 
        #--Projection to the image domain
        x = self.Pinv(x);
        x = x.view(b*c,1,h,w)
        return x
 
    def forward_N0_Pinv(self, x, b, c, h, w):
        #-- Pre-processing (use batch norm to avoid division by N0 ?)
        x = self.T(x);
        #--Projection to the image domain
        x = self.Pinv(x);
        x = x.view(b*c,1,h,w)
        N0_est = self.max(x);
        N0_est = N0_est.view(b*c,1,1,1);
        N0_est = N0_est.repeat(1,1,h,w);
        x = torch.div(x,N0_est);
        x=2*x-1; 
        return x
     
    def forward_N0_maptoimage(self, x, b, c, h, w):
        #-- Pre-processing(Recombining positive and negatve values+normalisation) 
        x = x[:,:,self.even_index] - x[:,:,self.uneven_index];
        
        #-- Pre-processing(Estimating No and normalizing by No) 
        x_est = self.Pinv(x);
        x_est = x_est.view(b*c,1,h,w);
        N0_est = self.max(x_est);
        N0_est = N0_est.view(b*c,1,1);
        N0_est = N0_est.repeat(1,1,self.M);
        x = torch.div(x,N0_est);
        x = 2*x-torch.reshape(self.Patt(torch.ones(b*c,1, h,w).to(x.device)),(b*c,1,self.M)); 

        #--Projection to the image domain
        x = self.fc1(x);
        x = x.view(b*c,1,h,w)
        return x
    
    def forward_N0_reconstruct(self, x, b, c, h, w):
        x = self.forward_N0_maptoimage(x, b, c, h, w)
        x = self.forward_postprocess(x, b, c, h, w)
        return x
 
    def forward_stat_comp(self, x, b, c, h, w):
        #-- Pre-processing(Recombining positive and negatve values+normalisation) 
        x = x[:,:,self.even_index] - x[:,:,self.uneven_index];
        x = x/self.N0;
        x = 2*x-torch.reshape(self.Patt(torch.ones(b*c,1, h,w).to(x.device)),(b*c,1,self.M)); 

        #--Projection to the image domain
        x = self.fc1(x);
        x = x.view(b*c,1,h,w) 
        return x
     
 

#==============================================================================    
# B. NOISY MEASUREMENTS (NOISE LEVEL IS VARYING) + denoising architecture
#==============================================================================
class DenoiCompNet(noiCompNet):
    def __init__(self, n, M, Mean, Cov, variant, N0, sig = 0.1, H=None, mean_denoi=False):
        super().__init__(n, M, Mean, Cov, variant, N0, sig, H)
        print("Denoised Measurements")
   
    def forward_maptoimage(self, x, b, c, h, w):
        #-- Pre-processing(Recombining positive and negatve values+normalisation) 
        var = x[:,:,self.even_index] + x[:,:,self.uneven_index];
        x = x[:,:,self.even_index] - x[:,:,self.uneven_index];
        x = x/self.N0;
        x = 2*x-torch.reshape(self.Patt(torch.ones(b*c,1, h,w).to(x.device)),(b*c,1,self.M)); 
        
        #--Denoising 
        sigma = self.sigma.repeat(b*c,1,1).to(x.device);
        x = torch.mul(torch.div(sigma, sigma+var/(self.N0)**2), x);

        #--Projection to the image domain
        x = self.fc1(x);
        x = x.view(b*c,1,h,w) 
        return x
    
    def forward_maptoimage_2(self, x, b, c, h, w):
        #-- Pre-processing(Recombining positive and negatve values+normalisation) 
        var = x[:,:,self.even_index] + x[:,:,self.uneven_index];
        x = x[:,:,self.even_index] - x[:,:,self.uneven_index];
        x = x/self.N0;
        x = 2*x-torch.reshape(self.Patt(torch.ones(b*c,1, h,w).to(x.device)),(b*c,1,self.M)); 
        
        #--Denoising 
        sigma = self.sigma.repeat(b*c,1,1).to(x.device);
        mu_1 = self.mu_1.repeat(b*c,1,1).to(x.device);
        x = mu_1 + torch.mul(torch.div(sigma, sigma+var/(self.N0)**2), x-mu_1);

        #--Projection to the image domain
        x = self.fc1(x);
        x = x.view(b*c,1,h,w) 
        return x
     
    def forward_denoised_Pinv(self, x, b, c, h, w):
        #-- Pre-processing(Recombining positive and negatve values+normalisation) 
        var = x[:,:,self.even_index] + x[:,:,self.uneven_index];
        x = x[:,:,self.even_index] - x[:,:,self.uneven_index];
        x = x/self.N0;
        x = 2*x-torch.reshape(self.Patt(torch.ones(b*c,1, h,w).to(x.device)),(b*c,1,self.M)); 
        
        #--Denoising 
        sigma = self.sigma.repeat(b*c,1,1).to(x.device);
        x = torch.mul(torch.div(sigma, sigma+4*var/(self.N0)**2), x);

        #--Projection to the image domain
        x = self.Pinv(x);
        x = x.view(b*c,1,h,w) 
        return x
   
    def forward_reconstruct(self, x, b, c, h, w):
        x = self.forward_maptoimage(x, b, c, h, w);
        x = self.forward_postprocess(x, b, c, h, w)
        return x

    def forward_NO_maptoimage(self, x, b, c, h, w):
        #-- Pre-processing(Recombining positive and negatve values+normalisation) 
        var = x[:,:,self.even_index] + x[:,:,self.uneven_index];
        x = x[:,:,self.even_index] - x[:,:,self.uneven_index];
        
        #-- Pre-processing(Estimating No and normalizing by No) 
        x_est = self.Pinv(x);
        x_est = x_est.view(b*c,1,h,w);
        N0_est = self.max(x_est);
        N0_est = N0_est.view(b*c,1,1);
        N0_est = N0_est.repeat(1,1,self.M);
        x = torch.div(x,N0_est);
         
        #--Denoising 
        sigma = self.sigma.repeat(b*c,1,1).to(x.device);
        x = torch.mul(torch.div(sigma, sigma+torch.div(var,N0_est**2)), x);
        x = 2*x-torch.reshape(self.Patt(torch.ones(b*c,1, h,w).to(x.device)),(b*c,1,self.M)); 

        #--Projection to the image domain
        x = self.fc1(x);
        x = x.view(b*c,1,h,w) 
        return x;

    def forward_N0_maptoimage_expe(self, x, b, c, h, w, C, s, g):
        #-- Pre-processing(Recombining positive and negatve values+normalisation) 
        var = g**2*(x[:,:,self.even_index] + x[:,:,self.uneven_index]) - 2*C*g +2*s**2;
        x = x[:,:,self.even_index] - x[:,:,self.uneven_index];
        
        #-- Pre-processing(Estimating No and normalizing by No) 
        x_est = self.Pinv(x);
        x_est = x_est.view(b*c,1,h,w);
        N0_est = self.max(x_est);
        N0_est = N0_est.view(b*c,1,1);
        N0_est = N0_est.repeat(1,1,self.M);
        x = torch.div(x,N0_est);
         
        #--Denoising 
        sigma = self.sigma.repeat(b*c,1,1).to(x.device);
        x = torch.mul(torch.div(sigma, sigma+torch.div(var,N0_est**2)), x);
        x = 2*x-torch.reshape(self.Patt(torch.ones(b*c,1, h,w).to(x.device)),(b*c,1,self.M)); 
 
        #--Projection to the image domain
        x = self.fc1(x);
        x = x.view(b*c,1,h,w) 
        return x;

    
    def forward_N0_reconstruct_expe(self, x, b, c, h, w,C,s,g):
        x = self.forward_N0_maptoimage_expe(x, b, c, h, w,C,s,g)
        x = self.forward_postprocess(x, b, c, h, w)
        return x
 
    def forward_N0_maptoimage_expe_bis(self, x, b, c, h, w, C, s, g, N0):
        #-- Pre-processing(Recombining positive and negatve values+normalisation) 
        var = g**2*(x[:,:,self.even_index] + x[:,:,self.uneven_index]) - 2*C*g +2*s**2;
        var = x[:,:,self.even_index] + x[:,:,self.uneven_index];
        x = x[:,:,self.even_index] - x[:,:,self.uneven_index];
        
        #-- Pre-processing(Estimating No and normalizing by No) 
        x_est = self.Pinv(x);
        x_est = x_est.view(b*c,1,h,w);
        N0_est = self.max(x_est);
        N0_est = N0_est.view(b*c,1,1);
        N0_est = N0_est.repeat(1,1,self.M);
        sigma = self.sigma.repeat(b*c,1,1).to(x.device);
        print(N0_est)
        x = x/N0;
#        x = torch.div(x,N0_est);


        x = torch.mul(torch.div(sigma, sigma+torch.div(var,N0_est**2)), x);
        x = 2*x-torch.reshape(self.Patt(torch.ones(b*c,1, h,w).to(x.device)),(b*c,1,self.M)); 
 
#        var = x[:,:,self.even_index] + x[:,:,self.uneven_index];
#        x = x[:,:,self.even_index] - x[:,:,self.uneven_index];
#        x = torch.div(x,N0_est);
#        x = x/N0;
#         
#        #--Denoising 
#        sigma = self.sigma.repeat(b*c,1,1).to(x.device);
#        x = torch.mul(torch.div(sigma, sigma+var/(N0)**2), x);
#        x = 2*x-torch.reshape(self.Patt(torch.ones(b*c,1, h,w).to(x.device)),(b*c,1,self.M)); 
# 
        #--Projection to the image domain
        x = self.fc1(x);
        x = x.view(b*c,1,h,w) 
        return x;

    
    def forward_N0_reconstruct_expe_bis(self, x, b, c, h, w,C,s,g, N0):
        x = self.forward_N0_maptoimage_expe_bis(x, b, c, h, w,C,s,g, N0)
        x = self.forward_postprocess(x, b, c, h, w)
        return x
 

########################################################################
# 2. Define a custom Loss function
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# Creating custom loss function
# ---------------------------
# Just to make sure that all functions work the same way...   
# ie that they take the same number of arguments

class Weight_Decay_Loss(nn.Module):
    
    def __init__(self, loss):
        super(Weight_Decay_Loss,self).__init__()
        self.loss = loss;

    def forward(self,x,y, net):
        mse=self.loss(x,y);
        return mse


