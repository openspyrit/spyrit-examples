#%% to debug needs to run
import collections
collections.Callable = collections.abc.Callable

# %%
import torch.nn as nn
import numpy as np
import torch
from spyrit.core.meas import LinearSplit
from spyrit.core.recon import PseudoInverse

class LinearSplitPN(LinearSplit):
# ==================================================================================
    r""" 
    Computes linear measurements from incoming images: :math:`y = Px`, 
    where :math:`P` is a linear operator (matrix) and :math:`x` is a 
    vectorized image.
    
    The matrix :math:`P` contains only positive values and is obtained by 
    splitting a measurement matrix :math:`H` such that 
    :math:`P = \begin{bmatrix}{H_{+}}\\{H_{-}}\end{bmatrix}`, where 
    :math:`H_{+} = \max(0,H)` and :math:`H_{-} = \max(0,-H)`.
         
    The class is constructed from the :math:`M` by :math:`N` matrices 
    :math:`H_+` and :math:`H_-`, where :math:`N` represents the number of 
    pixels in the image and :math:`M` the number of measurements.
    
    Args:
        - :math:`H_+` (np.ndarray): positive component of the measurement 
        matrix (linear operator) with shape :math:`(M, N)`.
        
        - :math:`H_-` (np.ndarray): negative  component of the measurement 
        matrix (linear operator) with shape :math:`(M, N)`.
        
    Example:
        >>> H_pos = np.array(np.random.random([400,1000]))
        >>> H_neg = np.array(np.random.random([400,1000]))
        >>> meas_op =  LinearSplit(H_pos, H_neg)
    """

    def __init__(self, H_pos: np.ndarray, H_neg: np.ndarray, 
                         pinv = None, reg: float = 1e-15): 
        
        H = H_pos - H_neg
        
        super().__init__(H, pinv, reg)
                
        even_index = range(0,2*self.M,2);
        odd_index = range(1,2*self.M,2);
        
        P = np.zeros((2*self.M,self.N));
        P[even_index,:] = H_pos
        P[odd_index,:] = H_neg
        
        self.P = nn.Linear(self.N, 2*self.M, False) 
        self.P.weight.data = torch.from_numpy(P)
        self.P.weight.data = self.P.weight.data.float()
        self.P.weight.requires_grad=False

class Pinv1dNet(nn.Module):
    r""" Pseudo inverse reconstruction across 1 dimension
    
    .. math:
        
        
    Args:
        :attr:`noise`: Acquisition operator (see :class:`~spyrit.core.noise`) 
        
        :attr:`prep`: Preprocessing operator (see :class:`~spyrit.core.prep`)
        
        :attr:`denoi` (optional): Image denoising operator 
        (see :class:`~spyrit.core.nnet`). 
        Default :class:`~spyrit.core.nnet.Identity`
    
    Input / Output:
        :attr:`input`: Ground-truth images with shape :math:`(B,C,H,W)` 
        
        :attr:`output`: Reconstructed images with shape :math:`(B,C,H,W)`
    
    Attributes:
        :attr:`Acq`: Acquisition operator initialized as :attr:`noise`
        
        :attr:`prep`: Preprocessing operator initialized as :attr:`prep`
        
        :attr:`pinv`: Analytical reconstruction operator initialized as 
        :class:`~spyrit.core.recon.PseudoInverse()`
        
        :attr:`Denoi`: Image denoising operator initialized as :attr:`denoi`

    
    Example:
        >>> B, C, H, M = 10, 1, 64, 64**2
        >>> Ord = np.ones((H,H))
        >>> meas = HadamSplit(M, H, Ord)
        >>> noise = NoNoise(meas)
        >>> prep = SplitPoisson(1.0, M, H*H)
        >>> recnet = PinvNet(noise, prep)
        >>> x = torch.FloatTensor(B,C,H,H).uniform_(-1, 1)
        >>> z = recnet(x)
        >>> print(z.shape)
        >>> print(torch.linalg.norm(x - z)/torch.linalg.norm(x))
        torch.Size([10, 1, 64, 64])
        tensor(5.8912e-06)
    """
    def __init__(self, noise, prep, denoi=nn.Identity()):
        super().__init__()
        self.acqu = noise 
        self.prep = prep
        self.pinv = PseudoInverse()
        self.denoi = denoi

    def forward(self, x):
        r""" Full pipeline of reconstrcution network
            
        Args:
            :attr:`x`: ground-truth images
        
        Shape:
            :attr:`x`: ground-truth images with shape :math:`(B,C,H,W)`
            
            :attr:`output`: reconstructed images with shape :math:`(B,C,H,W)`
        
        Example:
            >>> B, C, H, M = 10, 1, 64, 64**2
            >>> Ord = np.ones((H,H))
            >>> meas = HadamSplit(M, H, Ord)
            >>> noise = NoNoise(meas)
            >>> prep = SplitPoisson(1.0, M, H*H)
            >>> recnet = PinvNet(noise, prep)
            >>> x = torch.FloatTensor(B,C,H,H).uniform_(-1, 1)
            >>> z = recnet(x)
            >>> print(z.shape)
            >>> print(torch.linalg.norm(x - z)/torch.linalg.norm(x))
            torch.Size([10, 1, 64, 64])
            tensor(5.8912e-06)
        """
        
        # Acquisition
        x = self.acqu(x)                     # shape x = [*, 2*M]

        # Reconstruction 
        x = self.reconstruct(x)             # shape x = [*, N]
        
        return x
    

    def reconstruct(self, x):
        r""" Reconstruction step of a reconstruction network
            
        Args:
            :attr:`x`: raw measurement vectors
        
        Shape:
            :attr:`x`: :math:`(BC,2M)`
            
            :attr:`output`: :math:`(BC,1,H,W)`
        
        Example:
            >>> B, C, H, M = 10, 1, 64, 64**2
            >>> Ord = np.ones((H,H))
            >>> meas = HadamSplit(M, H, Ord)
            >>> noise = NoNoise(meas)
            >>> prep = SplitPoisson(1.0, M, H**2)
            >>> recnet = PinvNet(noise, prep) 
            >>> x = torch.rand((B*C,2*M), dtype=torch.float)
            >>> z = recnet.reconstruct(x)
            >>> print(z.shape)
            torch.Size([10, 1, 64, 64])
        """
        # Preprocessing in the measurement domain
        x = self.prep(x) # shape x = [b*c, M]
    
        # measurements to image-domain processing
        x = self.pinv(x, self.acqu.meas_op)               # shape x = [b*c,N]
                
        # Image-domain denoising
        x = self.denoi(x)                       
        
        return x
    
#%%
from spyrit.misc.walsh_hadamard import walsh_matrix
from spyrit.misc.statistics import data_loaders_stl10
from spyrit.misc.disp import imagesc
from spyrit.core.meas import LinearSplit
from spyrit.core.prep import SplitPoisson
from spyrit.core.noise import NoNoise, PoissonApproxGauss

M = 24
N = 64

# A batch of images
dataloaders = data_loaders_stl10('../../data', img_size=N, batch_size=10)  
x, _ = next(iter(dataloaders['train']))
x = (x.view(-1,N,N) + 1)/2 

# Associated measurements
H = walsh_matrix(N)
H = H[:M]

# Raw measurement
linop = LinearSplit(H, pinv=True)
noise = NoNoise(linop)
y = noise(x)

# Reconstruction with method #1
prep = SplitPoisson(1.0, linop)
m = prep(y)
x1 = linop.pinv(m)

# Reconstruction with method #2
recnet = Pinv1dNet(noise, prep)
x2 = recnet.reconstruct(y)

# plot
imagesc(x[0,:,:])
imagesc(y[0,:,:].T)
imagesc(m[0,:,:].T)
imagesc(x1[0,:,:])
imagesc(x2[0,:,:])

#%%
from spyrit.misc.walsh_hadamard import walsh_matrix
from spyrit.misc.statistics import data_loaders_stl10
from spyrit.misc.disp import imagesc
from spyrit.core.prep import SplitPoisson
from spyrit.core.noise import NoNoise, PoissonApproxGauss

M = 24
N = 64

# A batch of images
dataloaders = data_loaders_stl10('../../data', img_size=N, batch_size=10)  
x, _ = next(iter(dataloaders['train']))
x = (x.view(-1,N,N) + 1)/2 

# Associated measurements
H = walsh_matrix(N)
H = H[:M]

H_pos = np.zeros(H.shape)
H_neg = np.zeros(H.shape)
H_pos[H>0] = H[H>0]
H_neg[H<0] = -H[H<0]

# Raw measurement
linop = LinearSplitPN(H_pos, H_neg, pinv=True)
noise = NoNoise(linop)
y = noise(x)

# Reconstruction
recnet = Pinv1dNet(noise, prep)
x = recnet.reconstruct(y)

# plot
imagesc(x[0,:,:])
imagesc(y[0,:,:].T)
imagesc(m[0,:,:].T)
imagesc(x[0,:,:])

#%% 
from spyrit.core.recon import TikhonovMeasurementPriorDiag
class DC1dNet(nn.Module):
# ===========================================================================================
    r""" Denoised completion reconstruction network
    
    .. math:
        
    
    Args:
        :attr:`noise`: Acquisition operator (see :class:`~spyrit.core.noise`) 
        
        :attr:`prep`: Preprocessing operator (see :class:`~spyrit.core.prep`)
        
        :attr:`sigma`: UPDATE!! Tikhonov reconstruction operator of type 
        :class:`~spyrit.core.recon.TikhonovMeasurementPriorDiag()`
        
        :attr:`denoi` (optional): Image denoising operator 
        (see :class:`~spyrit.core.nnet`). 
        Default :class:`~spyrit.core.nnet.Identity`
    
    Input / Output:
        :attr:`input`: Ground-truth images with shape :math:`(B,C,H,W)` 
        
        :attr:`output`: Reconstructed images with shape :math:`(B,C,H,W)`
    
    Attributes:
        :attr:`Acq`: Acquisition operator initialized as :attr:`noise`
        
        :attr:`PreP`: Preprocessing operator initialized as :attr:`prep`
        
        :attr:`DC_Layer`: Data consistency layer initialized as :attr:`tikho`
        
        :attr:`Denoi`: Image denoising operator initialized as :attr:`denoi`

    
    Example:
        >>> B, C, H, M = 10, 1, 64, 64**2
        >>> Ord = np.ones((H,H))
        >>> meas = HadamSplit(M, H, Ord)
        >>> noise = NoNoise(meas)
        >>> prep = SplitPoisson(1.0, M, H*H)
        >>> sigma = np.random.random([H**2, H**2])
        >>> recnet = DCNet(noise,prep,sigma)
        >>> x = torch.FloatTensor(B,C,H,H).uniform_(-1, 1)
        >>> z = recnet(x)
        >>> print(z.shape)
        torch.Size([10, 1, 64, 64])
    """
    def __init__(self, 
                 noise, 
                 prep, 
                 sigma,
                 denoi = nn.Identity()):
        
        super().__init__()
        self.acqu = noise 
        self.prep = prep
        Perm = noise.meas_op.Perm.weight.data.cpu().numpy().T
        sigma_perm = Perm @ sigma @ Perm.T
        self.tikho = TikhonovMeasurementPriorDiag(sigma_perm, noise.meas_op.M)
        self.denoi = denoi
        
    def forward(self, x):
        r""" ! update ! Full pipeline of the reconstruction network
            
        Args:
            :attr:`x`: ground-truth images
        
        Shape:
            :attr:`x`: ground-truth images with shape :math:`(B,C,H,W)`
            
            :attr:`output`: reconstructed images with shape :math:`(B,C,H,W)`
        
        Example: 
            >>> B, C, H, M = 10, 1, 64, 64**2
            >>> Ord = np.ones((H,H))
            >>> meas = HadamSplit(M, H, Ord)
            >>> noise = NoNoise(meas)
            >>> prep = SplitPoisson(1.0, M, H*H)
            >>> sigma = np.random.random([H**2, H**2])
            >>> recnet = DCNet(noise,prep,sigma)
            >>> x = torch.FloatTensor(B,C,H,H).uniform_(-1, 1)
            >>> z = recnet(x)
            >>> print(z.shape)
            torch.Size([10, 1, 64, 64])
        """
        # Acquisition
        x = self.acqu(x)                     # shape x = [b*c, 2*M]
        # Reconstruction 
        x = self.reconstruct(x)             # shape x = [bc, 1, h,w]
        
        return x

    def reconstruct(self, x):
        r""" ! update ! Reconstruction step of a reconstruction network
            
        Args:
            :attr:`x`: raw measurement vectors
        
        Shape:
            :attr:`x`: raw measurement vectors with shape :math:`(BC,2M)`
            
            :attr:`output`: reconstructed images with shape :math:`(BC,1,H,W)`
        
        Example:
            >>> B, C, H, M = 10, 1, 64, 64**2
            >>> Ord = np.ones((H,H))
            >>> meas = HadamSplit(M, H, Ord)
            >>> noise = NoNoise(meas)
            >>> prep = SplitPoisson(1.0, M, H*H)
            >>> sigma = np.random.random([H**2, H**2])
            >>> recnet = DCNet(noise,prep,sigma)
            >>> x = torch.rand((B*C,2*M), dtype=torch.float)
            >>> z = recnet.reconstruct(x)
            >>> print(z.shape)
            torch.Size([10, 1, 64, 64])
        """    
        # Preprocessing
        var_noi = self.prep.sigma(x)
        x = self.prep(x)
    
        # measurements to image domain processing
        x_0 = torch.zeros_like(x, device = x.device)
        x = self.tikho(x, x_0, var_noi, self.acqu.meas_op)
        
        # Image domain denoising
        x = self.denoi(x)               
        
        return x        

class Hadam1dSplit(LinearSplit):
    r""" 
    Computes linear measurements from incoming images: :math:`y = Px`, 
    where :math:`P` is a linear operator (matrix) with positive entries and 
    :math:`x` is a vectorized image.
    
    The class is relies on a matrix :math:`H` with 
    shape :math:`(M,N)` where :math:`N` represents the number of pixels in the 
    image and :math:`M \le N` the number of measurements. The matrix :math:`P` 
    is obtained by splitting the matrix :math:`H` such that 
    :math:`P = \begin{bmatrix}{H_{+}}\\{H_{-}}\end{bmatrix}`, where 
    :math:`H_{+} = \max(0,H)` and :math:`H_{-} = \max(0,-H)`. 
    
    The matrix :math:`H` is obtained by retaining the first :math:`M` rows of 
    a permuted Hadamard matrix :math:`GF`, where :math:`G` is a 
    permutation matrix with shape with shape :math:`(M,N)` and :math:`F` is a 
    "full" Hadamard matrix with shape :math:`(N,N)`. The computation of a
    Hadamard transform :math:`Fx` benefits a fast algorithm, as well as the
    computation of inverse Hadamard transforms.
    
    .. note::
        :math:`H = H_{+} - H_{-}`
    
    Args:
        - :attr:`M`: Number of measurements
        - :attr:`h`: Image height :math:`h`. The image is assumed to be square.
        - :attr:`Ord`: Order matrix with shape :math:`(h,h)` used to compute the permutation matrix :math:`G^{T}` with shape :math:`(N, N)` (see the :mod:`~spyrit.misc.sampling` submodule)
    
    .. note::
        The matrix H has shape :math:`(M,N)` with :math:`N = h^2`.
        
    Example:
        >>> Ord = np.random.random([32,32])
        >>> meas_op = HadamSplit(400, 32, Ord)
    """

    def __init__(self, M: int, h: int, Ord: np.ndarray):
        
        F =  walsh2_matrix(h) # full matrix
        Perm = Permutation_Matrix(Ord)
        F = Perm@F # If Perm is not learnt, could be computed mush faster
        H = F[:M,:]
        w = h   # we assume a square image
        
        super().__init__(H)
        
        self.Perm = nn.Linear(self.N, self.N, False)
        self.Perm.weight.data=torch.from_numpy(Perm.T)
        self.Perm.weight.data=self.Perm.weight.data.float()
        self.Perm.weight.requires_grad=False
        self.h = h
        self.w = w
    
    def inverse(self, x: torch.tensor) -> torch.tensor:
        r""" Inverse transform of Hadamard-domain images 
        :math:`x = H_{had}^{-1}G y` is a Hadamard matrix.
        
        Args:
            :math:`x`:  batch of images in the Hadamard domain
            
        Shape:
            :math:`x`: :math:`(b*c, N)` with :math:`b` the batch size, 
            :math:`c` the number of channels, and :math:`N` the number of
            pixels in the image.
            
            Output: math:`(b*c, N)`
            
        Example:

            >>> y = torch.rand([85,32*32], dtype=torch.float)  
            >>> x = meas_op.inverse(y)
            >>> print('Inverse:', x.shape)
            Inverse: torch.Size([85, 1024])
        """
        # permutations
        # todo: check walsh2_S_fold_torch to speed up
        b, N = x.shape
        x = self.Perm(x)
        x = x.view(b, 1, self.h, self.w)
        # inverse of full transform
        # todo: initialize with 1D transform to speed up
        x = 1/self.N*walsh2_torch(x)       
        x = x.view(b, N)
        return x
    
    def pinv(self, x: torch.tensor) -> torch.tensor:
        r""" Pseudo inverse transform of incoming mesurement vectors :math:`x`
        
        Args:
            :attr:`x`:  batch of measurement vectors.
            
        Shape:
            x: :math:`(*, M)`
            
            Output: :math:`(*, N)`
            
        Example:
            >>> y = torch.rand([85,400], dtype=torch.float)  
            >>> x = meas_op.pinv(y)
            >>> print(x.shape)
            torch.Size([85, 1024])
        """
        x = self.adjoint(x)/self.N
        return x
    
#%%
from spyrit.misc.walsh_hadamard import walsh_matrix
from spyrit.misc.statistics import data_loaders_stl10
from spyrit.misc.disp import imagesc
from spyrit.core.prep import SplitPoisson
from spyrit.core.noise import NoNoise, PoissonApproxGauss

M = 24
N = 64

# A batch of images
dataloaders = data_loaders_stl10('../../data', img_size=N, batch_size=10)  
x, _ = next(iter(dataloaders['train']))

# Associated measurements
H = walsh_matrix(N)
H = H[:M]

H_pos = np.zeros(H.shape)
H_neg = np.zeros(H.shape)
H_pos[H>0] = H[H>0]
H_neg[H<0] = -H[H<0]

# Raw measurement
linop = LinearSplitPN(H_pos, H_neg, pinv=True)
noise = NoNoise(linop)
y = noise(x)

# Reconstruction
sigma = np.eye(N)
recnet = DC1dNet(noise, prep, sigma)
x = recnet.reconstruct(y)

# plot
imagesc(x[0,:,:])
imagesc(y[0,:,:].T)
imagesc(m[0,:,:].T)
imagesc(x[0,:,:])

#%%
class Tikhonov(nn.Module): 
# ===========================================================================================   
    r"""
    Tikhonov regularization
    
    Considering linear measurements :math:`y = Hx`, where :math:`H` is the
    measurement matrix and :math:`x` is a vectorized image, it estimates 
    :math:`x` from :math:`y` by minimizing
    
    .. math::
        \| y - Hx \|^2_{\Sigma^{-1}_\alpha} + \|Lx - \ell_0\|^2_{\Sigma^{-1}}
    
    where :math:`\ell_0` is a mean prior, :math:`\Sigma` is a covariance 
    prior, and :math:`\Sigma_\alpha` is the measurement noise covariance. 
    
    The class is constructed from :math:`\Sigma` and :math:`L`.
    
    Args:
        - :attr:`sigma`:  covariance prior with shape :math:`(N, N)`
        - :attr:`M`: number of measurements
    
        
    Attributes:
        :attr:`comp`: The learnable completion layer initialized as 
        :math:`\Sigma_1 \Sigma_{21}^{-1}`. This layer is a :class:`nn.Linear`
        
        :attr:`denoi`: The learnable denoising layer initialized from 
        :math:`\Sigma_1`.
    
    Example:
        >>> sigma = np.random.random([32*32, 32*32])
        >>> recon_op = TikhonovMeasurementPriorDiag(sigma, 400)            
    """
    def __init__(self, Sigma: np.array, L: np.array, A: np.array):
        super().__init__()
        
        Sigma = L @ Sigma @ L_inv
        Bt = Sigma @ A.T
        C = A @ Bt
        
        M = A.shape[0]
        N = A.shape[1]
        
        self.comp.weight.data=torch.from_numpy(W)
        self.comp.weight.data=self.comp.weight.data.float()
        self.comp.weight.requires_grad=False
        
        self.C = nn.Linear(M, M, False)
        self.C.weight.data = torch.from_numpy(C)
        self.C.weight.data = self.comp.weight.data.float()
        self.C.weight.requires_grad = False
        
        self.Bt = nn.Linear(M, N, False)
        self.Bt.weight.data = torch.from_numpy(Bt)
        self.Bt.weight.data = self.comp.weight.data.float()
        self.Bt.weight.requires_grad = False
        
    def forward(self, 
                x: torch.tensor, 
                x_0: torch.tensor, 
                var: torch.tensor, 
                meas_op: HadamSplit) -> torch.tensor:
        r"""
        
        The solution is computed as
        
        .. math::
            \hat{x} = x_0 +C(C + \Sigma_\alpha)^{-1} (y - H x_0)
        
        with :math:`y_1 = C(C + \Sigma_\alpha)^{-1} (y - GF x_0)` and 
        :math:`y_2 = \Sigma_1 \Sigma_{21}^{-1} y_1`, where 
        :math:`\Sigma = \begin{bmatrix} \Sigma_1 & \Sigma_{21}^\top \\ \Sigma_{21} & \Sigma_2\end{bmatrix}`
        and  :math:`D_1 =\textrm{Diag}(\Sigma_1)`. Assuming the noise 
        covariance :math:`\Sigma_\alpha` is diagonal, the matrix inversion 
        involded in the computation of :math:`y_1` is straigtforward.
        
        
        Args:
            - :attr:`x`: A batch of measurement vectors :math:`y`
            - :attr:`x_0`: A batch of prior images :math:`x_0`
            - :attr:`var`: A batch of measurement noise variances :math:`\Sigma_\alpha`
            - :attr:`meas_op`: A measurement operator that provides :math:`GF` and :math:`F^{-1}`
            
        Shape:
            - :attr:`x`: :math:`(*, M)`
            - :attr:`x_0`: :math:`(*, N)`
            - :attr:`var` :math:`(*, M)`
            - Output: :math:`(*, N)`
            
        Example:
            >>> B, H, M = 85, 32, 512
            >>> sigma = np.random.random([H**2, H**2])
            >>> recon_op = TikhonovMeasurementPriorDiag(sigma, M)
            >>> Ord = np.ones((H,H))
            >> meas = HadamSplit(M, H, Ord)
            >>> y = torch.rand([B,M], dtype=torch.float)  
            >>> x_0 = torch.zeros((B, H**2), dtype=torch.float)
            >>> var = torch.zeros((B, M), dtype=torch.float)
            >>> x = recon_op(y, x_0, var, meas)
            torch.Size([85, 1024])       
        """
        x = x - meas_op.forward_H(x_0)
        y1 = torch.mul(self.denoi(var),x)
        y2 = self.comp(y1)

        y = torch.cat((y1,y2),-1)
        x = x_0 + meas_op.inverse(y) 
        return x