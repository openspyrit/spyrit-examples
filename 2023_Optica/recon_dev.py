# -*- coding: utf-8 -*-
"""
Created on Fri Jun  2 12:03:41 2023

@author: ducros
"""
import torch
import torch.nn as nn
import numpy as np
from spyrit.core.recon import PseudoInverse
from spyrit.core.recon import TikhonovMeasurementPriorDiag

# =============================================================================
class Pinv1Net(nn.Module):
# =============================================================================
    r""" Pseudo inverse reconstruction across 1 dimension
    
    .. math:
        
        
    Args:
        :attr:`noise`: Acquisition operator (see :class:`~spyrit.core.noise`) 
        
        :attr:`prep`: Preprocessing operator (see :class:`~spyrit.core.prep`)
        
        :attr:`denoi` (optional): Image denoising operator 
        (see :class:`~spyrit.core.nnet`). 
        Default :class:`~spyrit.core.nnet.Identity`
    
    Input / Output:
        :attr:`input`: Ground-truth images with shape :math:`(b,c,h,w)` 
        
        :attr:`output`: Reconstructed images with shape :math:`(b,c,h,w)`
    
    Attributes:
        :attr:`acqu`: Acquisition operator initialized as :attr:`noise`
        
        :attr:`prep`: Preprocessing operator initialized as :attr:`prep`
        
        :attr:`pinv`: Analytical reconstruction operator initialized as 
        :class:`~spyrit.core.recon.PseudoInverse()`
        
        :attr:`denoi`: Image denoising operator initialized as :attr:`denoi`

    Example:
        >>> b,c,h,n = 10,1,48,64
        >>> H = np.random.rand(15,n)
        >>> meas = LinearSplit(H, pinv=True)
        >>> noise = NoNoise(meas)
        >>> prep = SplitPoisson(1.0, meas)
        >>> recnet = Pinv1Net(noise, prep)
        >>> x = torch.FloatTensor(b,c,h,n).uniform_(-1, 1)
        >>> z = recnet(x)
        >>> print(z.shape)
        >>> print(torch.linalg.norm(x - z)/torch.linalg.norm(x))
        torch.Size([10, 1, 64, 64])
        tensor(5.8912e-06)
        
    .. note::
        The measurement operator applies to the last dimension of the input 
        tensor, contrary :class:`~spyrit.core.recon.PinvNet` where it applies 
        to the last two dimensions. In both cases, the denoising operator 
        applies to the last two dimensions.
        
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
            :attr:`x`:  Ground-truth images 
        
        Shape:
            :attr:`x`: Ground-truth images with shape :math:`(b,c,h,w)`
            
            :attr:`output`: Reconstructed images with shape :math:`(b,c,h,w)`
        
        Example:
            >>> b,c,h,n = 10,1,48,64
            >>> H = np.random.rand(15,n)
            >>> meas = LinearSplit(H, pinv=True)
            >>> noise = NoNoise(meas)
            >>> prep = SplitPoisson(1.0, meas)
            >>> recnet = Pinv1Net(noise, prep)
            >>> x = torch.FloatTensor(b,c,h,n).uniform_(-1, 1)
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
            :attr:`x`: Raw measurement vectors
        
        Shape:
            :attr:`x`: Raw measurement vectors with shape :math:`(*,2M)`
            
            :attr:`output`: :math:`(*,W)`
        
        Example:
            >>> b,c,h,n,m = 10,1,48,64,15
            >>> H = np.random.rand(m,n)
            >>> meas = LinearSplit(H, pinv=True)
            >>> noise = NoNoise(meas)
            >>> prep = SplitPoisson(1.0, meas)
            >>> recnet = Pinv1Net(noise, prep)
            >>> x = torch.FloatTensor(b,c,h,2*m).uniform_(-1, 1)
            >>> z = recnet.reconstruct(x)
            >>> print(z.shape)
        """
        # Preprocessing in the measurement domain
        x = self.prep(x) # shape x = [b*c, M]
    
        # measurements to image-domain processing
        x = self.pinv(x, self.acqu.meas_op)               # shape x = [b*c,N]
                
        # Image-domain denoising
        x = self.denoi(x)                       
        
        return x
    
    def reconstruct_expe(self, x):
        r""" Reconstruction step of a reconstruction network
        
        Same as :meth:`reconstruct` reconstruct except that:
            
        1. The preprocessing step estimates the image intensity for normalization
        
        2. The output images are "denormalized", i.e., have units of photon counts
            
        Args:
            :attr:`x`: raw measurement vectors
        
        Shape:
            :attr:`x`: Raw measurement vectors with shape :math:`(*,2M)`
            
            :attr:`output`: :math:`(*,W)`

        """   
        # Preprocessing
        x, N0_est = self.prep.forward_expe(x, self.acqu.meas_op) # shape x = [b*c, M]
        print(N0_est)
    
        # measurements to image domain processing
        x = self.pinv(x, self.acqu.meas_op)               # shape x = [b*c,N]
        
        # Image-domain denoising
        x = self.denoi(x)                               # shape x = [b*c,1,h,w]
        print(x.max())
        
        # Denormalization 
        x = self.prep.denormalize_expe(x, N0_est, self.acqu.meas_op.h, 
                                                  self.acqu.meas_op.w)
        return x

# =============================================================================    
class DC1Net(nn.Module):
# =============================================================================
    r""" Denoised completion reconstruction network
    
    .. math:
        
    
    Args:
        :attr:`noise`: Acquisition operator (see :class:`~spyrit.core.noise`) 
        
        :attr:`prep`: Preprocessing operator (see :class:`~spyrit.core.prep`)
        
        :attr:`sigma`: Covariance prior`
        
        :attr:`denoi` (optional): Image denoising operator 
        (see :class:`~spyrit.core.nnet`). 
        Default :class:`~spyrit.core.nnet.Identity`
    
    Input / Output:
        :attr:`input`: Ground-truth images with shape :math:`(B,C,H,W)` 
        
        :attr:`output`: Reconstructed images with shape :math:`(B,C,H,W)`
    
    Attributes:
        :attr:`acqu`: Acquisition operator initialized as :attr:`noise`
        
        :attr:`prep`: Preprocessing operator initialized as :attr:`prep`
        
        :attr:`tikho`: Data consistency layer initialized as :attr:`tikho`
        
        :attr:`denoi`: Image denoising operator initialized as :attr:`denoi`

    
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
        #Perm = noise.meas_op.Perm.weight.data.cpu().numpy().T
        # CHECK bellow as it may be the transpose of it
        sigma = torch.from_numpy(sigma)
        sigma_perm = noise.meas_op.Perm(noise.meas_op.Perm(sigma.mT)).mT
        sigma_perm = sigma_perm.numpy()
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
        size = x.shape[:-1] + torch.Size([self.acqu.meas_op.N])
        x_0 = torch.zeros(size, device=x.device)
        x = self.tikho(x, x_0, var_noi, self.acqu.meas_op)
        
        # Image domain denoising
        x = self.denoi(x)               
        
        return x

# =============================================================================
class Tikhonov(nn.Module): 
# ============================================================================= 
    r"""
    Tikhonov regularization
    
    Considering linear measurements :math:`y = Ax`, where :math:`A` is the
    measurement matrix and :math:`x` is a vectorized image, it estimates 
    :math:`x` from :math:`y` by minimizing
    
    .. math::
        \| y - Ax \|^2_{\Sigma^{-1}_\alpha} + \|x - x_0\|^2_{\Sigma^{-1}}
    
    where :math:`\ell_0` is a mean prior, :math:`\Sigma` is a covariance 
    prior, and :math:`\Sigma_\alpha` is the measurement noise covariance. 
    
    The class is constructed from :math:`A` and :math:`\Sigma`.
    
    Args:
        - :attr:`A` (torch.tensor): measurement matrix with shape :math:`(M, N)`
        
        - :attr:`Sigma` (torch.tensor):  noise covariance with shape :math:`(M, M)` ???
        
        - :attr:`Sigma` (torch.tensor):  covariance prior with shape :math:`(N, N)`
    
        
    Attributes:
        :attr:`B`: The learnable completion layer initialized as 
        :math:`\Sigma_1 \Sigma_{21}^{-1}`. This layer is a :class:`nn.Linear`
        
        :attr:`C`: The learnable denoising layer initialized from 
        :math:`\Sigma_1`.
    
    Example:
        >>> sigma = np.random.random([32*32, 32*32])
        >>> recon_op = TikhonovMeasurementPriorDiag(sigma, 400)            
    """
    def __init__(self,  A: torch.tensor, Sigma: torch.tensor):
        super().__init__()
        B = A @ Sigma      # NB: we assume Sigma = Sigma.mT
        C = A @ B.mT
        self.register_buffer('B', B)
        self.register_buffer('C', C)
        
        
    def forward(self, x: torch.tensor, #x_0: torch.tensor, 
                      cov: torch.tensor) -> torch.tensor:
        r"""
        
        The solution is computed as
        
        .. math::
            \hat{x} = x_0 + B^\top (C + \Sigma_\alpha)^{-1} (y - A x_0)
        
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
        z = torch.linalg.lstsq(self.C + cov, x).solution # z = (C + cov)^-1 x
        x = z @ self.B                                   # x = B^T z 
        return x

# =============================================================================    
class Tikho1Net(nn.Module):
# =============================================================================
    def __init__(self, noise, prep, sigma: np.array, denoi = nn.Identity()):
        
        super().__init__()
        self.acqu = noise 
        self.prep = prep
        device = noise.meas_op.H.weight.device
        sigma = torch.as_tensor(sigma, dtype=torch.float32, device=device)
        self.tikho = Tikhonov(noise.meas_op.get_H(), sigma)
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
        cov_meas = self.prep.sigma(x)
        x = self.prep(x)
    
        # covariance of measurements
        cov_meas = torch.diag_embed(cov_meas) # 
        
        # measurements to image domain processing
        #size = x.shape[:-1] + torch.Size([self.acqu.meas_op.N])
        #x_0 = torch.zeros(size, device=x.device)
        x = self.tikho(x, cov_meas)
        
        # Image domain denoising
        x = self.denoi(x)               
        
        return x