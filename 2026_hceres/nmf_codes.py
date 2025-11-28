import numpy as np
import tensorly as tl
import torch
from tensorly.solvers.nnls import hals_nnls
from scipy.special import kl_div

# Putting some codes here for nmf since they are not (yet) available online
def compute_error(V, WH):
    """
    Elementwise Kullback Leibler divergence

    Parameters
    ----------
    V : 2darray
        input data, left hand side of KL
    WH : 2d array
        right hand side of KL
    ind0 : boolean 2d array, optional
        table with True where V is not small, by default None
    ind1 : boolean 2d array, optional
        table with False where V is almost 0, by default None

    Returns
    -------
    float
        elementwise KL divergence

    """
    return tl.sum(kl_div(V, WH))    


def spa(X, r, tol=1e-8, normalize=False):
    """Computes pure spectra and pure pixels from nonnegative matrix X, using the successive  projection algorithm (SPA).

    Parameters
    ----------
    X : numpy 2d array
        input matrix on which to compute separable nonnegative NMF
    r : int
        number of spectra to extract from pure pixels
    tol : float, optional
        tolerance for stopping inner iterations, by default 1e-8
    normalize : bool, optional
        normalize the input on the simplex, by default false

    Returns
    -------
    list
        indices of selected pure pixels
    numpy 2d array
        the estimated pure spectra
    numpy 2d array
        the estimated abundances
    """
    # Get dimensions
    _, n = tl.shape(X)
    X = tl.copy(X)  # copy in local scope to avoid modifying input
    
    # Optionally normalize so that columns of X sum to one
    if normalize:
        for j in range(n):
            X[:, j] = X[:, j]/tl.sum(tl.abs(X[:, j]))

    # Init
    # Set of selected indices
    K = [0 for i in range(r)]
    # Norm of columns of input X
    normX0 = tl.norm(X, axis=0)**2
    R = tl.copy(X)
    # Max of the columns norm
    nXmax = tl.max(normX0)
    # Init residual
    normR = tl.copy(normX0)

    # SPA loop
    i = 0
    while i < r and tl.sqrt(tl.max(normR)/nXmax) > tol:
        print(i, K)
        # Select column of X with largest l2-norm
        a = tl.max(normR)
        # Check ties up to 1e-6 precision
        b = np.argwhere((a - normR) / a <= 1e-6)
        if tl.ndim(b) > 1:
            # b should be 1d array, reduce to 1d
            b = tl.reshape(b, (-1,))
        # In case of a tie, select column with largest norm of the input matrix
        d = np.argmax(normX0[b])
        b = b[d]
        #print(b[d])
        # Save index of selected column, and column itself
        K[i] = int(b)
        U = X[:, K[:i+1]]  # can be optimimed by pre-allocations
       
        # Update residual coefficient
        R = R - tl.tenalg.outer([U[:, -1], U[:, -1].T@R])/tl.norm(U[:, -1])**2
        normR = tl.norm(R, axis=0)
        
        # Update residual (correct?)
        #for j in range(i-1):  # ??i or i-1
            #U[:, i] = U[:, i] - U[:, j] * np.dot(U[:, j], U[:, i])
        #U[:, i] = U[:, i]/np.linalg.norm(U[:, i])
        #normR = normR - (X.T @ U[:, i]) ** 2 # TODO BUGGED (?)

        # Increment iterator
        i += 1

    if len(np.unique(K)) == len(K):
        H = hals_nnls(U.T@X, U.T@U, n_iter_max=100, tol=1e-8)
    else:
        H = None
        print("There are duplicates in K, cannot estimate H")
    print(f"Returning {K} as estimated pure pixel indices")

    return K, U, H


def MU_SinglePixel(Y, H, A0, W0, lmbd=None, maxA=None, niter=1000, n_iter_inner=20, eps=1e-8, verbose=True, print_it=10):
    """
    Multiplicative Update algorithm for Nonnegative Matrix Factorization with Kullback-Leibler divergence.
    The model is Y ~ P(\alpha WAH^T), where Y is the observation matrix, W is the endmember matrix, A is the abundance matrix to be estimated, H is a positive Hadamard matrix, and N is noise. Parameter alpha accounts for the number of photons, the higher the less noisy.
    
    The optimization problem solved is:
        min_{A,W} D_KL(Y/alpha || WAH^T) + lambda * (||A||_1 + ||W||_1)
    subject to A >= 0, W >= 0
    
    Parameters
    ----------
    A0 : torch.Tensor
        Initial abundance matrix (PxM)
    W : torch.Tensor
        Endmember matrix (BxP)
    Y : torch.Tensor
        Data matrix, should be normalized by photon count (BxN)
    H : torch.Tensor
        Observation matrix, typically a positive Hadamard matrix (MxN)
    lmbd : list or float or None, optional
        Regularization parameter for the l1 norm on W and A, by default 1
        If a list is provided, it should contain two elements: [lambda_W, lambda_A]
        Otherwise the same lambda is used for both W and A.
    niter : int, optional
        Number of iterations, by default 1000
    eps : float, optional
        Small constant to avoid division by zero, by default 1e-8
    Atrue : torch.Tensor, optional
        Ground truth abundance matrix for oracle metrics, by default None
    print_metrics : bool, optional
        Whether to print metrics during iterations, by default True
    """
    M, _ = H.shape
    B, _ = W0.shape
    A = torch.clone(A0)
    W = torch.clone(W0)
    sumH = torch.ones((B, M))@H
    #sumH = torch.sum(H, axis=0)
    costs = []
    
    # Handle lambda parameter
    if lmbd is None:
        lmbd = 0.0
    if isinstance(lmbd, (list, tuple)):
        lmbd_W, lmbd_A = lmbd
    else:
        lmbd_W = lmbd
        lmbd_A = lmbd

    for k in range(niter):

        WtsumH = W.T@sumH
        for _ in range(n_iter_inner):
            A = A * ((W.T@(Y/(W@(A@H.T))))@H) / (WtsumH + lmbd_A)
            if maxA is not None:
                A = torch.clamp(A, min=eps, max=maxA)
            else:
                A = torch.clamp(A, min=eps)

        AH = A@H.T
        AHtsum = torch.sum(AH, axis=1)[None, :]
        for _ in range(n_iter_inner):
            W = W * ((Y/(W@AH))@AH.T) / (AHtsum + lmbd_W)
            W = torch.clamp(W, min=eps)
            
        if k % print_it == 0:
            WAHt = W@AH
            c = compute_error(Y, WAHt) + lmbd_A*torch.sum(A) + lmbd_W*torch.sum(W)
            costs.append(c.cpu().detach().numpy())

            if verbose:
                print(f'Iteration {k}, Cost: {c.cpu().detach().numpy()}')
                #print('Erreur :', e.cpu().detach().numpy())
                #print('PSNR :', torch.mean(p))

    return W, A, costs