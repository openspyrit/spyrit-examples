import torch
import torch.nn as nn
import math

import spyrit.core.meas as meas

cuda = True if torch.cuda.is_available() else False
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor


class SoftShk(nn.Module):
    def __init__(self, n_ch=64):
        super(SoftShk, self).__init__()

        self.n_ch = n_ch
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()

    def forward(self, x, l):
        out = self.relu1(x - l) - self.relu2(-x - l)
        return out


class DFBBlock(nn.Module):
    def __init__(self, channels, features, kernel_size=3, padding=1):
        super(DFBBlock, self).__init__()

        self.conv = nn.Conv2d(
            in_channels=channels,
            out_channels=features,
            kernel_size=kernel_size,
            padding=padding,
            bias=False,
        )
        self.conv_t = nn.ConvTranspose2d(
            in_channels=features,
            out_channels=channels,
            kernel_size=kernel_size,
            padding=padding,
            bias=False,
        )

        self.conv_t.weight = self.conv.weight
        self.nl = SoftShk(n_ch=channels)
        self.lip2 = 1e-3
        self.xlip2 = None

    def forward(self, u_in, x_ref, l, eval=False):
        if eval == False:
            # lip2 = self.op_norm2(x_ref.shape)
            self.lip2 = self.op_norm2(
                (1, x_ref.shape[1], x_ref.shape[2], x_ref.shape[3])
            )
        gamma = 1.8 / self.lip2
        tmp = x_ref - self.conv_t(u_in)
        g1 = u_in + gamma * self.conv(tmp)
        p1 = g1 - gamma * self.nl(g1 / gamma, l / gamma)
        return p1

    # def forward_eval(self, u_in, x_ref, l):  # Suggestion: add the eval function as a param in the function
    #     gamma = 1.8 / self.lip
    #     tmp = x_ref - self.conv_t(u_in)
    #     g1 = u_in + gamma * self.conv(tmp)
    #     p1 = g1 - gamma * self.nl(g1 / gamma, l / gamma)
    #     return p1

    def op_norm2(self, im_size, eval=False):
        tol = 1e-4
        max_iter = 300
        with torch.no_grad():
            if self.xlip2 is None or eval == True:
                xtmp = torch.randn(*im_size).type(Tensor)
                xtmp = xtmp / torch.linalg.norm(xtmp.flatten())
                val = 1
            else:
                xtmp = self.xlip2
                val = self.lip2

            for k in range(max_iter):
                old_val = val
                xtmp = self.conv_t(self.conv(xtmp))
                val = torch.linalg.norm(xtmp.flatten())
                rel_val = torch.absolute(val - old_val) / old_val
                if rel_val < tol:
                    break
                xtmp = xtmp / val

            self.xlip2 = xtmp
            # print('it ', k, ' val ', val)
        return val

    def update_lip(self, im_size):
        with torch.no_grad():
            self.lip2 = self.op_norm2(im_size, eval=True).item()


class DFBNet(nn.Module):
    def __init__(self, channels=3, features=64, num_of_layers=20):
        super(DFBNet, self).__init__()

        kernel_size, padding = 3, 1
        self.in_conv = nn.Conv2d(
            in_channels=channels,
            out_channels=features,
            kernel_size=kernel_size,
            padding=padding,
            bias=False,
        )
        self.out_conv = nn.ConvTranspose2d(
            in_channels=features,
            out_channels=channels,
            kernel_size=kernel_size,
            padding=padding,
            bias=False,
        )
        self.out_conv.weight = self.in_conv.weight

        self.linlist = nn.ModuleList(
            [
                DFBBlock(
                    channels=channels,
                    features=features,
                    kernel_size=kernel_size,
                    padding=padding,
                )
                for _ in range(num_of_layers - 2)
            ]
        )

    def forward(self, xref, xin, l, u=None):

        if u is None:
            u = self.in_conv(xin)

        for _ in range(len(self.linlist)):
            # print('Looping algo ', _, 'u shapes = ', u.shape, ' l shape ', l.shape)
            u = self.linlist[_](u, xref, l, eval=False)

        out = torch.clamp(xref - self.out_conv(u), min=0, max=1)

        return out

    def forward_eval(self, xref, xin, l, u=None):
        with torch.no_grad():  # Addition by nico
            if u is None:
                u = self.in_conv(xin)

            for _ in range(len(self.linlist)):
                u = self.linlist[_].forward(u, xref, l, eval=True)

            out = torch.clamp(xref - self.out_conv(u), min=0, max=1)

            return out, u

    def update_lip(self, im_size):
        for _ in range(len(self.linlist)):
            self.linlist[_].update_lip(im_size)

    def print_lip(self):
        print("Norms of linearities:")
        for _ in range(len(self.linlist)):
            print("Layer ", str(_), self.linlist[_].lip2)


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        nn.init.kaiming_normal_(m.weight.data, a=0, mode="fan_in")
    elif classname.find("Linear") != -1:
        nn.init.kaiming_normal_(m.weight.data, a=0, mode="fan_in")
    elif classname.find("BatchNorm") != -1:
        m.weight.data.normal_(mean=0, std=math.sqrt(2.0 / 9.0 / 64.0)).clamp_(
            -0.025, 0.025
        )
        nn.init.constant(m.bias.data, 0.0)


def get_model(architecture, n_ch=3, features=64, num_of_layers=20):

    if architecture == "DFBNet":
        net = DFBNet(channels=n_ch, features=features, num_of_layers=num_of_layers)
        lr = 1e-3
        clip_val = 1

    net.apply(weights_init_kaiming)

    net = nn.DataParallel(net)
    if torch.cuda.is_available():  # Move to GPU if possible
        net = net.cuda()

    return net, clip_val, lr


def load_checkpoint(model, filename):
    checkpoint = torch.load(
        filename, map_location=lambda storage, loc: storage, weights_only=True
    )
    try:
        model.module.load_state_dict(checkpoint, strict=True)
    except:
        model.module.load_state_dict(checkpoint.module.state_dict(), strict=True)

    model.eval()
    if "dpir" in filename:
        for k, v in model.module.named_parameters():
            print(k)
            v.requires_grad = False

    return model


def load_model(pth=None, n_ch=3, features=64, num_of_layers=20):
    """Having trouble loading a model that was trained using DataParallel and then sending it to DataParallel.
    The loading works but drastic slowdown.
    Latest option is to move to CPU, load, and then move back to Parallel GPU.
    """

    print("Loaded:", pth)
    model, _, _ = get_model("DFBNet", n_ch, features, num_of_layers)
    model = load_checkpoint(model, pth)

    return model


class DualPGD(nn.Module):
    r"""Pseudo inverse reconstruction network

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

        :attr:`denoi`: Image denoising operator initialized as :attr:`denoi`


    Example:
        >>> B, C, H, M = 10, 1, 64, 64**2
        >>> Ord = np.ones((H,H))
        >>> meas = HadamSplit(M, H, Ord)
        >>> noise = NoNoise(meas)
        >>> prep = SplitPoisson(1.0, M, H*H)
    """

    def __init__(
        self,
        acqu: meas.HadamSplit2d,
        prep,
        denoi,
        gamma=1.0,
        mu=1.0,
        iter_stop=1000,
        norm_stop=1e-4,
    ):
        super().__init__()
        self.mu = mu
        self.gamma = gamma
        self.norm_stop = norm_stop
        self.iter_stop = iter_stop
        # nn.module
        self.acqu = acqu
        self.prep = prep
        self.denoi = denoi

    def forward(self, x):
        r"""Full pipeline of reconstrcution network

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

        # b, c, _, _ = x.shape

        # Acquisition
        # x = x.view(b * c, self.acqu.meas_op.N)  # shape x = [b*c,h*w] = [b*c,N]
        x = self.acqu(x)  # shape x = [b*c, 2*M]

        # Reconstruction
        x = self.reconstruct(x)  # shape x = [bc, 1, h,w]
        # x = x.view(b, c, self.acqu.meas_op.h, self.acqu.meas_op.w)

        return x

    def acquire(self, x):
        r"""Simulate data acquisition

        Args:
            :attr:`x`: ground-truth images

        Shape:
            :attr:`x`: ground-truth images with shape :math:`(B,C,H,W)`

            :attr:`output`: measurement vectors with shape :math:`(BC,2M)`

        Example:
            >>> B, C, H, M = 10, 1, 64, 64**2
            >>> Ord = np.ones((H,H))
            >>> meas = HadamSplit(M, H, Ord)
            >>> noise = NoNoise(meas)
            >>> prep = SplitPoisson(1.0, M, H*H)
            >>> recnet = PinvNet(noise, prep)
            >>> x = torch.FloatTensor(B,C,H,H).uniform_(-1, 1)
            >>> z = recnet.acquire(x)
            >>> print(z.shape)
            torch.Size([10, 8192])
        """

        # b, c, _, _ = x.shape

        # Acquisition
        # x = x.view(b * c, self.acqu.meas_op.N)  # shape x = [b*c,h*w] = [b*c,N]
        x = self.acqu(x)  # shape x = [b*c, 2*M]

        return x

    def reconstruct(self, x, exp=False):
        r"""Reconstruction step of a reconstruction network

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
        with torch.no_grad():

            # Measurement to image domain mapping
            # bc, _ = x.shape

            # Preprocessing in the measurement domain
            if exp:
                m, N0_est = self.prep.forward_expe(x, self.acqu.meas_op)
            else:
                m = self.prep(x)

            # init solution and dual variable
            x = self.acqu.fast_pinv(m)
            x = self.acqu.unvectorize(x)
            u = None

            for i in range(self.iter_stop):
                # x = x.view(bc, self.acqu.meas_op.N)
                x_old = (
                    x.clone()
                )  # Do we needs detach here? https://discuss.pytorch.org/t/clone-and-detach-in-v0-4-0/16861/21

                # gradient step (data fidelity with scaling + quadratic regularization)
                res = self.acqu.measure_H(x) - m
                x = x - self.gamma * self.acqu.unvectorize(self.acqu.adjoint_H(res))

                # proximal step (prior). NB: scaling required as the prox was
                # learned for images in [0,1]
                # x = x.view(
                #     bc, 1, self.acqu.meas_op.h, self.acqu.meas_op.w
                # )  # shape x = [b*c,1,h,w]
                x = (x + 1) / 2
                x, u = self.denoi.module.forward_eval(x, x, l=self.mu * self.gamma, u=u)
                x = 2 * x - 1

                # stopping criteria
                # x_old = x_old.view(bc, 1, self.acqu.meas_op.h, self.acqu.meas_op.w)
                norm_it = torch.linalg.vector_norm(
                    x - x_old
                ) / torch.linalg.vector_norm(x)

                # print(f'{i}: |x - x_old| / |x| = {norm_it}')
                if norm_it < self.norm_stop:
                    break

            if exp:
                x = self.prep.denormalize_expe(
                    x, N0_est, self.acqu.meas_shape[0], self.acqu.meas_shape[1]
                )
        return x
