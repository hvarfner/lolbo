from typing import Optional, Union
import torch
from torch import Tensor
from gpytorch.kernels import Kernel, RBFKernel
from gpytorch.distributions import MultivariateNormal
    

class ZRBFKernel(RBFKernel):
    
    has_lengthscale = True
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def forward(
            self, 
            x1: Tensor, # batch_shape x 2D
            x2: Tensor, # batch_shape x 2D
            diag: bool = False, 
            **params
    ):
        print(x1.shape, x2.shape)
        breakpoint()        
        double_dim = x1.shape[-1]
        if double_dim % 2 != 0:
            raise ValueError(f"The double dim must be divisible by 2!, is {double_dim}")
        dim = double_dim // 2

        # when the inputs are of different sizes, what happens? (160 x 512 / 16 x 512)
        A = (self.lengthscale ** 2 + 
             x1[..., dim:].unsqueeze(-2) + x2[..., dim:].unsqueeze(-3)
        ).rsqrt() * self.lengthscale
        B1 = (self.lengthscale ** 2 + x1[..., dim:]).rsqrt() * self.lengthscale
        B2 = (self.lengthscale ** 2 + x2[..., dim:]).rsqrt() * self.lengthscale
        # output: batch_dim1 * batch_dim2 * dim (16 * 160 * 256)
        # how first element should scale * how second element should scale * per dimension
        z1 = x1[..., :dim] * B1
        z2 = x2[..., :dim] * B2

        return super().forward(z1, z2, diag=diag) * torch.prod(A, dim=-1)
