from typing import Optional, Union
import torch
from torch import Tensor
from gpytorch.kernels import Kernel, RBFKernel
from gpytorch.distributions import MultivariateNormal
    
     

def Z_to_X(Z: Tensor) -> Tensor:
    deterministic_fill = torch.zeros_like(Z)
    return torch.cat((Z, deterministic_fill), dim=-1)    


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
        dim = self.lengthscale.shape[-1] // 2

        # when the inputs are of different sizes, what happens? (160 x 512 / 16 x 512)
        A = (self.lengthscale[..., :dim] ** 2 + 
             x1[..., dim:].unsqueeze(-2) + x2[..., dim:].unsqueeze(-3)
        ).rsqrt() * self.lengthscale[..., :dim]
        B1 = (self.lengthscale[..., :dim] ** 2 + x1[..., dim:]).rsqrt() * self.lengthscale[..., :dim]
        B2 = (self.lengthscale[..., :dim] ** 2 + x2[..., dim:]).rsqrt() * self.lengthscale[..., :dim]
        # output: batch_dim1 * batch_dim2 * dim (16 * 160 * 256)
        # how first element should scale * how second element should scale * per dimension
        z1 = x1[..., :dim] * B1
        z2 = x2[..., :dim] * B2
        
        return super().forward(Z_to_X(z1), Z_to_X(z2), diag=diag) * torch.prod(A, dim=-1)
