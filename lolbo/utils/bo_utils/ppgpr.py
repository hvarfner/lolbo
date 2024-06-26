# ppgpr
import math
import torch
from torch import Tensor

import gpytorch
from gpytorch.models import ApproximateGP, ExactGP
from botorch.models.gpytorch import BatchedMultiOutputGPyTorchModel
from gpytorch.variational import CholeskyVariationalDistribution
from gpytorch.variational.variational_strategy import (
     VariationalStrategy, 
)
from gpytorch.variational.unwhitened_variational_strategy import (
     UnwhitenedVariationalStrategy, 
)
from torch.nn import Module
from .base import DenseNetwork
from lolbo.utils.bo_utils.latent_variational_strategy import (
     LatentVariationalStrategy, 
)
from gpytorch.priors import LogNormalPrior
from botorch.posteriors.gpytorch import GPyTorchPosterior
from lolbo.utils.bo_utils.zrbf import ZRBFKernel

# Multi-task Variational GP:
# https://docs.gpytorch.ai/en/v1.4.2/examples/04_Variational_and_Approximate_GPs/SVGP_Multitask_GP_Regression.html


class GPModel(ApproximateGP):
    def __init__(self, inducing_points, likelihood):
        variational_distribution = CholeskyVariationalDistribution(inducing_points.size(0) )
        variational_strategy = VariationalStrategy(
            self,
            inducing_points,
            variational_distribution,
            learn_inducing_locations=True
            )
        dim = inducing_points.shape[1]
        
        super(GPModel, self).__init__(variational_strategy)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(
             gpytorch.kernels.RBFKernel(ard_num_dims=dim)
        )
        self.num_outputs = 1
        self.likelihood = likelihood 

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
#model.covar_module.base_kernel.lengthscale
    def posterior(
            self, X, output_indices=None, observation_noise=False, *args, **kwargs
        ) -> GPyTorchPosterior:
            self.eval()  # make sure model is in eval mode
            # self.model.eval()
            self.likelihood.eval()
            dist = self.likelihood(self(X)) 

            return GPyTorchPosterior(mvn=dist)



class ZGPModel(ApproximateGP):
    def __init__(self, inducing_points, likelihood):
        variational_distribution = CholeskyVariationalDistribution(inducing_points.size(0) )
        variational_strategy = VariationalStrategy(
            self,
            inducing_points,
            variational_distribution,
            learn_inducing_locations=True
            )
        dim = inducing_points.shape[1]
        
        super(ZGPModel, self).__init__(variational_strategy)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(
             ZRBFKernel(ard_num_dims=dim)
        )
        self.covar_module.base_kernel.lengthscale = math.sqrt(dim)
        self.num_outputs = 1
        self.likelihood = likelihood 
        
    def forward(self, x, **kwargs):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x, **kwargs)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

    def posterior(
            self, X, output_indices=None, observation_noise=False, *args, **kwargs
        ) -> GPyTorchPosterior:
            self.eval()  # make sure model is in eval mode
            # self.model.eval()
            self.likelihood.eval()
            dist = self.likelihood(self(X)) 

            return GPyTorchPosterior(mvn=dist)
    

class VanillaBOGPModel(ApproximateGP):
    def __init__(self, inducing_points, likelihood, loc: float = 1, scale: float = 2):
        variational_distribution = CholeskyVariationalDistribution(inducing_points.size(0) )
        variational_strategy = VariationalStrategy(
            self,
            inducing_points,
            variational_distribution,
            learn_inducing_locations=True
            )
        dim = inducing_points.shape[1]
        
        super(VanillaBOGPModel, self).__init__(variational_strategy)
        self.mean_module = gpytorch.means.ConstantMean()
        scaled_loc = (loc + math.log(dim) / 2)
        self.covar_module = gpytorch.kernels.ScaleKernel(
             gpytorch.kernels.RBFKernel(
                ard_num_dims=dim, 
                #lengthscale_prior=LogNormalPrior(loc=scaled_loc, scale=scale)
            )
        )
        self.covar_module.base_kernel.lengthscale = math.sqrt(dim)
        self.num_outputs = 1
        self.likelihood = likelihood 
        
    def forward(self, x, **kwargs):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x, **kwargs)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

    def posterior(
            self, X, output_indices=None, observation_noise=False, *args, **kwargs
        ) -> GPyTorchPosterior:
            self.eval()  # make sure model is in eval mode
            # self.model.eval()
            self.likelihood.eval()
            dist = self.likelihood(self(X)) 

            return GPyTorchPosterior(mvn=dist)
    
    

class ExactGPModel(BatchedMultiOutputGPyTorchModel, ExactGP):
    def __init__(self, train_inputs, train_targets, loc: float = 1, scale: float = 2):
        
        self._set_dimensions(train_X=train_inputs, train_Y=train_targets)
        train_inputs, train_targets, _ = self._transform_tensor_args(train_inputs, train_targets, None)
        likelihood = gpytorch.likelihoods.GaussianLikelihood(batch_shape=self._aug_batch_shape, noise_prior=LogNormalPrior(-6, 0.1)).cuda() 
        ExactGP.__init__(self, train_inputs, train_targets, likelihood)
        self.mean_module = gpytorch.means.ConstantMean(batch_shape=self._aug_batch_shape)
        dim = train_inputs.shape[-1]
        scaled_loc = (loc + math.log(dim) / 2)
        self.covar_module = gpytorch.kernels.RBFKernel(
                ard_num_dims=dim, 
                lengthscale_prior=LogNormalPrior(loc=scaled_loc, scale=scale),
                batch_shape=self._aug_batch_shape
        )
        self.covar_module.lengthscale = math.sqrt(dim)
        
    def forward(self, x, **kwargs):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x, **kwargs)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
    

def Z_to_X(Z: Tensor) -> Tensor:
    deterministic_fill = torch.zeros_like(Z)
    return torch.cat((Z, deterministic_fill), dim=-1)    


def check_if_z(Z_or_X: Tensor, true_dim: int) -> Tensor:
    if Z_or_X.shape[-1] == true_dim:
        X = Z_to_X(Z_or_X)
    else:
        X = Z_or_X
    return X   


class ExactHenryModel(BatchedMultiOutputGPyTorchModel, ExactGP):
    def __init__(self, train_inputs, train_targets, loc: float = 0.5, scale: float = 2):
        
        self._set_dimensions(train_X=train_inputs, train_Y=train_targets)
        train_inputs, train_targets, _ = self._transform_tensor_args(train_inputs, train_targets, None)
        likelihood = gpytorch.likelihoods.GaussianLikelihood(batch_shape=self._aug_batch_shape, noise_prior=LogNormalPrior(-6, 0.1)).cuda() 
        ExactGP.__init__(self, train_inputs, train_targets, likelihood)
        self.mean_module = gpytorch.means.ConstantMean(batch_shape=self._aug_batch_shape)
        dim = train_inputs.shape[-1]
        self.true_dim = dim // 2
        scaled_loc = (loc + math.log(self.true_dim) / 2)
        self.covar_module = ZRBFKernel(
                ard_num_dims=dim, 
                lengthscale_prior=LogNormalPrior(loc=scaled_loc, scale=scale),
                batch_shape=self._aug_batch_shape
        )
        self.covar_module.lengthscale = math.sqrt(dim)
        
    def transform_inputs(self, X: Tensor, input_transform: Module | None = None) -> Tensor:
        X = check_if_z(X, self.true_dim)
        breakpoint()
        return super().transform_inputs(X, input_transform)
    
    def forward(self, x, **kwargs):
        
        print(x.shape, self.mean_module.constant)
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x, **kwargs)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
    


class LatentHenryModel(BatchedMultiOutputGPyTorchModel, ApproximateGP):
    def __init__(self, inducing_points, likelihood, loc: float = 1, scale: float = 2):
        variational_distribution = CholeskyVariationalDistribution(inducing_points.size(0) )
        variational_strategy = LatentVariationalStrategy(
            self,
            inducing_points,
            variational_distribution,
            learn_inducing_locations=True
            )
        super(LatentHenryModel, self).__init__(variational_strategy)

        dim = variational_strategy.inducing_points.shape[-1]
        self.true_dim = dim // 2
        scaled_loc = (loc + math.log(self.true_dim) / 2)

        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(ZRBFKernel(
                ard_num_dims=dim, 
                lengthscale_prior=LogNormalPrior(loc=scaled_loc, scale=scale)
        ), LogNormalPrior(loc=0, scale=1))
        self.covar_module.base_kernel.lengthscale = math.sqrt(self.true_dim)
        self.likelihood = likelihood 
        self._num_outputs = 1

    def transform_inputs(self, X: Tensor, input_transform: Module | None = None) -> Tensor:
        """For the posterior call only!

        Args:
            X (Tensor): _description_
            input_transform (Module | None, optional): _description_. Defaults to None.

        Returns:
            Tensor: _description_
        """        
        X = check_if_z(X, self.true_dim)
        return super().transform_inputs(X, input_transform)

    def forward(self, x, **kwargs):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x, **kwargs)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


# gp model with deep kernel
class ShallowGPModelDKL(ApproximateGP):
    def __init__(self, inducing_points, likelihood, hidden_dims=(256,) ):
        feature_extractor = DenseNetwork(
            input_dim=inducing_points.size(-1),
            hidden_dims=hidden_dims).to(inducing_points.device
            )
        inducing_points = feature_extractor(inducing_points)
        variational_distribution = CholeskyVariationalDistribution(inducing_points.size(0))
        variational_strategy = VariationalStrategy(
            self,
            inducing_points,
            variational_distribution,
            learn_inducing_locations=True
            )
        super(ShallowGPModelDKL, self).__init__(variational_strategy)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
        self.num_outputs = 1 #must be one
        self.likelihood = likelihood
        self.feature_extractor = feature_extractor

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

    def __call__(self, x, *args, **kwargs):
        x = self.feature_extractor(x)
        return super().__call__(x, *args, **kwargs)

    def posterior(
            self, X, output_indices=None, observation_noise=False, *args, **kwargs
        ) -> GPyTorchPosterior:
            self.eval()  # make sure model is in eval mode 
            # self.model.eval() 
            self.likelihood.eval()
            dist = self.likelihood(self(X))

            return GPyTorchPosterior(mvn=dist)


# gp model with deep kernel
class GPModelDKL(ApproximateGP):
    def __init__(self, inducing_points, likelihood, hidden_dims=(256, 256) ):
        feature_extractor = DenseNetwork(
            input_dim=inducing_points.size(-1),
            hidden_dims=hidden_dims).to(inducing_points.device
            )
        inducing_points = feature_extractor(inducing_points)
        variational_distribution = CholeskyVariationalDistribution(inducing_points.size(0))
        variational_strategy = VariationalStrategy(
            self,
            inducing_points,
            variational_distribution,
            learn_inducing_locations=True
            )
        super(GPModelDKL, self).__init__(variational_strategy)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
        self.num_outputs = 1 #must be one
        self.likelihood = likelihood
        self.feature_extractor = feature_extractor

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

    def __call__(self, x, *args, **kwargs):
        x = self.feature_extractor(x)
        return super().__call__(x, *args, **kwargs)

    def posterior(
            self, X, output_indices=None, observation_noise=False, *args, **kwargs
        ) -> GPyTorchPosterior:
            self.eval()  # make sure model is in eval mode 
            # self.model.eval() 
            self.likelihood.eval()
            dist = self.likelihood(self(X))

            return GPyTorchPosterior(mvn=dist)


# gp model with deep kernel
class DeepGPModelDKL(ApproximateGP):
    def __init__(self, inducing_points, likelihood, hidden_dims=(256, 256, 256, 256) ):
        feature_extractor = DenseNetwork(
            input_dim=inducing_points.size(-1),
            hidden_dims=hidden_dims).to(inducing_points.device
            )
        inducing_points = feature_extractor(inducing_points)
        variational_distribution = CholeskyVariationalDistribution(inducing_points.size(0))
        variational_strategy = VariationalStrategy(
            self,
            inducing_points,
            variational_distribution,
            learn_inducing_locations=True
            )
        super(DeepGPModelDKL, self).__init__(variational_strategy)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
        self.num_outputs = 1 #must be one
        self.likelihood = likelihood
        self.feature_extractor = feature_extractor

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

    def __call__(self, x, *args, **kwargs):
        x = self.feature_extractor(x)
        return super().__call__(x, *args, **kwargs)

    def posterior(
            self, X, output_indices=None, observation_noise=False, *args, **kwargs
        ) -> GPyTorchPosterior:
            self.eval()  # make sure model is in eval mode 
            # self.model.eval() 
            self.likelihood.eval()
            dist = self.likelihood(self(X))

            return GPyTorchPosterior(mvn=dist)


# gp model with deep kernel
class DeeperGPModelDKL(ApproximateGP):
    def __init__(self, inducing_points, likelihood, hidden_dims=(256, 256, 256, 256, 256, 256, 256, 256) ):
        feature_extractor = DenseNetwork(
            input_dim=inducing_points.size(-1),
            hidden_dims=hidden_dims).to(inducing_points.device
            )
        inducing_points = feature_extractor(inducing_points)
        variational_distribution = CholeskyVariationalDistribution(inducing_points.size(0))
        variational_strategy = VariationalStrategy(
            self,
            inducing_points,
            variational_distribution,
            learn_inducing_locations=True
            )
        super(DeeperGPModelDKL, self).__init__(variational_strategy)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
        self.num_outputs = 1 #must be one
        self.likelihood = likelihood
        self.feature_extractor = feature_extractor

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

    def __call__(self, x, *args, **kwargs):
        x = self.feature_extractor(x)
        return super().__call__(x, *args, **kwargs)

    def posterior(
            self, X, output_indices=None, observation_noise=False, *args, **kwargs
        ) -> GPyTorchPosterior:
            self.eval()  # make sure model is in eval mode 
            # self.model.eval() 
            self.likelihood.eval()
            dist = self.likelihood(self(X))

            return GPyTorchPosterior(mvn=dist)


# gp model with deep kernel
class ZGPModelDKL(ApproximateGP):
    def __init__(self, inducing_points, likelihood, hidden_dims=(256, 256) ):
        feature_extractor = DenseNetwork(
            input_dim=inducing_points.size(-1),
            hidden_dims=hidden_dims).to(inducing_points.device
            )
        inducing_points = feature_extractor(inducing_points)
        variational_distribution = CholeskyVariationalDistribution(inducing_points.size(0))
        variational_strategy = VariationalStrategy(
            self,
            inducing_points,
            variational_distribution,
            learn_inducing_locations=True
            )
        super(ZGPModelDKL, self).__init__(variational_strategy)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(ZRBFKernel())
        self.num_outputs = 1 #must be one
        self.likelihood = likelihood
        self.feature_extractor = feature_extractor

    def forward(self, x, **kwargs):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x, **kwargs)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

    def __call__(self, x, *args, **kwargs):
        x = self.feature_extractor(x)
        return super().__call__(x, *args, **kwargs)

    def posterior(
            self, X, output_indices=None, observation_noise=False, *args, **kwargs
        ) -> GPyTorchPosterior:
            self.eval()  # make sure model is in eval mode 
            # self.model.eval() 
            self.likelihood.eval()
            dist = self.likelihood(self(X))

            return GPyTorchPosterior(mvn=dist)
