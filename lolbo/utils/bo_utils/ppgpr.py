# ppgpr
import math
from .base import DenseNetwork
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
#model.covar_module.base_kernel.lengthscale
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
    

class VanillaBOZGPModel(ApproximateGP):
    def __init__(self, inducing_points, likelihood, loc: float = 2 ** 0.5, scale: float = 2):
        variational_distribution = CholeskyVariationalDistribution(inducing_points.size(0) )
        variational_strategy = LatentVariationalStrategy(
            self,
            inducing_points,
            variational_distribution,
            learn_inducing_locations=True
            )
        dim = inducing_points.shape[1]
        
        super(VanillaBOZGPModel, self).__init__(variational_strategy)
        self.mean_module = gpytorch.means.ConstantMean()
        scaled_loc = (loc + math.log(dim) / 2) * 2
        self.covar_module = gpytorch.kernels.ScaleKernel(
             ZRBFKernel(
                ard_num_dims=dim, 
                lengthscale_prior=LogNormalPrior(loc=scaled_loc, scale=scale)
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
  

class UnwhitenedVanillaBOGPModel(ApproximateGP):
    def __init__(self, inducing_points, likelihood, loc: float = 2 ** 0.5, scale: float = 2):
        variational_distribution = CholeskyVariationalDistribution(inducing_points.size(0) )
        variational_strategy = UnwhitenedVariationalStrategy(
            self,
            inducing_points,
            variational_distribution,
            learn_inducing_locations=True
        )
        dim = inducing_points.shape[1]
        
        super(UnwhitenedVanillaBOGPModel, self).__init__(variational_strategy)
        self.mean_module = gpytorch.means.ConstantMean()
        scaled_loc = (loc + math.log(dim) / 2) * 2
        self.covar_module = gpytorch.kernels.ScaleKernel(
             gpytorch.kernels.RBFKernel(
                ard_num_dims=dim, 
                lengthscale_prior=LogNormalPrior(loc=scaled_loc, scale=scale)
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
    

class UnwhitenedVanillaBOZGPModel(ApproximateGP):
    def __init__(self, inducing_points, likelihood, loc: float = 2 ** 0.5, scale: float = 2):
        variational_distribution = CholeskyVariationalDistribution(inducing_points.size(0) )
        variational_strategy = UnwhitenedVariationalStrategy(
            self,
            inducing_points,
            variational_distribution,
            learn_inducing_locations=True
        )
        dim = inducing_points.shape[1]
        
        super(UnwhitenedVanillaBOZGPModel, self).__init__(variational_strategy)
        self.mean_module = gpytorch.means.ConstantMean()
        scaled_loc = (loc + math.log(dim) / 2) * 2
        self.covar_module = gpytorch.kernels.ScaleKernel(
             ZRBFKernel(
                ard_num_dims=dim, 
                lengthscale_prior=LogNormalPrior(loc=scaled_loc, scale=scale)
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
