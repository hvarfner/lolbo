from typing import Tuple, Any
import math
import torch
from dataclasses import dataclass
from torch.quasirandom import SobolEngine
from botorch.acquisition import qExpectedImprovement, qLogExpectedImprovement
from botorch.optim import optimize_acqf
from .approximate_gp import *
from botorch.generation import MaxPosteriorSampling 
from lolbo.utils.bo_utils.ppgpr import (
    Z_to_X, 
    GPModelDKL,
    DeepGPModelDKL, 
    ShallowGPModelDKL
)
from botorch.sampling.pathwise import (
    SamplePath, 
    draw_matheron_paths
)
from botorch.models.model import Model
from botorch.utils.transforms import unnormalize
from torch.quasirandom import SobolEngine



def get_optimal_samples(
    model: Model,
    bounds: Tensor,
    num_optima: int,
    raw_samples: int = 1024,
    num_restarts: int = 20,
    maximize: bool = True,
) -> Tuple[Tensor, Tensor]:
    """Draws sample paths from the posterior and maximizes the samples using GD.

    Args:
        model (Model): The model from which samples are drawn.
        bounds: (Tensor): Bounds of the search space. If the model inputs are
            normalized, the bounds should be normalized as well.
        num_optima (int): The number of paths to be drawn and optimized.
        raw_samples (int, optional): The number of candidates randomly sample.
            Defaults to 1024.
        num_restarts (int, optional): The number of candidates to do gradient-based
            optimization on. Defaults to 20.
        maximize: Whether to maximize or minimize the samples.
    Returns:
        Tuple[Tensor, Tensor]: The optimal input locations and corresponding
        outputs, x* and f*.

    """
    paths = draw_matheron_paths(model, sample_shape=torch.Size([num_optima]))
    optimal_inputs, optimal_outputs = optimize_posterior_samples(
        paths,
        bounds=bounds,
        raw_samples=raw_samples,
        num_restarts=num_restarts,
        maximize=maximize,
    )
    return optimal_inputs, optimal_outputs


def optimize_posterior_samples(
    paths: SamplePath,
    bounds: Tensor,
    candidates: Optional[Tensor] = None,
    raw_samples: Optional[int] = 1024,
    num_restarts: int = 20,
    maximize: bool = True,
    **kwargs: Any,
) -> Tuple[Tensor, Tensor]:
    r"""Cheaply maximizes posterior samples by random querying followed by vanilla
    gradient descent on the best num_restarts points.

    Args:
        paths: Random Fourier Feature-based sample paths from the GP
        bounds: The bounds on the search space.
        candidates: A priori good candidates (typically previous design points)
            which acts as extra initial guesses for the optimization routine.
        raw_samples: The number of samples with which to query the samples initially.
        num_restarts: The number of points selected for gradient-based optimization.
        maximize: Boolean indicating whether to maimize or minimize

    Returns:
        A two-element tuple containing:
            - X_opt: A `num_optima x [batch_size] x d`-dim tensor of optimal inputs x*.
            - f_opt: A `num_optima x [batch_size] x 1`-dim tensor of optimal outputs f*.
    """
    if maximize:

        def path_func(x):
            return paths(x)

    else:

        def path_func(x):
            return -paths(x)

    candidate_set = unnormalize(
        SobolEngine(dimension=bounds.shape[1], scramble=True).draw(raw_samples).to(bounds), bounds
    )

    # queries all samples on all candidates - output shape
    # raw_samples * num_optima * num_models
    candidate_queries = path_func(candidate_set)
    argtop_k = torch.topk(candidate_queries, num_restarts, dim=-1).indices
    X_top_k = candidate_set[argtop_k, :]

    # to avoid circular import, the import occurs here
    from botorch.generation.gen import gen_candidates_torch

    X_top_k, f_top_k = gen_candidates_torch(
        X_top_k, path_func, lower_bounds=bounds[0], upper_bounds=bounds[1], **kwargs
    )
    f_opt, arg_opt = f_top_k.max(dim=-1, keepdim=True)

    # For each sample (and possibly for every model in the batch of models), this
    # retrieves the argmax. We flatten, pick out the indices and then reshape to
    # the original batch shapes (so instead of pickig out the argmax of a
    # (3, 7, num_restarts, D)) along the num_restarts dim, we pick it out of a
    # (21  , num_restarts, D)
    final_shape = candidate_queries.shape[:-1]
    X_opt = X_top_k.reshape(final_shape.numel(), num_restarts, -1)[
        torch.arange(final_shape.numel()), arg_opt.flatten()
    ].reshape(*final_shape, -1)
    if not maximize:
        f_opt = -f_opt
    return X_opt, f_opt


@dataclass
class TurboState:
    dim: int
    batch_size: int
    length: float = 0.8 
    length_min: float = 0.5 ** 7
    length_max: float = 1.6
    failure_counter: int = 0
    failure_tolerance: int = 32 
    success_counter: int = 0
    success_tolerance: int = 10 
    best_value: float = -float("inf")
    restart_triggered: bool = False

    # def __post_init__(self):
    #     self.failure_tolerance = math.ceil(
    #         max([4.0 / self.batch_size, float(self.dim ) / self.batch_size])
    #     )

def update_state(state, Y_next):
    if max(Y_next) > state.best_value + 1e-3 * math.fabs(state.best_value):
        state.success_counter += 1
        state.failure_counter = 0
    else:
        state.success_counter = 0
        state.failure_counter += 1

    if state.success_counter == state.success_tolerance:  # Expand trust region
        state.length = min(2.0 * state.length, state.length_max)
        state.success_counter = 0
    elif state.failure_counter == state.failure_tolerance:  # Shrink trust region
        state.length /= 2.0
        state.failure_counter = 0

    state.best_value = max(state.best_value, max(Y_next).item())
    if state.length < state.length_min:
        state.restart_triggered = True
    return state


def generate_batch(
    state,
    model,  # GP model
    X,  # Evaluated points on the domain [0, 1]^d
    Y,  # Function values
    batch_size,
    n_candidates=None,  # Number of candidates for Thompson sampling 
    num_restarts=1,
    raw_samples=256,
    ls_tr_ratio=16,
    acqf="ts",  # "ei" or "ts"
    dtype=torch.float32,
    device=torch.device('cuda'),
):
    assert acqf in ("ts", "ana_ts", "ei", "unmasked_ts")
    assert torch.all(torch.isfinite(Y))
    if n_candidates is None: n_candidates = min(768, max(1000, 200 * X.shape[-1]))
    
    if hasattr(model, "true_dim"):
        tr_dim = model.true_dim
    
    else:
        tr_dim = X.shape[-1]
    x_center = X[Y.argmax(), :tr_dim].clone()  
    weights = torch.ones_like(x_center)*8 # less than 4 stdevs on either side max 
    if isinstance(model, (GPModelDKL, DeepGPModelDKL, ShallowGPModelDKL)):
        tr_lb = x_center - weights * state.length / 2.0
        tr_ub = x_center + weights * state.length / 2.0 
    else:
        try:
            ls = model.covar_module.base_kernel.lengthscale[..., :tr_dim] 
        except AttributeError:
            ls = model.covar_module.lengthscale[..., :tr_dim]
        tr_lb = x_center.to(ls) - ls / ls_tr_ratio # The default size of the 
        tr_ub = x_center.to(ls) + ls / ls_tr_ratio # The default size of the  
        #if hasattr(model, "true_dim"):
        #    tr_lb = torch.cat((tr_lb, torch.zeros_like(tr_lb)), dim=-1)
        #    tr_ub = torch.cat((tr_ub, torch.zeros_like(tr_ub)), dim=-1)
        
    if acqf == "ei":
        ei = qLogExpectedImprovement(model.cuda(), Y.max().cuda()) 
        X_next, _ = optimize_acqf(
            ei,
            bounds=torch.stack([tr_lb, tr_ub]).squeeze(1).cuda(),
            q=batch_size, 
            num_restarts=num_restarts,
            raw_samples=raw_samples, 
            options={"batch_limit": 32}
        )

    elif (acqf == "ts") or (acqf == "unmasked_ts"):
        tr_lb = tr_lb.cuda()
        tr_ub = tr_ub.cuda() 
        sobol = SobolEngine(tr_dim, scramble=True) 
        pert = sobol.draw(n_candidates).to(dtype=dtype).cuda()
        pert = tr_lb + (tr_ub - tr_lb) * pert
        tr_lb = tr_lb.cuda()
        tr_ub = tr_ub.cuda() 

        if acqf == "unmasked_ts":
            X_cand = pert
        else:
            # Create a perturbation mask 
            prob_perturb = min(20.0 / tr_dim, 1.0)
            mask = (torch.rand(n_candidates, tr_dim, dtype=dtype, device=device)<= prob_perturb)
            ind = torch.where(mask.sum(dim=1) == 0)[0]
            mask[ind, torch.randint(0, tr_dim - 1, size=(len(ind),), device=device)] = 1
            mask = mask.cuda()

            # Create candidate points from the perturbations and the mask
            X_cand = x_center.expand(n_candidates, tr_dim).clone()
            X_cand = X_cand.cuda()
            X_cand[mask] = pert[mask]
        # Sample on the candidate points 
        thompson_sampling = MaxPosteriorSampling(model=model, replacement=False ) 
        X_next = thompson_sampling(X_cand.cuda(), num_samples=batch_size )

    elif acqf == "ana_ts":
        bounds = torch.stack([tr_lb, tr_ub]).squeeze(1)
        X_cand, f_val = get_optimal_samples(model=model, bounds=bounds, num_optima=batch_size, raw_samples=4096, num_restarts=10)
    #if hasattr(model, "true_dim"):
    #    X_next = Z_to_X(X_next)

    return X_next
