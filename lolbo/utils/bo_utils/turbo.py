import math
import torch
from dataclasses import dataclass
from torch.quasirandom import SobolEngine
from botorch.acquisition import qExpectedImprovement, qLogExpectedImprovement
from botorch.optim import optimize_acqf
from .approximate_gp import *
from botorch.generation import MaxPosteriorSampling 
from lolbo.utils.bo_utils.ppgpr import Z_to_X, GPModelDKL

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
    num_restarts=4,
    raw_samples=512,
    ls_tr_ratio=4,
    acqf="ts",  # "ei" or "ts"
    dtype=torch.float32,
    device=torch.device('cuda'),
):
    assert acqf in ("ts", "ei")
    assert torch.all(torch.isfinite(Y))
    if n_candidates is None: n_candidates = min(500, max(1000, 200 * X.shape[-1]))
    if hasattr(model, "true_dim"):
        tr_dim = model.true_dim
    else:
        tr_dim = X.shape[-1]
    x_center = X[Y.argmax(), :tr_dim].clone()  
    weights = torch.ones_like(x_center)*8 # less than 4 stdevs on either side max 
    if isinstance(model, GPModelDKL):
        tr_lb = x_center - weights * state.length / 2.0
        tr_ub = x_center + weights * state.length / 2.0 
    else:
        ls = model.covar_module.lengthscale[..., :tr_dim]
        tr_lb = x_center.to(ls) - ls / ls_tr_ratio # The default size of the 
        tr_ub = x_center.to(ls) + ls / ls_tr_ratio # The default size of the  

    if acqf == "ei":
        ei = qLogExpectedImprovement(model.cuda(), Y.max().cuda() ) 
        X_next, _ = optimize_acqf(ei,bounds=torch.stack([tr_lb, tr_ub]).squeeze(1).cuda(),q=batch_size, num_restarts=num_restarts,raw_samples=raw_samples,)

    if acqf == "ts":
        tr_lb = tr_lb.cuda()
        tr_ub = tr_ub.cuda() 
        sobol = SobolEngine(tr_dim, scramble=True) 
        pert = sobol.draw(n_candidates).to(dtype=dtype).cuda()
        pert = tr_lb + (tr_ub - tr_lb) * pert
        tr_lb = tr_lb.cuda()
        tr_ub = tr_ub.cuda() 
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

    #if hasattr(model, "true_dim"):
    #    X_next = Z_to_X(X_next)

    return X_next
