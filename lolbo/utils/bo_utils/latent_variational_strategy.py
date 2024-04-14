import warnings
from typing import Any, Dict, Iterable, Optional, Tuple

import torch
from torch.nn import Module
from linear_operator import to_dense
from linear_operator.operators import (
    CholLinearOperator,
    DiagLinearOperator,
    LinearOperator,
    MatmulLinearOperator,
    RootLinearOperator,
    SumLinearOperator,
    TriangularLinearOperator,
)
from linear_operator.utils.cholesky import psd_safe_cholesky
from linear_operator.utils.errors import NotPSDError
from torch import Tensor

#from lolbo.utils.variational_mods._variational_strategy import _VariationalStrategy
from gpytorch.variational.cholesky_variational_distribution import CholeskyVariationalDistribution

from gpytorch.distributions import MultivariateNormal
from gpytorch.models import ApproximateGP
from gpytorch.settings import _linalg_dtype_cholesky, trace_mode
from gpytorch.utils.errors import CachingError
from gpytorch.utils.memoize import cached, clear_cache_hook, pop_from_cache_ignore_args
from gpytorch.utils.warnings import OldVersionWarning
from gpytorch.variational import _VariationalDistribution


from gpytorch.variational.unwhitened_variational_strategy import (
    UnwhitenedVariationalStrategy
)
from gpytorch.variational.variational_strategy import _ensure_updated_strategy_flag_set

class LatentVariationalStrategy(UnwhitenedVariationalStrategy):

        # self._raw_inducing_points <- the actual inducing points
        # self.inducing points = 2D
        #1. Override init: call the inducing points self._inducing_points 
        # or self.raw_inducing_point
        #2. define something like
        # @property def inducing_points(self): 
        # padded_inds = torch.cat(self.raw_inducing_points, torch.zeros(...)) return padded_inds 
        # model(pad_zeros(vae(x))) 
        # class CarlHenryVariationalStrategy(VariationalStrategy): 
        # model(x, arbitrary_kwarg=blah) 
    def __init__(
        self,
        model: ApproximateGP,
        inducing_points: Tensor,
        variational_distribution: _VariationalDistribution,
        learn_inducing_locations: bool = True,
        jitter_val: Optional[float] = None,
    ):
        Module.__init__(self)
        object.__setattr__(self, "model", model)
        self._jitter_val = jitter_val
        self.register_buffer("updated_strategy", torch.tensor(True))
        self._register_load_state_dict_pre_hook(_ensure_updated_strategy_flag_set)
        self.has_fantasy_strategy = True

        # Variational distribution
        self._variational_distribution = variational_distribution
        self.register_buffer("variational_params_initialized", torch.tensor(0))

        if learn_inducing_locations:
            self.register_parameter(name="_inducing_points", parameter=torch.nn.Parameter(inducing_points))
        else:
            self.register_buffer("_inducing_points", inducing_points)
        self.register_buffer("_fill", torch.zeros_like(inducing_points))
        print("CHECK THE SHAPE OF THE CH STRAT _FILL AND _IND")
        breakpoint()

    @property
    def inducing_points(self):
        return torch.cat((self._inducing_points, self._fill), dim=-1)