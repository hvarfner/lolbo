import numpy as np
import torch
import gpytorch
from gpytorch.models import ExactGP
import math
from gpytorch.mlls import (
    PredictiveLogLikelihood, 
    ExactMarginalLogLikelihood
)
from botorch.fit import fit_gpytorch_mll
            
from lolbo.utils.bo_utils.turbo import TurboState, update_state, generate_batch
from lolbo.utils.utils import (
    update_models_end_to_end,
    update_surr_model,
    update_surr_model_sampled_z,
    update_exact_surr_model,
    update_exact_end_to_end,
    _get_predictions,
)
from lolbo.utils.bo_utils.ppgpr import GPModelDKL, ExactGPModel, ExactHenryModel, Z_to_X
from lolbo.utils.bo_utils.registry import get_model
from botorch.utils.transforms import standardize


class LOLBOState:

    def __init__(
        self,
        objective,
        train_x,
        train_y,
        train_z,
        k,
        minimize,
        num_update_epochs,
        init_n_epochs,
        vae_learning_rate,
        gp_learning_rate,
        bsz=10,
        acq_func='ts',
        model_class='dkl',
        verbose=True,
        normalize_y: bool = True,
        z_as_dist: bool = False,
        train_on_z_mean: bool = False,
        sample_z_e2e: bool = True,
        ):
        self.model_class = get_model(model_class)
        self.objective          = objective         # objective with vae for particular task
        self.train_x            = train_x           # initial train x data
        self.orig_train_y       = train_y           # initial train y data
        self.train_z            = train_z           # initial train z data
        self.minimize           = minimize          # if True we want to minimize the objective, otherwise we assume we want to maximize the objective
        self.k                  = k                 # track and update on top k scoring points found
        self.num_update_epochs  = num_update_epochs # num epochs update models
        self.init_n_epochs      = init_n_epochs     # num epochs train surr model on initial data
        self.vae_learning_rte   = vae_learning_rate  # lr to use for model updates
        self.gp_learning_rte    = gp_learning_rate
        self.bsz                = bsz               # acquisition batch size
        self.acq_func           = acq_func          # acquisition function (Expected Improvement (ei) or Thompson Sampling (ts))
        self.verbose            = verbose
        self.normalize_y        = normalize_y
        self.z_as_dist          = z_as_dist
        self.train_on_z_mean    = train_on_z_mean
        self.sample_z_e2e       = sample_z_e2e
        self.z_acqtype = ["i"] * len(self.orig_train_y)
        assert acq_func in ["ei", "ts", "logei", "ana_ts", "unmasked_ts"]
        self.duplicates = []
        self.progress_fails_since_last_e2e = 0
        self.tot_num_e2e_updates = 0
        self.best_score_seen = torch.max(self.orig_train_y)
        self.best_x_seen = train_x[torch.argmax(self.orig_train_y)]
        self.initial_model_training_complete = False # initial training of surrogate model uses all data for more epochs
        self.new_best_found = False
            
        self.initialize_surrogate_model()
        if isinstance(self.model, ExactGP):
            self.update_surrogate_model()
        self.initialize_tr_state()
        self.initialize_xs_to_scores_dict()

    def initialize_xs_to_scores_dict(self,):
        # put initial xs and ys in dict to be tracked by objective
        init_xs_to_scores_dict = {}
        for idx, x in enumerate(self.train_x):
            init_xs_to_scores_dict[x] = self.orig_train_y.squeeze()[idx].item()
        self.objective.xs_to_scores_dict = init_xs_to_scores_dict

    def initialize_tr_state(self):
        # initialize turbo trust region state
        self.tr_state = TurboState( # initialize turbo state
            dim=self.train_z.shape[-1],
            batch_size=self.bsz, 
            best_value=(torch.max(self.orig_train_y).item() - self.ymean) / self.ystd 
            )

        return self

    def initialize_surrogate_model(self ):
        if self.model_class is ExactHenryModel:
            train_x, train_y, train_z = self.get_training_data(k=-1, renormalize=True)
            z_mu, z_sigma, batch_losses  = _get_predictions(None, train_x, obj=self.objective, return_loss=True)
            z_cat = torch.cat((z_mu, z_sigma), dim=-1) 
            self.model = self.model_class(
                z_cat, 
                train_y.cuda(), 
            ).cuda()
            self.mll = ExactMarginalLogLikelihood(self.model.likelihood, self.model)
        
        elif issubclass(self.model_class, ExactGP):
            train_x, train_y, train_z = self.get_training_data(self.k, renormalize=True)
            self.model = self.model_class(
                train_z.cuda(), 
                train_y.cuda(), 
            ).cuda()
            self.mll = ExactMarginalLogLikelihood(self.model.likelihood, self.model)
        
        else:
            train_x, train_y, train_z = self.get_training_data(min(self.train_z.shape[0], 1024), renormalize=True)
            n_pts = min(self.train_z.shape[0], 1024)
            likelihood = gpytorch.likelihoods.GaussianLikelihood()
            self.model = self.model_class(train_z[:n_pts, :].cuda(), likelihood=likelihood ).cuda()
            self.mll = PredictiveLogLikelihood(self.model.likelihood, self.model, num_data=self.train_z.size(-2))
            self.model = self.model.eval() 
            self.model = self.model.cuda()

        return self


    def update_next(self, z_next_, y_next_, x_next_, duplicates: np.array, acquisition=False):
        '''Add new points (z_next, y_next, x_next) to train data
            and update progress (top k scores found so far)
            and update trust region state
        '''

        z_next_ = z_next_.detach().cpu()[~duplicates] 
        y_next_ = y_next_.detach().cpu()[~duplicates]
        x_next_ = np.array(x_next_)[~duplicates]
        if (~duplicates).sum() > 0:
            if len(y_next_.shape) > 1:
                y_next_ = y_next_.squeeze() 
            if len(z_next_.shape) == 1:
                z_next_ = z_next_.unsqueeze(0)
            progress = False
            self.train_x.extend(x_next_)
            
            if y_next_.max().item() > self.best_score_seen:
                self.progress_fails_since_last_e2e = 0
                progress = True
                self.best_score_seen = y_next_.max().item() #update best
                self.best_x_seen = x_next_[y_next_.argmax().item()]
                self.new_best_found = True
            if (not progress) and acquisition: # if no progress msde, increment progress fails
                self.progress_fails_since_last_e2e += 1
        
        y_next_ = y_next_.unsqueeze(-1) 

        if y_next_.shape[0] == 0:
            self.tr_state = update_state(state=self.tr_state, Y_next=torch.Tensor([-100000]))
        else:
            self.tr_state = update_state(state=self.tr_state, Y_next=y_next_)
        self.train_z = torch.cat((self.train_z, z_next_), dim=-2)
        for z_idx in range(len(z_next_)):
            if acquisition:
                self.z_acqtype.append("a")
            else:
                self.z_acqtype.append("r")

        self.orig_train_y = torch.cat((self.orig_train_y, y_next_), dim=-2)
        return self

    def standardize_current_train(self, y: torch.Tensor, renormalize: bool = False):
        if renormalize:
            self.ystd = y.std()
            self.ymean = y.mean()
        
        return (y - self.ymean) / self.ystd
    

    def get_training_data(self, k: int, recent: int = 0, renormalize: bool = False):
        ''' get top k x, y, and zs'''
        # track top k scores found
        if k == -1:
            k = len(self.train_x)
        top_k_scores, top_k_idxs = torch.topk(self.orig_train_y.squeeze(), min(k, len(self.orig_train_y)))
        top_k_xs = [self.train_x[i] for i in top_k_idxs]
        top_k_zs = self.train_z[top_k_idxs]
        
        if recent > 0:
            recent_x = self.train_x[-recent:]
            recent_scores = self.orig_train_y[-recent:].flatten()
            recent_z = self.train_z[-recent:]
            x = top_k_xs + recent_x
            y = torch.cat((top_k_scores, recent_scores))
            z = torch.cat((top_k_zs, recent_z))
            
        else:            
            x = top_k_xs
            y = top_k_scores.to(top_k_zs).unsqueeze(-1)
            z = top_k_zs
        y = self.standardize_current_train(y, renormalize=renormalize)
        
        return x, y.to(z), z
        
        return 
    

    def update_surrogate_model(self): 
        if not self.initial_model_training_complete:
            # first time training surr model --> train on all data
            n_epochs = self.init_n_epochs
        else:
            n_epochs = self.num_update_epochs
        
        if isinstance(self.model, ExactGP):
            if not self.initial_model_training_complete:
                print("initial")
                self.initialize_surrogate_model()
                fit_gpytorch_mll(self.mll)

            else:
                print("update")
                update_exact_surr_model(
                    self.model.train_inputs[0],
                    self.model.train_targets,
                    self.objective,
                    self.model,
                    self.mll,
                    self.gp_learning_rte,
                    n_epochs,
                )
        else:
            train_x, train_y, train_z = self.get_training_data(k=0, recent=self.bsz, renormalize=True)
            
            self.model = update_surr_model(
                self.model,
                self.mll,
                self.gp_learning_rte,
                train_z,
                train_y,
                n_epochs
            )

        self.initial_model_training_complete = True

        return self

    def update_models_e2e(self):
        '''Finetune VAE end to end with surrogate model'''
        self.progress_fails_since_last_e2e = 0

        train_x, train_y, train_z = self.get_training_data(k=self.k, recent=self.bsz)
        if isinstance(self.model, ExactGP):
            self.initialize_surrogate_model()
            fit_gpytorch_mll(self.mll)
            self.objective, self.model = update_exact_end_to_end(
            train_x,
            train_y,
            self.objective,
            self.model,
            self.mll,
            self.vae_learning_rte,
            self.gp_learning_rte,
            self.num_update_epochs,
            )
        else:
            self.objective, self.model = update_models_end_to_end(
                train_x,
                train_y,
                self.objective,
                self.model,
                self.mll,
                self.vae_learning_rte,
                self.gp_learning_rte,
                self.num_update_epochs,
                sample_z_e2e=self.sample_z_e2e
            )
        self.tot_num_e2e_updates += 1

        return self


    def recenter(self):
        '''Pass SELFIES strings back through
            VAE to find new locations in the
            new fine-tuned latent space
        '''
        if isinstance(self.model, ExactHenryModel):
            return self
        
        self.objective.vae.eval()
        self.model.train()
        optimizer1 = torch.optim.Adam([{'params': self.model.parameters(),'lr': self.gp_learning_rte} ])
        train_x, _ , _ = self.get_training_data(k=self.k, recent=self.bsz)

        max_string_len = len(max(train_x, key=len))
        # max batch size smaller to avoid memory limit 
        #   with longer strings (more tokens) 
        bsz = max(1, int(2560/max_string_len))
        num_batches = math.ceil(len(train_x) / bsz) 
        for batch_ix in range(num_batches):
            start_idx, stop_idx = batch_ix*bsz, (batch_ix+1)*bsz
            batch_list = train_x[start_idx:stop_idx] 
            z, _ = self.objective.vae_forward(batch_list)
            out_dict = self.objective(z)
            scores_arr = out_dict['scores'] 
            valid_zs = out_dict['valid_zs']
            selfies_list = out_dict['decoded_xs']
            duplicates = out_dict['duplicates']
            if len(scores_arr) > 0: # if some valid scores
                scores_arr = torch.from_numpy(scores_arr)
                if self.minimize:
                    scores_arr = scores_arr * -1
                #pred = self.model(valid_zs)
                #loss = -self.mll(pred, (scores_arr  - self.ymean) / self.ystd).cuda())
                #optimizer1.zero_grad()
                #loss.backward()
                #torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                #optimizer1.step() 
                with torch.no_grad(): 
                    valid_zs = valid_zs.detach().cpu()
                    if hasattr(self.model, "true_dim"):
                        valid_zs = Z_to_X(valid_zs)
                    print("recenter duplicates", duplicates)
                    self.update_next(valid_zs, scores_arr, selfies_list, duplicates=duplicates)
        torch.cuda.empty_cache()
        self.model.eval() 

        return self


    def acquisition(self):
        '''Generate new candidate points, 
        evaluate them, and update data
        '''
        # 1. Generate a batch of candidates in 
        #   trust region using surrogate model
        _, train_y, train_z = self.get_training_data(k=self.k)
        z_next = generate_batch(
            state=self.tr_state,
            model=self.model,
            X=train_z,
            Y=train_y,
            batch_size=self.bsz, 
            acqf=self.acq_func,
        )
        # 2. Evaluate the batch of candidates by calling oracle
        with torch.no_grad():
            out_dict = self.objective(z_next)
            z_next = out_dict['valid_zs']
            y_next = out_dict['scores']
            x_next = out_dict['decoded_xs']     
            duplicates = out_dict['duplicates']
            if self.minimize:
                y_next = y_next * -1
        # 3. Add new evaluated points to dataset (update_next)
        
        if hasattr(self.model, "true_dim"):
            z_next = Z_to_X(z_next)
        if len(y_next) != 0:
            y_next = torch.from_numpy(y_next).float()
            self.update_next(
                z_next,
                y_next,
                x_next,
                duplicates=duplicates,
                acquisition=True
            )
        else:
            self.progress_fails_since_last_e2e += 1
            if self.verbose:
                print("GOT NO VALID Y_NEXT TO UPDATE DATA, RERUNNING ACQUISITOIN...")

        self.duplicates.append(duplicates.tolist())