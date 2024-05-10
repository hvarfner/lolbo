import math
from typing import Any
import torch

torch.cuda.empty_cache()
from torch import Tensor
import random
import numpy as np
import fire
import warnings
warnings.filterwarnings('ignore')
from rdkit import RDLogger
import pandas as pd
RDLogger.DisableLog('rdApp.*')
import os
os.environ["WANDB_SILENT"] = "True"
from lolbo.lolbo import LOLBOState
from lolbo.latent_space_objective import LatentSpaceObjective
from lolbo.utils.pred_utils import batchable, plot_predictions
try:
    import wandb
    WANDB_IMPORTED_SUCCESSFULLY = True
except ModuleNotFoundError:
    WANDB_IMPORTED_SUCCESSFULLY = False



class Optimize(object):
    """
    Run LOLBO Optimization
    Args:
        task_id: String id for optimization task, by default the wandb project name will be f'optimize-{task_id}'
        seed: Random seed to be set. If None, no particular random seed is set
        track_with_wandb: if True, run progress will be tracked using Weights and Biases API
        wandb_entity: Username for your wandb account (valid username necessary iff track_with_wandb is True)
        wandb_project_name: Name of wandb project where results will be logged (if no name is specified, default project name will be f"optimimze-{self.task_id}")
        minimize: If True we want to minimize the objective, otherwise we assume we want to maximize the objective
        max_n_oracle_calls: Max number of oracle calls allowed (budget). Optimization run terminates when this budget is exceeded
        learning_rte: Learning rate for model updates
        acq_func: Acquisition function, must be either ei or ts (ei-->Expected Imporvement, ts-->Thompson Sampling)
        bsz: Acquisition batch size
        num_initialization_points: Number evaluated data points used to optimization initialize run
        init_n_update_epochs: Number of epochs to train the surrogate model for on initial data before optimization begins
        num_update_epochs: Number of epochs to update the model(s) for on each optimization step
        e2e_freq: Number of optimization steps before we update the models end to end (end to end update frequency)
        update_e2e: If True, we update the models end to end (we run LOLBO). If False, we never update end to end (we run TuRBO)
        k: We keep track of and update end to end on the top k points found during optimization
        verbose: If True, we print out updates such as best score found, number of oracle calls made, etc. 
    """
    def __init__(
        self,
        task_id: str,
        seed: int=42,
        track_with_wandb: bool=False,
        wandb_entity: str="",
        wandb_project_name: str="",
        minimize: bool=False,
        num_iter: int=100_000,
        vae_learning_rate: float=0.001,
        gp_learning_rate: float=0.001,
        acq_func: str="ts",
        model: str="dkl",
        bsz: int=10,
        num_init: int=10_000,
        init_n_update_epochs: int=20,
        num_update_epochs: int=2,
        e2e_freq: int=10,
        update_e2e: bool=True,          
        k: int=1_000,
        verbose: bool=True,
        experiment_name: str = "test",
        z_as_dist: bool = False,
        normalize_y: bool = True,
        train_on_z_mean: bool = False,
        sample_z_e2e: bool = True,
        sample_scale: float = 1.0,
        load_init: bool = False,
        save_moments: bool = False
    ):
        # add all local args to method args dict to be logged by wandb
        self.method_args = {}
        self.method_args['init'] = locals()
        del self.method_args['init']['self']
        self.seed = seed
        self.track_with_wandb = track_with_wandb
        self.wandb_entity = wandb_entity 
        self.task_id = task_id
        self.max_n_oracle_calls = num_iter
        self.verbose = verbose
        self.num_initialization_points = num_init
        self.e2e_freq = e2e_freq
        self.model = model
        self.update_e2e = update_e2e 
        self.experiment_name = experiment_name
        self.train_on_z_mean = train_on_z_mean
        self.z_as_dist = z_as_dist
        self.sample_scale = sample_scale
        self.save_moments = save_moments
        self.set_seed()
        if wandb_project_name: # if project name specified
            self.wandb_project_name = wandb_project_name
        else: # otherwise use defualt
            self.wandb_project_name = f"optimize-{self.task_id}"
        if not WANDB_IMPORTED_SUCCESSFULLY:
            assert not self.track_with_wandb, "Failed to import wandb, to track with wandb, try pip install wandb"
        if self.track_with_wandb:
            assert self.wandb_entity, "Must specify a valid wandb account username (wandb_entity) to run with wandb tracking"
        
        # initialize train data for particular task
        #   must define self.init_train_x, self.init_train_y, and self.init_train_z

        # initialize latent space objective (self.objective) for particular task
        self.initialize_objective()
        if load_init:
            self.load_train_data()
        else:
            self.sample_train_data()
        # initialize lolbo state
        assert isinstance(self.objective, LatentSpaceObjective), "self.objective must be an instance of LatentSpaceObjective"
        assert type(self.init_train_x) is list, "load_train_data() must set self.init_train_x to a list of xs"
        assert torch.is_tensor(self.init_train_y), "load_train_data() must set self.init_train_y to a tensor of ys"
        assert torch.is_tensor(self.init_train_z), "load_train_data() must set self.init_train_z to a tensor of zs"
        assert len(self.init_train_x) == self.num_initialization_points, f"load_train_data() must initialize exactly self.num_initialization_points={self.num_initialization_points} xs, instead got {len(self.init_train_x)} xs"
        assert self.init_train_y.shape[0] == self.num_initialization_points, f"load_train_data() must initialize exactly self.num_initialization_points={self.num_initialization_points} ys, instead got {self.init_train_y.shape[0]} ys"
        assert self.init_train_z.shape[0] == self.num_initialization_points, f"load_train_data() must initialize exactly self.num_initialization_points={self.num_initialization_points} zs, instead got {self.init_train_z.shape[0]} zs"
        
        self.lolbo_state = LOLBOState(
            objective=self.objective,
            train_x=self.init_train_x,
            train_y=self.init_train_y,
            train_z=self.init_train_z,
            minimize=minimize,
            k=k,
            num_update_epochs=num_update_epochs,
            init_n_epochs=init_n_update_epochs,
            vae_learning_rate=vae_learning_rate,
            gp_learning_rate=gp_learning_rate,
            bsz=bsz,
            acq_func=acq_func,
            model_class=model,
            verbose=verbose,
            normalize_y=normalize_y,
            z_as_dist=z_as_dist,
            train_on_z_mean=train_on_z_mean,
            sample_z_e2e=sample_z_e2e,
        )

    def initialize_objective(self):
        ''' Initialize Objective for specific task
            must define self.objective object
            '''
        return self


    def load_train_data(self):
        ''' Load in or randomly initialize self.num_initialization_points
            total initial data points to kick-off optimization 
            Must define the following:
                self.init_train_x (a list of x's)
                self.init_train_y (a tensor of scores/y's)
                self.init_train_y (a tensor of corresponding latent space points)
        '''
        return self

    def sample_train_data(self):
        self.init_train_z = torch.randn(torch.Size([10 * self.num_initialization_points, self.objective.dim])) * self.sample_scale
        batch_size = 8
        num_batches = np.ceil(self.num_initialization_points / batch_size).astype(int)
        train_y = np.zeros((0, 1))
        train_z = torch.zeros((0, self.objective.dim))
        train_x = []
        valid_z = []
        idx = 0
        while len(train_y) < self.num_initialization_points:
            lb, ub = (batch_size * idx), (batch_size * (idx + 1))
            Z = self.init_train_z[lb:ub].cuda()
            out_dict = self.objective(Z)
            valid = out_dict['scores']
            train_y = np.append(train_y, out_dict['scores'][:, None], axis=0) 
            train_x = train_x + out_dict['decoded_xs'].tolist()
            train_z = torch.cat((train_z, out_dict['valid_zs'].cpu()))
            idx += 1

        self.init_train_z = train_z[:self.num_initialization_points]    
        self.init_train_y = torch.Tensor(train_y)[:self.num_initialization_points]
        self.init_train_x = train_x[:self.num_initialization_points]

    def set_seed(self):
        # The flag below controls whether to allow TF32 on matmul. This flag defaults to False
        # in PyTorch 1.12 and later.
        torch.backends.cuda.matmul.allow_tf32 = False
        # The flag below controls whether to allow TF32 on cuDNN. This flag defaults to True.
        torch.backends.cudnn.allow_tf32 = False
        if self.seed is not None:
            torch.manual_seed(self.seed) 
            random.seed(self.seed)
            np.random.seed(self.seed)
            torch.cuda.manual_seed(self.seed)
            torch.cuda.manual_seed(self.seed)
            torch.cuda.manual_seed_all(self.seed)
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
            os.environ["PYTHONHASHSEED"] = str(self.seed)

        return self


    def create_wandb_tracker(self):
        if self.track_with_wandb:
            self.tracker = wandb.init(
                project=self.wandb_project_name,
                entity=self.wandb_entity,
                config={k: v for method_dict in self.method_args.values() for k, v in method_dict.items()},
            ) 
            self.wandb_run_name = wandb.run.name
        else:
            self.tracker = None 
            self.wandb_run_name = 'no-wandb-tracking'
        
        return self


    def log_data_to_wandb_on_each_loop(self):
        if self.track_with_wandb:
            dict_log = {
                "best_found":self.lolbo_state.best_score_seen,
                "n_oracle_calls":self.lolbo_state.objective.num_calls,
                "total_number_of_e2e_updates":self.lolbo_state.tot_num_e2e_updates,
                "best_input_seen":self.lolbo_state.best_x_seen,
            }
            dict_log[f"TR_length"] = self.lolbo_state.tr_state.length
            self.tracker.log(dict_log)

        return self


    def run_lolbo(self): 
        ''' Main optimization loop
        '''
        # creates wandb tracker iff self.track_with_wandb == True
        self.create_wandb_tracker()
        #main optimization loop
        
        while self.lolbo_state.objective.num_calls < self.max_n_oracle_calls:
            self.log_data_to_wandb_on_each_loop()
            # update models end to end when we fail to make
            #   progress e2e_freq times in a row (e2e_freq=10 by default)
            if (self.lolbo_state.progress_fails_since_last_e2e >= self.e2e_freq) and self.update_e2e:
                self.lolbo_state.update_models_e2e()
                self.lolbo_state.recenter()
                self.save_to_csv()
            
            else: # otherwise, just update the surrogate model on data
                self.lolbo_state.update_surrogate_model()
            # generate new candidate points, evaluate them, and update data
            if self.save_moments:
                mu, sigma = self.lolbo_state.acquisition(save_moments=True)
                self.save_moments_to_csv(mu, sigma)
            else:
                self.lolbo_state.acquisition()
            if self.lolbo_state.tr_state.restart_triggered:
                self.lolbo_state.initialize_tr_state()
            # if a new best has been found, print out new best input and score:
            if self.lolbo_state.new_best_found:
                if self.verbose:
                    print("\nNew best found:")
                    self.print_progress_update()
                self.lolbo_state.new_best_found = False
            self.save_to_csv()
        
        self.save_to_csv()
        # if verbose, print final results
        if self.verbose:
            print("\nOptimization Run Finished, Final Results:")
            self.print_progress_update()

            
        return self  
    
    
    def run_vanilla(self): 
        ''' Main optimization loop
        '''
        # creates wandb tracker iff self.track_with_wandb == True
        self.create_wandb_tracker()
        #main optimization loop

        while self.lolbo_state.objective.num_calls < self.max_n_oracle_calls:
            self.log_data_to_wandb_on_each_loop()
            # update models end to end when we fail to make
            #   progress e2e_freq times in a row (e2e_freq=10 by default)
            if (self.lolbo_state.progress_fails_since_last_e2e >= self.e2e_freq) and self.update_e2e:
                self.lolbo_state.update_models_e2e()
                self.lolbo_state.recenter()
                self.save_to_csv()
            
            else: # otherwise, just update the surrogate model on data
                self.lolbo_state.update_surrogate_model()
            # generate new candidate points, evaluate them, and update data
            if self.save_moments:
                mu, sigma = self.lolbo_state.acquisition(save_moments=True)
                self.save_moments_to_csv(mu, sigma)
            else:
                self.lolbo_state.acquisition()
            if self.lolbo_state.tr_state.restart_triggered:
                self.lolbo_state.initialize_tr_state()
            # if a new best has been found, print out new best input and score:
            if self.lolbo_state.new_best_found:
                if self.verbose:
                    print("\nNew best found:")
                    self.print_progress_update()
                self.lolbo_state.new_best_found = False
            self.save_to_csv()
        self.save_to_csv()
        # if verbose, print final results
        if self.verbose:
            print("\nOptimization Run Finished, Final Results:")
            self.print_progress_update()

            
        return self  

    @batchable
    def _get_input_predictions(self, X: list):
        gp = self.lolbo_state.model
        obj = self.lolbo_state.objective
        obj.vae.eval()
        gp.eval()
        z, loss, z_mu, z_sigma = obj.vae_forward(X, return_mu_sigma=True)
        post = gp.posterior(z_mu)

        return (
            z_mu, 
            post.mean  * self.lolbo_state.ystd + self.lolbo_state.ymean, 
            post.variance.sqrt() * self.lolbo_state.ystd,
        )
    
    @batchable
    def _get_train_predictions(self, Z: list):
        gp = self.lolbo_state.model
        obj = self.lolbo_state.objective
        gp.eval()
        post = gp.posterior(Z.cuda())
        
        return (
            post.mean * self.lolbo_state.ystd + self.lolbo_state.ymean, 
            post.variance.sqrt() * self.lolbo_state.ystd,
        )
    

    @batchable
    def _get_output(self, X, Z_mu: Tensor):
        obj = self.lolbo_state.objective
        obj.vae.eval()
        decoded_output = obj(Z_mu, filter_invalid=False)
        recon_error = decoded_output["decoded_xs"] == np.array(X)
        valid = decoded_output["bool_arr"]
        return Tensor(recon_error), Tensor(valid)

    @batchable
    def _get_latent_predictions(self, Z_sample: Tensor):
        gp = self.lolbo_state.model
        obj = self.lolbo_state.objective
        obj.vae.eval()
        gp.eval()
        decoded_output = obj(Z_sample, filter_invalid=False)
        post = gp.posterior(Z_sample)
        valid = decoded_output["bool_arr"]
        y_decoded = Tensor(decoded_output["scores"])
        return (
            y_decoded.unsqueeze(-1), 
            post.mean * self.lolbo_state.ystd + self.lolbo_state.ymean, 
            post.variance.sqrt() * self.lolbo_state.ystd,
        )
    

    def run_prediction(self):
        # calibrate the GP by itself, then do the end-to-end
        # not really necessary for DKL since there are (almost) no GP HPs to 
        # tune anyway
        DIM = 256
        num_test = self.num_initialization_points * 100 

        print("End to end")
        self.lolbo_state.update_surrogate_model()
        self.lolbo_state.update_models_e2e()
        print("\n\nDone. Starting the predictions!\n\n")
        latent_mean, latent_std = self._get_train_predictions(self.lolbo_state.train_z)
        
        Z_mu, latent_sampled_mean, latent_sampled_std = self._get_input_predictions(self.init_train_x)
        #recon_error, num_valid = self._get_output(self.init_train_x, Z_mu)
        Z_sample = torch.randn(num_test, DIM).to(Z_mu) * self.sample_scale
        
        y_decoded, mean, std = self._get_latent_predictions(Z_sample)
        
        os.makedirs(f"{self.experiment_name}/{self.task_id}", exist_ok=True)
        pd.DataFrame(y_decoded.detach().numpy()).to_csv(f"{self.experiment_name}/{self.task_id}/{self.sample_scale}_vae_samples.csv")
        raise SystemExit
        plot_predictions(
            observations=y_decoded.cpu(), 
            pred=mean.cpu(), 
            uncert=std.cpu(), 
            save_path=self.experiment_name, 
            plot_name="decode_error",
        )
        #input_pred_rmse = torch.pow(self.init_train_y.cpu() - latent_mean.cpu(), 2).mean().sqrt()
        #print(latent_pred_rmse, input_pred_rmse)
        
        plot_predictions(
            observations=self.init_train_y.cpu(), 
            pred=latent_mean.cpu(), 
            uncert=latent_std.cpu(), 
            save_path=self.experiment_name, 
            plot_name="prediction_error_on_train_z",
        )
        plot_predictions(
            observations=self.init_train_y.cpu(), 
            pred=latent_sampled_mean.cpu(), 
            uncert=latent_sampled_std.cpu(), 
            save_path=self.experiment_name, 
            plot_name="prediction_error_on_mean",
        )

    def print_progress_update(self):
        ''' Important data printed each time a new
            best input is found, as well as at the end prediction
            of the optimization run
            (only used if self.verbose==True)
            More print statements can be added her as desired
        '''
        if self.track_with_wandb:
            print(f"Optimization Run: {self.wandb_project_name}, {wandb.run.name}")
        print(f"Best X Found: {self.lolbo_state.best_x_seen}")
        print(f"Best {self.objective.task_id} Score: {self.lolbo_state.best_score_seen}")
        print(f"Total Number of Oracle Calls (Function Evaluations): {self.lolbo_state.objective.num_calls}")

        return self


    def log_topk_table_wandb(self):
        ''' After optimization finishes, log
            top k inputs and scores found
            during optimization '''
        if self.track_with_wandb:
            cols = ["Top K Scores", "Top K Strings"]
            data_list = []
            for ix, score in enumerate(self.lolbo_state.top_k_scores):
                data_list.append([ score, str(self.lolbo_state.top_k_xs[ix]) ])
            top_k_table = wandb.Table(columns=cols, data=data_list)
            self.tracker.log({f"top_k_table": top_k_table})
            self.tracker.finish()

        return self

    def save_moments_to_csv(self, mean, std):
        save_dir = os.environ.get("SAVE_DIR", "..")
        moments_save_path = f"{save_dir}/result_moments/{self.experiment_name}/{self.task_id}/{self.model}_{self.lolbo_state.acq_func}"
        os.makedirs(moments_save_path, exist_ok=True)
        dups = self.lolbo_state.duplicates
        moments = torch.cat((mean, std))
        moments = moments.cpu().detach().numpy()
        filepath = f"{moments_save_path}/{self.task_id}_{self.model}_{self.lolbo_state.acq_func}_{self.seed}.csv"
        mean_cols = [f"mu_{i}" for i in range(mean.shape[0])]
        std_cols = [f"std_{i}" for i in range(mean.shape[0])]
        try:
            df = pd.read_csv(filepath)
            new_df = pd.DataFrame(moments.T, columns=mean_cols + std_cols)
            df = df.append(new_df)
            df.to_csv(filepath, index=False)

        except FileNotFoundError:
            df = pd.DataFrame(moments.T, columns=mean_cols + std_cols)

        
    def save_to_csv(self, save_best_nbr: int = 1000):
        save_dir = os.environ.get("SAVE_DIR", "..")
        res_save_path = f"{save_dir}/result_values/{self.experiment_name}/{self.task_id}/{self.model}_{self.lolbo_state.acq_func}"
        str_save_path = f"{save_dir}/result_strings/{self.experiment_name}/{self.task_id}/{self.model}_{self.lolbo_state.acq_func}"
        z_save_path = f"{save_dir}/result_z/{self.experiment_name}/{self.task_id}/{self.model}_{self.lolbo_state.acq_func}"
        dups_save_path = f"{save_dir}/duplicates/{self.experiment_name}/{self.task_id}/{self.model}_{self.lolbo_state.acq_func}"
        os.makedirs(res_save_path, exist_ok=True)
        os.makedirs(str_save_path, exist_ok=True)
        os.makedirs(z_save_path, exist_ok=True)
        os.makedirs(dups_save_path, exist_ok=True)
        Y = self.lolbo_state.orig_train_y.flatten()
        df = pd.DataFrame({self.task_id: Y})
        df.to_csv(f"{res_save_path}/{self.task_id}_{self.model}_{self.lolbo_state.acq_func}_{self.seed}.csv")
        best_indices = torch.argsort(Y, descending=True)[:save_best_nbr]
        best_X = np.array(self.lolbo_state.train_x)[best_indices].tolist()
        best_y = Y[best_indices]
        df_indices = pd.DataFrame({"string": best_X, "{self.task_id}": best_y})
        df_dups = pd.DataFrame(self.lolbo_state.duplicates)
        df_z = pd.DataFrame(self.lolbo_state.train_z.numpy().astype(np.float16))
        
        df_z["acqtype"] = self.lolbo_state.z_acqtype
        df_z.to_csv(f"{z_save_path}/{self.task_id}_{self.model}_{self.lolbo_state.acq_func}_{self.seed}_z.csv")
        df_dups.to_csv(f"{dups_save_path}/{self.task_id}_{self.model}_{self.lolbo_state.acq_func}_{self.seed}_dups.csv")
        df_indices.to_csv(f"{str_save_path}/{self.task_id}_{self.model}_{self.lolbo_state.acq_func}_{self.seed}_strings.csv")



    def done(self):
        return None


def new(**kwargs):
    return Optimize(**kwargs)

if __name__ == "__main__":
    fire.Fire(Optimize)
