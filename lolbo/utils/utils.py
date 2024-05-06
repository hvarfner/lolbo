from typing import Any
import torch
import math
import numpy as np
import selfies as sf
from torch.utils.data import TensorDataset, DataLoader
from lolbo.utils.pred_utils import batchable
from lolbo.latent_space_objective import LatentSpaceObjective

import gpytorch
from gpytorch.models import ApproximateGP, ExactGP
from botorch.sampling.qmc import NormalQMCEngine
from botorch.fit import fit_gpytorch_mll

def update_models_end_to_end(
    train_x,
    train_y_scores,
    objective,
    model,
    mll,
    vae_learning_rte,
    gp_learning_rte,
    num_update_epochs,
    sample_z_e2e: bool = True,
    train_on_z: bool = False,
):
    '''Finetune VAE end to end with surrogate model
    This method is build to be compatible with the 
    SELFIES VAE interface
    '''
    objective.vae.train()
    #breakpoint()
    model.train() 

    optimizer = torch.optim.Adam([
            {'params': objective.vae.parameters(), 'lr': vae_learning_rte},
            {'params': model.parameters(), 'lr': gp_learning_rte} ])
    # max batch size smaller to avoid memory limit with longer strings (more tokens)
    max_string_length = len(max(train_x, key=len))
    bsz = max(1, int(2560/max_string_length)) 
    num_batches = math.ceil(len(train_x) / bsz)
    for _ in range(num_update_epochs):
        for batch_ix in range(num_batches):
            start_idx, stop_idx = batch_ix*bsz, (batch_ix+1)*bsz
            batch_list = train_x[start_idx:stop_idx]
            z, vae_loss, z_mu, z_sigma = objective.vae_forward(batch_list, return_mu_sigma=True)
            if train_on_z:
                z = torch.cat((z_mu, z_sigma), dim=-1)
            batch_y = train_y_scores[start_idx:stop_idx]
            batch_y = torch.tensor(batch_y).float() 
            if not sample_z_e2e:
                pred = model(z_mu)
            else:
                pred = model(z)
            surr_loss = -mll(pred, batch_y.cuda())
            # add losses and back prop 
            loss = vae_loss + surr_loss
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(objective.vae.parameters(), max_norm=1.0)
            optimizer.step()
    objective.vae.eval()
    model.eval()

    return objective, model

def update_henry_surr_model(
    train_x,
    train_y,
    objective,
    model,
    mll,
    vae_learning_rte,
    gp_learning_rte,
    n_epochs: int = 20,
    batch_size: int = 256,
    train_e2e: bool = False,
):
    model.train()
    objective.vae.train()
    params = [{'params': model.parameters(), 'lr': gp_learning_rte}]
    if train_e2e:
        params = params + [{'params': objective.vae.parameters(), 'lr': vae_learning_rte}]

    optimizer = torch.optim.Adam(params=params)
    # need to pass through VAE here to get the recon losses
    num_batches = math.ceil(len(train_x) / batch_size)
    shuffled_idcs = np.arange(len(train_x)).astype(int)

    if not train_e2e:
        z_mu, z_sigma, batch_losses = _get_predictions(None, train_x, obj=objective, return_loss=True)
        z_all = torch.cat((z_mu, z_sigma), dim=-1).detach()

    for ep in range(n_epochs):
        # inplace for some insane reason
        np.random.shuffle(shuffled_idcs)
        for batch_idx in range(num_batches):
            optimizer.zero_grad()
            lb, ub = (batch_idx * batch_size), ((batch_idx + 1) * batch_size)
            indices = shuffled_idcs[lb:ub]
            batch_x = np.array(train_x)[indices].tolist()
            batch_y = train_y[indices]
            if train_e2e:
                z_mu, z_sigma, batch_losses = _get_predictions(None, batch_x, obj=objective, return_loss=True)
                z = torch.cat((z_mu, z_sigma), dim=-1) 
            else:
                z = z_all[indices]
            pred = model(z)
            loss = -mll(pred, batch_y.cuda().squeeze(-1))
            if train_e2e:
                loss = loss + batch_losses.mean()
                print(batch_losses.mean())
            loss.backward()
            optimizer.step()
    
    model = model.eval()
    objective.vae.eval()
    
    return model


def update_surr_model(
    model,
    mll,
    gp_learning_rte,
    train_z,
    train_y,
    n_epochs,
    train_on_z: bool = False,
):
    model = model.train() 
    optimizer = torch.optim.Adam([{'params': model.parameters(), 'lr': gp_learning_rte} ])
    is_exact = isinstance(model, ExactGP)
    if is_exact:
        train_bsz = len(train_z)
    else:
        train_bsz = min(len(train_y), 16)
    train_dataset = TensorDataset(train_z.cuda(), train_y.cuda())
    train_loader = DataLoader(train_dataset, batch_size=train_bsz, shuffle=not is_exact)
    means = torch.empty(0)
    all_scores = torch.empty(0)
    for ep in range(n_epochs):

        #print(model.covar_module.base_kernel.lengthscale)
        #print(f"{ep} / {n_epochs}")
        for (inputs, scores) in train_loader:
            optimizer.zero_grad()
            
            output = model(inputs.cuda())
            loss = -mll(output, scores.cuda().squeeze(-1))
            loss.backward()
            optimizer.step()
            #if ep == (n_epochs - 1):
            #    means = torch.cat((means, output.mean.flatten().to(means)))
            #    all_scores = torch.cat((all_scores, torch.Tensor(scores).to(all_scores)))
    model = model.eval()
    return model


def update_exact_end_to_end(
    train_x,
    train_y_scores,
    objective,
    model,
    mll,
    vae_learning_rte,
    gp_learning_rte,
    num_update_epochs,
):
    '''Finetune VAE end to end with surrogate model
    This method is build to be compatible with the 
    SELFIES VAE interface
    '''
    objective.vae.train()
    #breakpoint()
    model.train() 
    print("Training end to end")

    optimizer = torch.optim.Adam([
            {'params': objective.vae.parameters(), 'lr': vae_learning_rte},
            {'params': model.parameters(), 'lr': gp_learning_rte} ])
    # max batch size smaller to avoid memory limit with longer strings (more tokens)
    y = train_y_scores.cuda().squeeze(-1)

    for idx in range(num_update_epochs):
        optimizer.zero_grad()
        # need to pass through VAE here to get the recon losses
        z_mu, z_sigma, batch_losses = _get_predictions(None, train_x, obj=objective, return_loss=True)
        z = torch.cat((z_mu, z_sigma), dim=-1) 
        model.set_train_data(inputs=z, targets=y.to(z), strict=(idx>0))
        pred = model(z)
        surr_loss = -mll(pred, y.to(z))
        # add losses and back prop 
        loss = batch_losses.mean() + surr_loss.mean()

        loss.backward()
        torch.nn.utils.clip_grad_norm_(objective.vae.parameters(), max_norm=1.0)
        optimizer.step()
        

    z_mu, z_sigma, batch_losses = _get_predictions(None, train_x, obj=objective, return_loss=True)
    z = torch.cat((z_mu, z_sigma), dim=-1) 
    
    model.set_train_data(inputs=z, targets=y.to(z))
    objective.vae.eval()
    model.eval()

    return objective, model

def update_exact_surr_model(
    train_z,
    train_y_scores,
    objective,
    model,
    mll,
    gp_learning_rte,
    num_update_epochs,
    samples_per_x: int = 16,
):
    '''Finetune VAE end to end with surrogate model
    This method is build to be compatible with the 
    SELFIES VAE interface
    '''
    model.train() 
    optimizer = torch.optim.Adam([
        {'params': model.parameters(), 'lr': gp_learning_rte}
    ])
    z = train_z.cuda()
    y = train_y_scores.cuda().squeeze(-1)
    for _ in range(num_update_epochs):
        optimizer.zero_grad()
        with gpytorch.settings.debug(False):
            pred = model(z)
        loss = -mll(pred, y)
        # add losses and back prop 

        loss.mean().backward()
        optimizer.step()
    objective.vae.eval()
    model.eval()

    return model
    return model



def update_surr_model_sampled_z(
    train_x,
    train_y_scores,
    objective,
    model,
    mll,
    gp_learning_rte,
    num_update_epochs,
    samples_per_x: int = 16,
):
    '''Finetune VAE end to end with surrogate model
    This method is build to be compatible with the 
    SELFIES VAE interface
    '''
    z_mu, z_sigma = _get_predictions(None, train_x, obj=objective)
    model.train() 
    optimizer = torch.optim.Adam([
        {'params': model.parameters(), 'lr': gp_learning_rte}
    ])
    # max batch size smaller to avoid memory limit with longer strings (more tokens)
    max_string_length = len(max(train_x, key=len))
    bsz = max(1, int(2560/max_string_length)) 
    is_exact = isinstance(model, ExactGP)
    if is_exact:
        bsz = len(train_x)

    num_batches = math.ceil(len(train_x) / bsz)
    # since Cov == I, we don't need a multi-dimensional and can just loc-scale instead
    for batch_ix in range(num_batches):
        start_idx, stop_idx = batch_ix*bsz, (batch_ix+1)*bsz
        batch_list = train_x[start_idx:stop_idx]
    sampler = NormalQMCEngine(z_mu.shape[-1])

    for _ in range(num_update_epochs):
        base_samples = sampler.draw(samples_per_x).unsqueeze(-2).to(z_mu)
    
        for batch_ix in range(num_batches):
            optimizer.zero_grad()
            start_idx, stop_idx = batch_ix*bsz, (batch_ix+1)*bsz
            batch_list = train_x[start_idx:stop_idx]
            batch_y = train_y_scores[start_idx:stop_idx]
            batch_y = torch.tensor(batch_y).float() 
            batch_z_mu = z_mu[start_idx:stop_idx]
            batch_z_sigma = z_sigma[start_idx:stop_idx]
            z_samples = base_samples * batch_z_sigma + batch_z_mu
            with gpytorch.settings.debug(False):
                pred = model(z_samples)
            loss = -mll(pred, batch_y.cuda())
            # add losses and back prop 

            loss.mean().backward()
            optimizer.step()
    objective.vae.eval()
    model.eval()

    return model


@batchable
def _get_predictions(_: Any, X: list, obj: LatentSpaceObjective, return_loss: bool = False):
    obj.vae.eval()
    z, loss, z_mu, z_sigma = obj.vae_forward(X, return_mu_sigma=True)
    if return_loss:
        return z_mu, z_sigma, loss.unsqueeze(0)
    return z_mu, z_sigma