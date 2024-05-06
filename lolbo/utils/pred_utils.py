import os
import math
from typing import Any
import torch
from torch import Tensor

import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'serif'


def batchable(function, batch_size: int = 4):
    
    def decorated(cls: Any, *batch_inputs, **kwargs):
        input_length = len(batch_inputs[0])
        num_batches = math.ceil(input_length / batch_size)
        
        check_output = function(cls, 
            *[b_input[0:batch_size] for b_input in batch_inputs], 
        **kwargs)
        if not isinstance(check_output, tuple):
            check_output = (check_output, )
        collected_output = [torch.empty((0,) + op.shape[1:]).to(op) for op in check_output]

        for batch_index in range(num_batches):
            lower = batch_index * batch_size
            upper = min(input_length, (batch_index + 1) * batch_size)
            batched_input = [*[b_input[lower:upper] for b_input in batch_inputs]]
            new_output = function(cls, *batched_input, **kwargs)

            if not isinstance(new_output, tuple):
                new_output = (new_output, )

            for idx, op in enumerate(new_output):
                with torch.no_grad():
                    collected_output[idx] = torch.cat(
                        (collected_output[idx], op), dim=0)
        
        return tuple(c_op for c_op in collected_output)
    
    return decorated


def plot_predictions(observations: Tensor, pred: Tensor, uncert: Tensor, save_path: str, plot_name: str):
    sort_order = torch.sort(observations, dim=0).indices
    sorted_output = observations[sort_order].flatten()
    sorted_means = pred[sort_order].flatten()
    sorted_stds = uncert[sort_order].flatten()
    diff = sorted_output[-1] - sorted_output[0]
    range_ = sorted_output[0] - 0.1 * diff, sorted_output[-1] + 0.1 * diff 
    input_pred_rmse = torch.pow(sorted_output - sorted_means, 2).mean().sqrt().item()

    os.makedirs(f"../pred_plots/{save_path}", exist_ok=True)
    with torch.no_grad():
        plt.cla()
        plt.plot(range_, range_, color='blue', linestyle="dashed")
        plt.vlines(sorted_output, sorted_means - 2 * sorted_stds, sorted_means + 2 * sorted_stds, color="grey", alpha=0.15, linewidth=0.1)
        plt.scatter(sorted_output, sorted_means, color="black", s=5)
        plt.xlabel("Actual values")
        plt.ylabel("Predicted values")
        plt.title(f"RMSE: {input_pred_rmse}", fontsize=16)    
        plt.grid(True)
        plt.savefig(f"../pred_plots/{save_path}/{plot_name}.pdf")
                