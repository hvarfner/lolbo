import os
import math
from typing import Union, List, Optional
from os.path import join
from glob import glob

from fire import Fire
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import sem

plt.rcParams['font.family'] = 'serif'


COLORS = {
    "dkl_ts": "red",
    "vanilla_ts": "blue", 
    "vanilla_ei": "green", 
}


NICE_BENCH_NAMES = {
"med1": "Median molecules 1", 
"med2": "Median molecules 2", 
"pdop": "Perindopril MPO",
"osmb": "Osimertinib MPO",
"adip": "Amlodipine MPO",
"siga": "Sitagliptin MPO",
"zale": "Zaleplon MPO",
"valt": "Valsartan SMARTS",
"dhop": "Deco Hop",
"shop": "Scaffold Hop",
"rano": "Ranolazine MPO",
"fexo": "Fexofenadine MPO",
}

NICE_METHOD_NAMES = {
    "dkl_ts": "LOLBO",
    "vanilla_ei": "Vanilla BO (EI)",
    "vanilla_ts": "Vanilla BO (TS)",
    "unvanilla": "Unwhitened Vanilla BO",
    "unzvanilla": "$Unwhitened  k_{Henry}$",
    }

MAX_PLOTS_PER_ROW = 3
N_ERROR = 1



def _typecheck_list(inp: Union[str, list]):
    if isinstance(inp, str):
        return [inp]
    return inp


def filter_benchmark_or_method(paths: List[str], to_keep: Union[List, str]):
    paths_to_keep = []
    if to_keep is not None:
        to_keep = _typecheck_list(to_keep)
        
        for p in paths:
            for tk in to_keep:
                if tk in p:
                    paths_to_keep.append(p)
                    break
    return paths_to_keep

def _get_run_seed(result_path: str):
    return int(result_path.split("_")[-1].replace(".csv", ""))

def compute_min(df):
    return np.maximum.accumulate(df, axis=1)

@Fire
def plot_results(
    path: str = "result_values", 
    benchmarks: Optional[Union[str, list]] = None, 
    methods: Optional[Union[str, list]] = None, 
    use_length: str = "min",
    plot_name: str = "lolbo_results",
    plot_each: bool = False,
    start_at: int = 0,
) -> None:
    all_results = {}
    bench_paths = glob(join(path, "*"))
    if benchmarks is not None:
        bench_paths = filter_benchmark_or_method(bench_paths, benchmarks)
    
    for bench in bench_paths:
        bench_name = bench.split("/")[-1]
        #bench_name = NICE_BENCH_NAMES.get(bench_name, bench_name)
        all_results[bench_name] = {}
        method_paths = glob(join(bench, "*"))
        if methods is not None:
            method_paths = filter_benchmark_or_method(method_paths, methods)
        for method in method_paths:
            method_name =  method.split("/")[-1]
            #method_name = NICE_METHOD_NAMES.get(method_name, method_name)
            csvs = glob(join(method, "*.csv"))
            print(bench, method, csvs)
            
            for csv in csvs:
                try: 
                    df = pd.read_csv(csv)
                except pd.errors.EmptyDataError:
                    print(csv, "was missing. Removing.")
                    os.remove(csv)
    
            csv_lengths = [len(pd.read_csv(csv)) for csv in csvs]
            result_length = eval(use_length)(csv_lengths)
            result_array = np.zeros((len(csvs), result_length)) * np.nan
            
            for r_idx, csv in enumerate(sorted(csvs, key=lambda x: _get_run_seed(x))):
                df = pd.read_csv(csv).iloc[:result_length, :]
                res = df.loc[:, bench_name]
                result_array[r_idx, :len(res)] = res
            
            all_results[bench_name][method_name] = result_array
    
    assert len(bench_paths) > 0, "All benchmarks are filtered out."
    nrows, ncols = math.ceil(len(bench_paths) / MAX_PLOTS_PER_ROW), min(len(bench_paths), MAX_PLOTS_PER_ROW) 
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(4 * ncols, 3 * nrows))
    axes = np.array(axes).reshape(nrows, -1)
    for bench_idx, b_name in enumerate(all_results.keys()):
        ax = axes[bench_idx // MAX_PLOTS_PER_ROW, bench_idx % MAX_PLOTS_PER_ROW]
        methods_for_bench = all_results[b_name]
        for m_name, m_result in methods_for_bench.items():
            res = compute_min(m_result)
            mean, std = res.mean(axis=0), sem(res, axis=0)
            X_plot = np.arange(0, m_result.shape[-1])[:100000]
            X_plot, mean, std = X_plot[start_at:], mean[start_at:100000], std[start_at:100000]
            if plot_each:
                ax.plot(X_plot, res.T, label=NICE_METHOD_NAMES.get(m_name, m_name), color=COLORS[m_name])
            else:
                ax.plot(X_plot, mean, label=NICE_METHOD_NAMES.get(m_name, m_name), color=COLORS[m_name])
                ax.fill_between(X_plot, mean - N_ERROR * std, mean + N_ERROR * std, alpha=0.15, color=COLORS[m_name])
                ax.plot(X_plot, mean + N_ERROR * std, alpha=0.3, color=COLORS[m_name])
                ax.plot(X_plot, mean - N_ERROR * std, alpha=0.3, color=COLORS[m_name])

        ylabel = "Best Observed Value"
        ax.set_ylabel(ylabel, fontsize=18)
        ax.set_xlabel("Iteration", fontsize=18)
        ax.tick_params(axis='both', labelsize=15)
        ax.set_title(NICE_BENCH_NAMES.get(b_name, b_name), fontsize=20)
        ax.grid(True)
    plt.tight_layout()
    plt.legend(fontsize=16)
    plt.savefig(f"{plot_name}.pdf")
        
    plt.show()