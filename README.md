# Tree Construction + Primal-Dual Routing

This repository contains a single-script runner `Tree_construction_Primal_Dual.py`.
It supports multiple datasets and routing methods, and automatically generates:
- a `plot_output/` folder
- result CSV files


# Environment
Option 1: Conda

    conda env create -f environment.yml
    conda activate pd-routing

Option 2: pip

    pip install -r requirements.txt


# Inputs
Required arguments (all runs):

    --dataset_input: one of
        uniform_distribution_data,
        GMM_data,
        meituan_dataset_weekday,
        meituan_dataset_weekdend_day,
        worst_instance_greedy

    --routing_method: one of 
        PD_DFS, 
        PD_Greedy, 
        PD_DGreedy, 
        Greedy_baseline


Dataset-specific required arguments:

    uniform_distribution_data requires:
        --st_pair_size_input, --depot_size_input, --level_size_input, --seed_value_list_input

    GMM_data requires:
        --GMM_cluster_input, --GMM_sigma_input,
        --st_pair_size_input, --depot_size_input, --level_size_input, --seed_value_list_input

    meituan_dataset_weekday / meituan_dataset_weekdend_day requires:
        --depot_size_input, --level_size_input

    worst_instance_greedy requires:
        --st_pair_size_input


Routing_method-specific required arguments:

    PD_DFS, PD_Greedy, PD_DGreedy requires: 
        --use_constant_mst: 0 (disable) or 1 (enable)
        --mst_multiplier: 0 if disabled, otherwise a positive integer (e.g., 7)


# Output

After each run, the script creates:

    plot_output/ (figures)
    CSV result files (saved by the script)


# Examples

uniform_distribution_data + PD_DFS:

    python Tree_combination_Primal_Dual.py \
    --dataset_input uniform_distribution_data \
    --routing_method PD_DFS \
    --use_constant_mst 1 --mst_multiplier 7 \
    --st_pair_size_input 1000 \
    --depot_size_input 90 \
    --level_size_input 3 \
    --seed_value_list_input 1,2,3,4,5


GMM_data + PD_DFS:

    python Tree_combination_Primal_Dual.py \
    --dataset_input GMM_data \
    --routing_method PD_DFS \
    --use_constant_mst 0 --mst_multiplier 0 \
    --st_pair_size_input 1000 \
    --depot_size_input 90 \
    --level_size_input 3 \
    --seed_value_list_input 1,2,3,4,5 \
    --GMM_cluster_input 5 \
    --GMM_sigma_input 5

worst_instance_greedy + PD_DFS:

    python Tree_combination_Primal_Dual.py \
    --dataset_input worst_instance_greedy \
    --routing_method PD_DFS \
    --use_constant_mst 0 --mst_multiplier 0 \
    --st_pair_size_input 1000

meituan_dataset_weekday + PD_DGreedy:

    python Tree_combination_Primal_Dual.py \
    --dataset_input meituan_dataset_weekday \
    --routing_method PD_DGreedy \
    --use_constant_mst 0 --mst_multiplier 0 \
    --depot_size_input 120 \
    --level_size_input 3

uniform_distribution_data + Greedy_baseline:

    python Tree_combination_Primal_Dual.py \
    --dataset_input uniform_distribution_data \
    --routing_method Greedy_baseline \
    --st_pair_size_input 1000 \
    --depot_size_input 90 \
    --level_size_input 3 \
    --seed_value_list_input 1,2,3,4,5


# Notes

    Please run commands from the repository root.
    The script writes outputs to plot_output/ and CSV files as implemented in the script
