import os
import sys

import numpy as np
import torch
from torch.utils import data

from models.comboptnet import ilp_solver
from utils.constraint_generation import sample_constraints
from utils.utils import compute_normalized_solution, save_pickle, load_pickle, AvgMeters, check_equal_ys, \
    solve_unconstrained, load_with_default_yaml, save_dict_as_one_line_csv


def load_dataset(dataset_type, base_dataset_path, **dataset_params):
    dataset_path = os.path.join(base_dataset_path, dataset_type)
    dataset_loader_dict = dict(static_constraints=static_constraint_dataloader, knapsack=knapsack_dataloader)
    return dataset_loader_dict[dataset_type](dataset_path=dataset_path, **dataset_params)


def static_constraint_dataloader(dataset_path, dataset_specification, num_gt_variables, num_gt_constraints,
                                 dataset_seed, train_dataset_size, loader_params):
    dataset_path = os.path.join(dataset_path, dataset_specification, str(num_gt_variables) + '_dim',
                                str(num_gt_constraints) + '_const', str(dataset_seed), 'dataset.p')
    datasets = load_pickle(dataset_path)

    train_ys = [tuple(y) for c, y in datasets['train'][:train_dataset_size]]
    test_ys = [tuple(y) for c, y in datasets['test'][:train_dataset_size]]

    print(f'Successfully loaded Static Constraints dataset.\n'
          f'Number of distinct solutions in train set: {len(set(train_ys))}\n'
          f'Number of distinct solutions in test set: {len(set(test_ys))}')

    training_set = Dataset(datasets['train'][:train_dataset_size])
    train_iterator = data.DataLoader(training_set, **loader_params)

    test_iterator = data.DataLoader(Dataset(datasets['test']), **loader_params)

    return (train_iterator, test_iterator), datasets['metadata']


def knapsack_dataloader(dataset_path, loader_params):
    variable_range = dict(lb=0, ub=1)
    num_variables = 10

    train_encodings = np.load(os.path.join(dataset_path, 'train_encodings.npy'))
    train_ys = compute_normalized_solution(np.load(os.path.join(dataset_path, 'train_sols.npy')), **variable_range)
    train_dataset = list(zip(train_encodings, train_ys))
    training_set = Dataset(train_dataset)
    train_iterator = data.DataLoader(training_set, **loader_params)

    test_encodings = np.load(os.path.join(dataset_path, 'test_encodings.npy'))
    test_ys = compute_normalized_solution(np.load(os.path.join(dataset_path, 'test_sols.npy')), **variable_range)
    test_dataset = list(zip(test_encodings, test_ys))
    test_set = Dataset(test_dataset)
    test_iterator = data.DataLoader(test_set, **loader_params)

    distinct_ys_train = len(set([tuple(y) for y in train_ys]))
    distinct_ys_test = len(set([tuple(y) for y in test_ys]))
    print(f'Successfully loaded Knapsack dataset.\n'
          f'Number of distinct solutions in train set: {distinct_ys_train},\n'
          f'Number of distinct solutions in test set: {distinct_ys_test}')

    metadata = {"variable_range": variable_range,
                "num_variables": num_variables}

    return (train_iterator, test_iterator), metadata


class Dataset(data.Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        x, y = [torch.from_numpy(_x) for _x in self.dataset[index]]
        return x, y


def gen_constraints_dataset(train_dataset_size, test_dataset_size, seed, variable_range, num_variables,
                            num_constraints, positive_costs, constraint_params):
    np.random.seed(seed)
    constraints = sample_constraints(variable_range=variable_range,
                                     num_variables=num_variables,
                                     num_constraints=num_constraints,
                                     seed=seed, **constraint_params)
    metadata = dict(true_constraints=constraints, num_variables=num_variables, num_constraints=num_constraints,
                    variable_range=variable_range)

    c_l = []
    y_l = []
    dataset = []
    for _ in range(test_dataset_size + train_dataset_size):
        cost_vector = 2 * (np.random.rand(constraints.shape[1] - 1) - 0.5)
        if positive_costs:
            cost_vector = np.abs(cost_vector)
        y = ilp_solver(cost_vector=cost_vector, constraints=constraints, **variable_range)[0]
        y_norm = compute_normalized_solution(y, **variable_range)
        dataset.append((cost_vector, y_norm))
        c_l.append(cost_vector)
        y_l.append(y)
    cs, ys = np.stack(c_l, axis=0), np.stack(y_l, axis=0)

    num_distinct_ys = len(set([tuple(y) for _, y in dataset]))
    ys_uncon = solve_unconstrained(cs, **variable_range)
    match_boxconst_solution_acc = check_equal_ys(y_1=ys, y_2=ys_uncon)[1].mean()
    metrics = dict(num_distinct_ys=num_distinct_ys, match_boxconst_solution_acc=match_boxconst_solution_acc)
    print(f'Num distinct ys: {num_distinct_ys}, Match boxconst acc: {match_boxconst_solution_acc}')

    test_set = dataset[:test_dataset_size]
    train_set = dataset[test_dataset_size:]
    datasets = dict(metadata=metadata, train=train_set, test=test_set)
    return datasets, metrics


def main(working_dir, num_seeds, num_constraints, num_variables, data_gen_params):
    avg_meter = AvgMeters()
    all_metrics = {}
    for num_const, num_var in zip(num_constraints, num_variables):
        print(f'Gnerating dataset with {num_var} variables and {num_const} constraints...')
        for seed in range(num_seeds):
            dir = os.path.join(working_dir, str(num_var) + "_dim", str(num_const) + "_const", str(seed))
            os.makedirs(dir, exist_ok=True)
            datasets, metrics = gen_constraints_dataset(seed=seed, num_variables=num_var,
                                                        num_constraints=num_const, **data_gen_params)
            save_pickle(datasets, os.path.join(dir, 'dataset.p'))
            avg_meter.update(metrics)
        all_metrics.update(
            avg_meter.get_averages(prefix=str(num_var) + "_dim_" + str(num_const) + "_const_"))
        avg_meter.reset()
    save_dict_as_one_line_csv(all_metrics, filename=os.path.join(working_dir, "metrics.csv"))
    return all_metrics


if __name__ == "__main__":
    param_path = sys.argv[1]
    param_dict = load_with_default_yaml(path=param_path)
    main(**param_dict)
