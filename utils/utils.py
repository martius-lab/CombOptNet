import csv
import os
import pickle
import random
import sys
from itertools import chain, combinations

import numpy as np
import torch
import yaml
from torch import Tensor
from torch.nn.parameter import Parameter

epsilon_constant = 1e-8


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name=None, fmt=":f"):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = "{name} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)


class AvgMeters(object):
    def __init__(self):
        self.metrics = {}

    def update_metric(self, name, value, n=1):
        if name not in self.metrics.keys():
            self.metrics[name] = AverageMeter(name)
        self.metrics[name].update(value, n=n)

    def get_averages(self, prefix=''):
        return {prefix + key: avg_meter.avg for key, avg_meter in self.metrics.items()}

    def update(self, dct, n=1):
        for key, value in dct.items():
            self.update_metric(name=key, value=value, n=n)

    def reset(self):
        self.metrics = {}


def check_equal_ys(y_1, y_2, threshold=1e-5):
    equal_variable_wise = np.abs(y_1 - y_2) < threshold
    equal_instance_wise = equal_variable_wise.all(axis=1)
    return equal_variable_wise, equal_instance_wise


def compute_metrics(y, y_true, y_uncon=None, prefix='', acc_threshold=1e-5):
    y = y.cpu().detach().numpy()
    metric_dict = {}

    y_true = y_true.cpu().detach().numpy()
    correct, correct_perfect = check_equal_ys(y, y_true, threshold=acc_threshold)
    metric_dict.update(dict(acc=correct, perfect_accuracy=correct_perfect))

    if y_uncon is not None:
        y_uncon = y_uncon.cpu().detach().numpy()
        match_uncon, match_uncon_perfect = check_equal_ys(y, y_uncon, threshold=acc_threshold)
        metric_dict.update(dict(match_uncon_accuracy=match_uncon, match_uncon_perfect_accuracy=match_uncon_perfect))

    metric_dict = {prefix + key: value.mean() for key, value in metric_dict.items()}
    return metric_dict


def check_if_zero_or_one(y):
    return torch.all(torch.logical_or(torch.isclose(y, torch.ones_like(y)), torch.isclose(y, torch.zeros_like(y))))


class HammingLoss(torch.nn.Module):
    def forward(self, suggested, target):
        suggested += 0.5  # solutions are normalized to [-0.5, 0.5]
        target += 0.5
        if not (check_if_zero_or_one(suggested) and check_if_zero_or_one(target)):
            print(suggested, target)
            raise ValueError(
                f'Hamming loss only defined for zero/one predictions and targets. Instead received {suggested}, {target}.')
        errors = suggested * (1.0 - target) + (1.0 - suggested) * target
        return errors.mean(dim=0).sum()


class IOULoss(torch.nn.Module):
    def forward(self, suggested, target):
        suggested += 0.5  # solutions are normalized to [-0.5, 0.5]
        target += 0.5
        if not (check_if_zero_or_one(suggested) and check_if_zero_or_one(target)):
            raise ValueError(
                f'Hamming loss only defined for zero/one predictions and targets. Instead received {suggested}, {target}.')
        matches = suggested * target  # // true positives
        fps = torch.relu(suggested - target)  # // false positives
        fns = torch.relu(target - suggested)  # // false negatives
        iou = (matches.sum()) / (matches.sum() + fps.sum() + fns.sum())
        return 1 - iou  # // or (1-iou)*(1-iou)


class L0Loss(torch.nn.Module):
    def forward(self, suggested, target):
        errors = (suggested - target).abs()
        return torch.max(errors, dim=-1)[0].mean()


class HuberLoss(torch.nn.Module):
    def __init__(self, beta=0.3):
        self.beta = beta
        super(HuberLoss, self).__init__()

    def forward(self, suggested, target):
        errors = torch.abs(suggested - target)
        mask = errors < self.beta
        l2_errors = 0.5 * (errors ** 2) / self.beta
        l1_errors = errors - 0.5 * self.beta
        combined_errors = mask * l2_errors + ~mask * l1_errors
        return combined_errors.mean(dim=0).sum()


def loss_from_string(loss_name):
    dct = {"Hamming": HammingLoss(), "MSE": torch.nn.MSELoss(), "L1": torch.nn.L1Loss(), "L0": L0Loss(),
           "IOU": IOULoss(), "Huber": HuberLoss(beta=0.3)}
    return dct[loss_name]


def optimizer_from_string(optimizer_name):
    dct = {"Adam": torch.optim.Adam, "SGD": torch.optim.SGD, "RMSprop": torch.optim.RMSprop}
    return dct[optimizer_name]


def save_pickle(data, path):
    with open(path, "wb") as fh:
        pickle.dump(data, fh)


def load_pickle(path):
    with open(path, "rb") as fh:
        return pickle.load(fh)


def set_seed(seed):
    if seed is not None:
        random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        np.random.seed(seed)


class ParallelProcessing:
    def __init__(self):
        self.remote_fns = dict()

    def ray_available(self):
        if 'ray' in sys.modules and sys.modules['ray'].is_initialized():
            self.ray = sys.modules['ray']
            return True
        else:
            return False

    def get_remote(self, function):
        if type(function) is self.ray.remote_function.RemoteFunction:
            return function.remote
        else:
            if function in self.remote_fns:
                remote_fn = self.remote_fns[function]
            else:
                remote_fn = self.ray.remote(function)
                self.remote_fns[function] = remote_fn
            assert isinstance(remote_fn, self.ray.remote_function.RemoteFunction)
            return remote_fn.remote

    def maybe_parallelize(self, function, static_kwargs, dynamic_kwargs):
        if self.ray_available():
            remote_fn = self.get_remote(function)
            return self.ray.get([remote_fn(**static_kwargs, **kwargs) for kwargs in dynamic_kwargs])
        else:
            return [function(**static_kwargs, **kwargs) for kwargs in dynamic_kwargs]


def torch_parameter_from_numpy(array):
    param = Parameter(Tensor(*array.shape))
    param.data = torch.from_numpy(array)
    return param


def general_cat(list_of_arrays, axis):
    if isinstance(list_of_arrays[0], torch.Tensor):
        concatenated_array = torch.cat(list_of_arrays, dim=axis)
    elif isinstance(list_of_arrays[0], np.ndarray):
        concatenated_array = np.concatenate(list_of_arrays, axis=axis)
    else:
        raise NotImplementedError
    return concatenated_array


def general_sum(array, axis):
    if isinstance(array, torch.Tensor):
        summed_array = torch.sum(array, dim=axis)
    elif isinstance(array, np.ndarray):
        summed_array = np.sum(array, axis=axis)
    else:
        raise NotImplementedError
    return summed_array


def general_l2norm(array, axis, keepdims):
    if isinstance(array, torch.Tensor):
        norm = torch.norm(array, p=2, dim=axis, keepdim=keepdims)
    elif isinstance(array, np.ndarray):
        norm = np.linalg.norm(array, axis=axis, keepdims=keepdims)
    else:
        raise NotImplementedError
    return norm


def compute_normalized_solution(y, lb, ub):
    mean = (lb + ub) / 2
    size = ub - lb
    y_normalized = (y - mean) / size
    return y_normalized


def compute_denormalized_solution(y_normalized, lb, ub):
    mean = (ub + lb) / 2
    size = ub - lb
    y = y_normalized * size + mean
    return y


def solve_unconstrained(cost_vector, lb, ub):
    mean = (ub + lb) / 2
    size = ub - lb
    # indicator is -1 if less than, 0 if equal to and 1 if greater than zero
    if isinstance(cost_vector, torch.Tensor):
        indicator = (cost_vector >= 0).to(torch.float) - (cost_vector <= 0).to(torch.float)
    else:
        indicator = (cost_vector >= 0).astype(float) - (cost_vector <= 0).astype(float)
    # minus because we minimize
    y = mean - indicator * size / 2
    return y


def knapsack_round(y_denorm, constraints, knapsack_capacity):
    # for cvxpy knapsack
    weights = constraints[:, 0, :-1]
    n_batch, n_sol = y_denorm.shape
    rounded_sol = torch.zeros_like(y_denorm)
    for batch_i in range(n_batch):
        # Add indices until we hit capacity
        sol_i_sort = y_denorm[batch_i].sort(descending=True).indices
        for j in range(n_sol):
            candidate = rounded_sol[batch_i].clone()
            candidate[sol_i_sort[j]] = 1.
            if sum(weights[batch_i] * candidate) <= knapsack_capacity:
                rounded_sol[batch_i] = candidate
            else:
                break
    return rounded_sol


def load_with_default_yaml(path):
    base_path = os.getcwd()
    with open(os.path.join(base_path, path), 'r') as stream:
        param_dict = yaml.safe_load(stream)
    default_yaml = param_dict.pop('__import__', None)
    if default_yaml is not None:
        with open(os.path.join(base_path, default_yaml), 'r') as stream:
            default_dict = yaml.safe_load(stream)
        param_dict = merge(param_dict, default_dict)
    return param_dict


def merge(source, destination):
    for key, value in source.items():
        if isinstance(value, dict):
            node = destination.setdefault(key, {})
            merge(value, node)
        else:
            destination[key] = value

    return destination


def powerset(iterable):
    "list(powerset([1,2,3])) --> [(), (1,), (2,), (3,), (1,2), (1,3), (2,3), (1,2,3)]"
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s) + 1))


def sample_at_least_one_of_each(subsets, num_classes, num_samples, num_iterations=1000):
    min_number_of_subsets_containing_idx = 1
    for _ in range(num_iterations):
        good_sample = True
        samples = random.sample(subsets, num_samples)
        # ensure each class is present in the drawn subsets
        for i in range(num_classes):
            num_subsets_containing_i = len([s for s in samples if i in s])
            if num_subsets_containing_i <= min_number_of_subsets_containing_idx:
                good_sample = False
                break
        if good_sample:
            return samples
    raise ValueError(
        f'Could not draw {num_samples} samples from {len(subsets)} subsets that contain a representative of each of '
        f'the {num_classes} classes in {num_iterations} iterations.')


def print_eval_acc(metrics):
    print(f"Evaluation:: Loss: {metrics['eval_loss']:.4f}, "
          f"Perfect acc: {metrics['eval_perfect_accuracy']:.4f}")


def print_train_acc(metrics, epoch):
    print(f"Epoch: {epoch + 1:>2}, Train loss: {metrics['train_loss']:.4f}, "
          f"Perfect acc: {metrics['train_perfect_accuracy']:.4f}")


def save_dict_as_one_line_csv(dct, filename):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, 'w') as f:
        writer = csv.DictWriter(f, fieldnames=dct.keys())
        writer.writeheader()
        writer.writerow(dct)
