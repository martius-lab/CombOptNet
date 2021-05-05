from abc import ABC, abstractmethod

import torch

from models.models import KnapsackExtractWeightsCostFromEmbeddingMLP, baseline_mlp_dict
from models.modules import get_solver_module, StaticConstraintModule, CvxpyModule, CombOptNetModule
from utils.utils import loss_from_string, optimizer_from_string, set_seed, AvgMeters, compute_metrics, \
    knapsack_round, compute_normalized_solution, compute_denormalized_solution, solve_unconstrained


def get_trainer(trainer_name, **trainer_params):
    trainer_dict = dict(MLPTrainer=MLPBaselineTrainer,
                        KnapsackConstraintLearningTrainer=KnapsackConstraintLearningTrainer,
                        RandomConstraintLearningTrainer=RandomConstraintLearningTrainer)
    return trainer_dict[trainer_name](**trainer_params)


class BaseTrainer(ABC):
    def __init__(self, train_iterator, test_iterator, use_cuda, optimizer_name, loss_name, optimizer_params, metadata,
                 model_params, seed):
        set_seed(seed)
        self.use_cuda = use_cuda
        self.device = 'cuda' if self.use_cuda else 'cpu'

        self.train_iterator = train_iterator
        self.test_iterator = test_iterator

        self.true_variable_range = metadata['variable_range']
        self.num_variables = metadata['num_variables']
        self.variable_range = self.true_variable_range

        model_parameters = self.build_model(**model_params)
        self.optimizer = optimizer_from_string(optimizer_name)(model_parameters, **optimizer_params)
        self.loss_fn = loss_from_string(loss_name)

    @abstractmethod
    def build_model(self, **model_params):
        pass

    @abstractmethod
    def calculate_loss_metrics(self, **data_params):
        pass

    def train_epoch(self):
        self.train = True
        metrics = AvgMeters()

        for i, data in enumerate(self.train_iterator):
            x, y_true_norm = [dat.to(self.device) for dat in data]
            loss, metric_dct = self.calculate_loss_metrics(x=x, y_true_norm=y_true_norm)
            metrics.update(metric_dct, n=x.size(0))

            self.optimizer.zero_grad()
            loss.backward(retain_graph=True)
            self.optimizer.step()

        results = metrics.get_averages(prefix='train_')
        return results

    def evaluate(self):
        self.train = False
        metrics = AvgMeters()

        for i, data in enumerate(self.test_iterator):
            x, y_true_norm = [dat.to(self.device) for dat in data]
            loss, metric_dct = self.calculate_loss_metrics(x=x, y_true_norm=y_true_norm)
            metrics.update(metric_dct, n=x.size(0))

        results = metrics.get_averages(prefix='eval_')
        return results


class MLPBaselineTrainer(BaseTrainer):
    def build_model(self, model_name, **model_params):
        self.model = baseline_mlp_dict[model_name](num_variables=self.num_variables, **model_params).to(
            self.device)
        return self.model.parameters()

    def calculate_loss_metrics(self, x, y_true_norm):
        y_norm = self.model(x=x)
        loss = self.loss_fn(y_norm.double(), y_true_norm)

        metrics = dict(loss=loss.item())
        y_denorm = compute_denormalized_solution(y_norm, **self.variable_range)
        y_denorm_rounded = torch.round(y_denorm)
        y_true_denorm = compute_denormalized_solution(y_true_norm, **self.true_variable_range)
        metrics.update(compute_metrics(y=y_denorm_rounded, y_true=y_true_denorm))
        return loss, metrics


class ConstraintLearningTrainerBase(BaseTrainer, ABC):
    @abstractmethod
    def forward(self, x):
        pass

    def calculate_loss_metrics(self, x, y_true_norm):
        y_denorm, y_denorm_roudned, solutions_denorm_dict, cost_vector = self.forward(x)
        y_norm = compute_normalized_solution(y_denorm, **self.variable_range)
        loss = self.loss_fn(y_norm.double(), y_true_norm)

        metrics = dict(loss=loss.item())
        y_uncon_denorm = solve_unconstrained(cost_vector=cost_vector, **self.variable_range)
        y_true_denorm = compute_denormalized_solution(y_true_norm, **self.true_variable_range)
        metrics.update(compute_metrics(y=y_denorm_roudned, y_true=y_true_denorm, y_uncon=y_uncon_denorm))
        for prefix, solution in solutions_denorm_dict.items():
            metrics.update(
                compute_metrics(y=solution, y_true=y_true_denorm, y_uncon=y_uncon_denorm, prefix=prefix + "_"))
        return loss, metrics


class RandomConstraintLearningTrainer(ConstraintLearningTrainerBase):
    def build_model(self, constraint_module_params, solver_module_params):
        self.static_constraint_module = StaticConstraintModule(variable_range=self.variable_range,
                                                               num_variables=self.num_variables,
                                                               **constraint_module_params).to(self.device)
        self.solver_module = get_solver_module(variable_range=self.variable_range,
                                               **solver_module_params).to(self.device)
        self.ilp_solver_module = CombOptNetModule(variable_range=self.variable_range).to(self.device)
        model_parameters = list(self.static_constraint_module.parameters()) + list(self.solver_module.parameters())
        return model_parameters

    def forward(self, x):
        cost_vector = x
        cost_vector = cost_vector / torch.norm(cost_vector, p=2, dim=-1, keepdim=True)
        constraints = self.static_constraint_module()

        y_denorm = self.solver_module(cost_vector=cost_vector, constraints=constraints)
        y_denorm_rounded = torch.round(y_denorm)
        solutions_dict = {}

        if not self.train and isinstance(self.solver_module, CvxpyModule):
            y_denorm_ilp = self.ilp_solver_module(cost_vector=cost_vector, constraints=constraints)
            update_dict = dict(ilp_postprocess=y_denorm_ilp)
            solutions_dict.update(update_dict)

        return y_denorm, y_denorm_rounded, solutions_dict, cost_vector


class KnapsackConstraintLearningTrainer(ConstraintLearningTrainerBase):
    def build_model(self, solver_module_params, backbone_module_params):
        self.backbone_module = KnapsackExtractWeightsCostFromEmbeddingMLP(**backbone_module_params).to(self.device)
        self.solver_module = get_solver_module(variable_range=self.variable_range,
                                               **solver_module_params).to(self.device)
        model_parameters = list(self.backbone_module.parameters()) + list(self.solver_module.parameters())
        return model_parameters

    def forward(self, x):
        cost_vector, constraints = self.backbone_module(x)
        cost_vector = cost_vector / torch.norm(cost_vector, p=2, dim=-1, keepdim=True)

        y_denorm = self.solver_module(cost_vector=cost_vector, constraints=constraints)
        if isinstance(self.solver_module, CvxpyModule):
            y_denorm_rounded = knapsack_round(y_denorm=y_denorm, constraints=constraints,
                                              knapsack_capacity=self.backbone_module.knapsack_capacity)
        else:
            y_denorm_rounded = y_denorm
        return y_denorm, y_denorm_rounded, {}, cost_vector
