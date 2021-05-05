import cvxpy as cp
import torch
from cvxpylayers.torch import CvxpyLayer
from diffcp import SolverError

from models.comboptnet import CombOptNetModule
from utils.constraint_generation import sample_offset_constraints_numpy, compute_constraints_in_base_coordinate_system, \
    compute_normalized_constraints
from utils.utils import torch_parameter_from_numpy


class StaticConstraintModule(torch.nn.Module):
    """
    Module wrapping parameters of a learnable constraint set.
    """

    def __init__(self, num_constraints, num_variables, variable_range, normalize_constraints, learn_offsets=True,
                 learn_bs=True, **constraint_sample_params):
        """
        Initializes the learnable constraint ste
        @param num_constraints: int, cardinality of the learned constraint set
        @param num_variables: int, number of variables in the constraints
        @param variable_range: dict(lb, ub), range of variables in the ILP
        @param normalize_constraints: boolean flag, if true all constraints are normalized to unit norm
        @param learn_offsets: boolean flag, if false the initial origin offsets are not further learned
        @param learn_bs: boolean flag, if false the initial constraint bias terms are not further learned
        @param constraint_sample_params: dict, additional parameters for the sampling of initial constraint set
        """
        super().__init__()
        constraints_in_offset_system, offsets = sample_offset_constraints_numpy(variable_range=variable_range,
                                                                                num_variables=num_variables,
                                                                                num_constraints=num_constraints,
                                                                                request_offset_const=True,
                                                                                **constraint_sample_params)

        offsets = torch_parameter_from_numpy(offsets)
        self.A = torch_parameter_from_numpy(constraints_in_offset_system[..., :-1])
        b = torch_parameter_from_numpy(constraints_in_offset_system[..., -1])
        self.b = b if learn_bs else b.detach()
        self.offsets = offsets if learn_offsets else offsets.detach()

        self.normalize_constraints = normalize_constraints

    def forward(self):
        """
        @return: current set of learned constraints, with representation Ab such that A @ y + b <= 0
        """
        constraints_in_offset_system = torch.cat([self.A, self.b[..., None]], dim=-1)
        constraints = compute_constraints_in_base_coordinate_system(
            constraints_in_offset_system=constraints_in_offset_system, offsets=self.offsets)

        if self.normalize_constraints:
            constraints = compute_normalized_constraints(constraints)
        return constraints


class CvxpyModule(torch.nn.Module):
    def __init__(self, variable_range, use_entropy):
        """
        @param variable_range: dict(lb, ub), range of variables in the LP
        """
        super().__init__()
        self.variable_range = variable_range
        self.ilp_solver = CombOptNetModule(variable_range=self.variable_range)
        self.solver = None
        self.use_entropy = use_entropy

    def init_solver(self, num_variables, num_constraints, lb, ub):
        _p = cp.Parameter(num_variables)
        _A = cp.Parameter([num_constraints, num_variables])
        _b = cp.Parameter(num_constraints)
        _x = cp.Variable(num_variables)
        if self.use_entropy:
            cons = [_A @ _x + _b <= 0]
            obj = cp.Minimize(_p @ _x - sum(cp.entr(_x - lb) + cp.entr(ub - _x)))
        else:
            cons = [_A @ _x + _b <= 0, lb <= _x, _x <= ub]
            obj = cp.Minimize(_p @ _x)
        prob = cp.Problem(obj, cons)
        solver = CvxpyLayer(prob, parameters=[_A, _b, _p], variables=[_x])
        return solver

    def forward(self, cost_vector, constraints):
        """
        Forward pass of the CVXPY module running a differentiable LP solver
        @param cost_vector: torch.Tensor of shape (bs, num_variables) with batch of ILP cost vectors
        @param constraints: torch.Tensor of shape (bs, num_const, num_variables + 1) or (num_const, num_variables + 1)
                            with (potentially batch of) ILP constraints
        @return: torch.Tensor of shape (bs, num_variables) with integer values capturing the solution of the LP
        """
        A = constraints[..., :-1]
        b = constraints[..., -1]
        if self.solver is None:
            num_constraints, num_variables = A.shape[-2:]

            self.solver = self.init_solver(num_variables=num_variables, num_constraints=num_constraints,
                                           **self.variable_range)

        try:
            y, = self.solver(A, b, cost_vector)
        except SolverError as e:
            print(f'Dummy zero solution should be handled as special case.')
            y = torch.zeros_like(cost_vector).to(cost_vector.device)
        return y


def get_solver_module(**params):
    solver_name = params.pop('solver_name')
    return solver_module_dict[solver_name](**params)


solver_module_dict = dict(CombOptNet=CombOptNetModule, Cvxpy=CvxpyModule)
