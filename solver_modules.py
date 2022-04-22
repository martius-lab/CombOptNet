import cvxpy as cp
import torch
from comboptnet_module import CombOptNetModule
from cvxpylayers.torch import CvxpyLayer
from diffcp import SolverError


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
    solver_module_dict = dict(CombOptNet=CombOptNetModule, Cvxpy=CvxpyModule)
    return solver_module_dict[solver_name](**params)
