import warnings

import gurobipy as gp
import jax.numpy as jnp
import numpy as np
import torch
from gurobipy import GRB, quicksum
from jax import grad

from comboptnet_module.comboptnet_module_utils import compute_delta_y, check_point_feasibility, softmin, \
    signed_euclidean_distance_constraint_point, tensor_to_jax, ParallelProcessing


class CombOptNetModule(torch.nn.Module):
    def __init__(self, variable_range, tau=None, clip_gradients_to_box=True, use_canonical_basis=False):
        super().__init__()
        """
        @param variable_range: dict(lb, ub), range of variables in the ILP
        @param tau: a float/np.float32/torch.float32, the value of tau for computing the constraint gradient
        @param clip_gradients_to_box: boolean flag, if true the gradients are projected into the feasible hypercube
        @param use_canonical_basis: boolean flag, if true the canonical basis is used instead of delta basis
        """
        self.solver_params = dict(tau=tau, variable_range=variable_range, clip_gradients_to_box=clip_gradients_to_box,
                                  use_canonical_basis=use_canonical_basis, parallel_processing=ParallelProcessing())
        self.solver = DifferentiableILPsolver

    def forward(self, cost_vector, constraints):
        """
        Forward pass of CombOptNet running a differentiable ILP solver
        @param cost_vector: torch.Tensor of shape (bs, num_variables) with batch of ILP cost vectors
        @param constraints: torch.Tensor of shape (bs, num_const, num_variables + 1) or (num_const, num_variables + 1)
                            with (potentially batch of) ILP constraints
        @return: torch.Tensor of shape (bs, num_variables) with integer values capturing the solution of the ILP
        """
        if len(constraints.shape) == 2:
            bs = cost_vector.shape[0]
            constraints = torch.stack(bs * [constraints])
        y, infeasibility_indicator = self.solver.apply(cost_vector, constraints, self.solver_params)
        return y


class DifferentiableILPsolver(torch.autograd.Function):
    """
    Differentiable ILP solver as a torch.Function
    """

    @staticmethod
    def forward(ctx, cost_vector, constraints, params):
        """
        Implementation of the forward pass of a batched (potentially parallelized) ILP solver.
        @param ctx: context for backpropagation
        @param cost_vector: torch.Tensor of shape (bs, num_variables) with batch of ILp cost vectors
        @param constraints: torch.Tensor of shape (bs, num_const, num_variables + 1) with batch of  ILP constraints
        @param params: a dict of additional params. Must contain:
                tau: a float/np.float32/torch.float32, the value of tau for computing the constraint gradient
                clip_gradients_to_box: boolean flag, if true the gradients are projected into the feasible hypercube
        @return: torch.Tensor of shape (bs, num_variables) with integer values capturing the solution of the ILP,
                 torch.Tensor of shape (bs) with 0/1 values, where 1 corresponds to an infeasible ILP instance
        """
        device = constraints.device
        maybe_parallelize = params['parallel_processing'].maybe_parallelize

        dynamic_args = [{"cost_vector": cost_vector, "constraints": const} for cost_vector, const in
                        zip(cost_vector.cpu().detach().numpy(), constraints.cpu().detach().numpy())]

        result = maybe_parallelize(ilp_solver, params['variable_range'], dynamic_args)
        y, infeasibility_indicator = [torch.from_numpy(np.array(res)).to(device) for res in zip(*result)]

        ctx.params = params
        ctx.save_for_backward(cost_vector, constraints, y, infeasibility_indicator)
        return y, infeasibility_indicator

    @staticmethod
    def backward(ctx, y_grad, _):
        """
        Backward pass computation.
        @param ctx: context from the forward pass
        @param y_grad: torch.Tensor of shape (bs, num_variables) describing the incoming gradient dy for y
        @return: torch.Tensor of shape (bs, num_variables) gradient dL / cost_vector
                 torch.Tensor of shape (bs, num_constraints, num_variables + 1) gradient dL / constraints
        """
        cost_vector, constraints, y, infeasibility_indicator = ctx.saved_tensors
        assert y.shape == y_grad.shape

        grad_mismatch_function = grad(mismatch_function, argnums=[0, 1])
        grad_cost_vector, grad_constraints = grad_mismatch_function(tensor_to_jax(cost_vector),
                                                                    tensor_to_jax(constraints),
                                                                    tensor_to_jax(y),
                                                                    tensor_to_jax(y_grad),
                                                                    tensor_to_jax(infeasibility_indicator),
                                                                    variable_range=ctx.params['variable_range'],
                                                                    clip_gradients_to_box=ctx.params[
                                                                        'clip_gradients_to_box'],
                                                                    use_canonical_basis=ctx.params[
                                                                        'use_canonical_basis'],
                                                                    tau=ctx.params['tau'])

        cost_vector_grad = torch.from_numpy(np.array(grad_cost_vector)).to(y_grad.device)
        constraints_gradient = torch.from_numpy(np.array(grad_constraints)).to(y_grad.device)
        return cost_vector_grad, constraints_gradient, None


def ilp_solver(cost_vector, constraints, lb, ub):
    """
    ILP solver using Gurobi. Computes the solution of a single integer linear program
    y* = argmin_y (c * y) subject to A @ y + b <= 0, y integer, lb <= y <= ub

    @param cost_vector: np.array of shape (num_variables) with cost vector of the ILP
    @param constraints: np.array of shape (num_const, num_variables + 1) with constraints of the ILP
    @param lb: float, lower bound of variables
    @param ub: float, upper bound of variables
    @return: np.array of shape (num_variables) with integer values capturing the solution of the ILP,
             boolean flag, where true corresponds to an infeasible ILP instance
    """
    A, b = constraints[:, :-1], constraints[:, -1]
    num_constraints, num_variables = A.shape

    model = gp.Model("mip1")
    model.setParam('OutputFlag', 0)
    model.setParam("Threads", 1)

    variables = [model.addVar(lb=lb, ub=ub, vtype=GRB.INTEGER, name='v' + str(i)) for i in range(num_variables)]
    model.setObjective(quicksum(c * var for c, var in zip(cost_vector, variables)), GRB.MINIMIZE)
    for a, _b in zip(A, b):
        model.addConstr(quicksum(c * var for c, var in zip(a, variables)) + _b <= 0)
    model.optimize()
    try:
        y = np.array([v.x for v in model.getVars()])
        infeasible = False
    except AttributeError:
        warnings.warn(f'Infeasible ILP encountered. Dummy solution should be handled as special case.')
        y = np.zeros_like(cost_vector)
        infeasible = True
    return y, infeasible


def mismatch_function(cost_vector, constraints, y, y_grad, infeasibility_indicator, variable_range, tau,
                      clip_gradients_to_box, use_canonical_basis, average_solution=True, use_cost_mismatch=True):
    """
    Computes the combined mismatch function for cost vectors and constraints P_(dy)(A, b, c) = P_(dy)(A, b) + P_(dy)(c)
    P_(dy)(A, b) = sum_k(lambda_k * P_(delta_k)(A, b)),
    P_(dy)(c) = sum_k(lambda_k * P_(delta_k)(c)),
    where delta_k = y'_k - y

    @param cost_vector: jnp.array of shape (bs, num_variables) with batch of ILP cost vectors
    @param constraints: jnp.array of shape (bs, num_const, num_variables + 1) with batch of ILP constraints
    @param y: jnp.array of shape (bs, num_variables) with batch of ILP solutions
    @param y_grad: jnp.array of shape (bs, num_variables) with batch of incoming gradients for y
    @param infeasibility_indicator: jnp.array of shape (bs) with batch of indicators whether ILP has feasible solution
    @param variable_range: dict(lb, ub), range of variables in the ILP
    @param tau: a float/np.float32/torch.float32, the value of tau for computing the constraint gradient
    @param clip_gradients_to_box: boolean flag, if true the gradients are projected into the feasible hypercube
    @param use_canonical_basis: boolean flag, if true the canonical basis is used instead of delta basis

    @return: jnp.array scalar with value of mismatch function
    """
    num_constraints = constraints.shape[1]
    if num_constraints > 1 and tau is None:
        raise ValueError('If more than one constraint is used the parameter tau needs to be specified.')
    if num_constraints == 1 and tau is not None:
        warnings.warn('The specified parameter tau has no influence as only a single constraint is used.')

    delta_y, lambdas = compute_delta_y(y=y, y_grad=y_grad, clip_gradients_to_box=clip_gradients_to_box,
                                       use_canonical_basis=use_canonical_basis, **variable_range)
    y = y[:, None, :]
    y_prime = y + delta_y
    cost_vector = cost_vector[:, None, :]
    constraints = constraints[:, None, :, :]

    y_prime_feasible_constraints, y_prime_inside_box = check_point_feasibility(point=y_prime, constraints=constraints,
                                                                               **variable_range)
    feasibility_indicator = 1.0 - infeasibility_indicator
    correct_solution_indicator = jnp.all(jnp.isclose(y, y_prime), axis=-1)
    # solution can only be correct if we also have a feasible problem
    # (fixes case in which infeasible solution dummy matches ground truth)
    correct_solution_indicator *= feasibility_indicator[:, None]
    incorrect_solution_indicator = 1.0 - correct_solution_indicator

    constraints_mismatch = compute_constraints_mismatch(constraints=constraints, y=y, y_prime=y_prime,
                                                        y_prime_feasible_constraints=y_prime_feasible_constraints,
                                                        y_prime_inside_box=y_prime_inside_box, tau=tau,
                                                        incorrect_solution_indicator=incorrect_solution_indicator)
    cost_mismatch = compute_cost_mismatch(cost_vector=cost_vector, y=y, y_prime=y_prime,
                                          y_prime_feasible_constraints=y_prime_feasible_constraints,
                                          y_prime_inside_box=y_prime_inside_box)
    total_mismatch = constraints_mismatch
    if use_cost_mismatch:
        total_mismatch += cost_mismatch

    total_mismatch = jnp.mean(total_mismatch * lambdas, axis=-1)  # scale mismatch functions of sparse y' with lambda
    if average_solution:
        total_mismatch = jnp.mean(total_mismatch)
    return total_mismatch


def compute_cost_mismatch(cost_vector, y, y_prime, y_prime_feasible_constraints, y_prime_inside_box):
    """
    Computes the mismatch function for cost vectors P_(delta_k)(c), where delta_k = y'_k - y
    """
    c_diff = jnp.sum(cost_vector * y_prime, axis=-1) - jnp.sum(cost_vector * y, axis=-1)
    cost_mismatch = jnp.maximum(c_diff, 0.0)

    # case distinction in paper: if y' is (constraint-)infeasible or outside of hypercube cost mismatch function is zero
    cost_mismatch = cost_mismatch * y_prime_inside_box * y_prime_feasible_constraints
    return cost_mismatch


def compute_constraints_mismatch(constraints, y, y_prime, y_prime_inside_box, y_prime_feasible_constraints,
                                 incorrect_solution_indicator, tau):
    """
    Computes the mismatch function for constraints P_(delta_k)(A, b), where delta_k = y'_k - y
    """
    # case 1 in paper: if y' is (constraint-)feasible, y' is inside the hypercube and y != y'
    constraints_mismatch_feasible = compute_constraints_mismatch_feasible(constraints=constraints, y=y, tau=tau)
    constraints_mismatch_feasible *= y_prime_feasible_constraints

    # case 2 in paper: if y' is (constraint-)infeasible, y' is inside the hypercube and y != y'
    constraints_mismatch_infeasible = compute_constraints_mismatch_infeasible(constraints=constraints, y_prime=y_prime)
    constraints_mismatch_infeasible *= (1.0 - y_prime_feasible_constraints)

    constraints_mismatch = constraints_mismatch_feasible + constraints_mismatch_infeasible

    # case 3 in paper: if y prime is outside the hypercube or y = y' constraint mismatch function is zero
    constraints_mismatch = constraints_mismatch * y_prime_inside_box * incorrect_solution_indicator
    return constraints_mismatch


def compute_constraints_mismatch_feasible(constraints, y, tau):
    distance_y_const = signed_euclidean_distance_constraint_point(constraints=constraints, point=y)
    constraints_mismatch_feasible = jnp.maximum(-distance_y_const, 0.0)
    constraints_mismatch_feasible = softmin(constraints_mismatch_feasible, tau=tau, axis=-1)
    return constraints_mismatch_feasible


def compute_constraints_mismatch_infeasible(constraints, y_prime):
    distance_y_prime_const = signed_euclidean_distance_constraint_point(constraints=constraints, point=y_prime)
    constraints_mismatch_infeasible = jnp.maximum(distance_y_prime_const, 0.0)
    constraints_mismatch_infeasible = jnp.sum(constraints_mismatch_infeasible, axis=-1)
    return constraints_mismatch_infeasible
