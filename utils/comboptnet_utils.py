import jax.numpy as jnp

epsilon_constant = 1e-8


def logsumexp(array, tau, axis):
    exp = jnp.exp(array / tau)
    sumexp = jnp.sum(exp, axis=axis)
    offset = jnp.log(array.shape[axis])
    logsumexp = (jnp.log(sumexp) - offset) * tau
    return logsumexp


def softmin(array, axis, tau):
    if tau is None:
        tau = 0.0
    assert tau >= 0.0
    if jnp.isclose(tau, 0.0):
        return jnp.min(array, axis=axis)
    else:
        return -logsumexp(-array, tau=tau, axis=axis)


def compute_delta_y(y, y_grad, lb, ub, clip_gradients_to_box, use_canonical_basis):
    """
    Computes integer basis delta_k from dy such that dy = sum_k(lambda_k * delta_k)
    with delta_k the signed indicator vector of the first k dominant directions of dy

    @param y: jnp.array of shape (bs, num_variables) with batch of ILP solutions
    @param y_grad: jnp.array of shape (bs, num_variables) with batch of incoming gradients dy for y
    @param lb: float, lower bound of variables in ILP
    @param ub: float, upper bound of variables in ILP
    @param clip_gradients_to_box: boolean flag, if true the gradients are projected into the feasible hypercube
    @param use_canonical_basis: boolean flag, if true canonical basis delta_k = e_k is used

    @return: jnp.array of shape (bs, num_variables, num_variables) with basis delta_k,
             jnp.array of shape (bs, num_variables) with (positive) weightings lambda_k of the delta_k
    """
    bs, dim = y.shape
    y_grad = -y_grad

    if clip_gradients_to_box:
        y_test = y + jnp.sign(y_grad)
        inside_box_indicator = jnp.greater_equal(y_test, lb) * jnp.less_equal(y_test, ub)
        y_grad = y_grad * inside_box_indicator

    if use_canonical_basis:
        delta_y = jnp.sign(y_grad)[:, None, :] * jnp.eye(dim)[None, :, :]
        lambdas = jnp.abs(y_grad)
    else:
        sort_indices = jnp.argsort(jnp.abs(y_grad))
        ranks_of_abs_y_grad = jnp.argsort(sort_indices)

        triangular_matrix = jnp.triu(jnp.ones((dim, dim)))
        permuted_triangular_matrix = jnp.take_along_axis(triangular_matrix[None, :, :],
                                                         ranks_of_abs_y_grad[:, None, :],
                                                         axis=-1)

        delta_y = permuted_triangular_matrix * jnp.sign(y_grad + epsilon_constant)[:, None, :]

        abs_grad_sorted = jnp.take_along_axis(jnp.abs(y_grad), sort_indices, axis=-1)
        sorted_grad_zero_append = jnp.concatenate((jnp.zeros((bs, 1)), abs_grad_sorted), axis=-1)
        lambdas = sorted_grad_zero_append[:, 1:] - sorted_grad_zero_append[:, :-1]
    return delta_y, lambdas


def signed_euclidean_distance_constraint_point(constraints, point):
    constraint_lhs = jnp.sum(constraints[..., :-1] * point[..., None, :], axis=-1) + constraints[..., -1]
    distance_point_const = constraint_lhs / (jnp.linalg.norm(constraints[..., :-1], axis=-1) + epsilon_constant)
    return distance_point_const


def check_point_feasibility(point, constraints, lb, ub):
    distance_point_const = signed_euclidean_distance_constraint_point(constraints=constraints, point=point)
    feasibility_indicator_constraints = jnp.all(jnp.less_equal(distance_point_const, 0.0), axis=-1)
    feasibility_indicator_lb = jnp.all(jnp.greater_equal(point, lb), axis=-1)
    feasibility_indicator_ub = jnp.all(jnp.less_equal(point, ub), axis=-1)
    y_prime_inside_box = feasibility_indicator_lb * feasibility_indicator_ub
    return feasibility_indicator_constraints, y_prime_inside_box


def tensor_to_jax(tensor):
    return jnp.array(tensor.cpu().detach().numpy())
