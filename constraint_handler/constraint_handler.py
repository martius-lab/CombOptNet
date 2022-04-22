import torch

from constraint_handler.constraint_handler_utils import sample_offset_constraints_numpy, compute_constraints_in_base_coordinate_system, \
    compute_normalized_constraints, torch_parameter_from_numpy


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
