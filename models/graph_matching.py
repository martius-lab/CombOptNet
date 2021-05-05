import torch

from models.modules import StaticConstraintModule, CombOptNetModule


class StaticConstraintLearnerModule(torch.nn.Module):
    """
    Combined module for the graph matching demonstration, containing the constraint and solver module.
    Use with repo https://github.com/martius-lab/blackbox-deep-graph-matching.
    """

    def __init__(self, keypoints_per_image):
        """
        @param keypoints_per_image: int, fixed number of keypoints in src and tgt image
        """
        super().__init__()

        variable_range = dict(lb=0.0, ub=1.0)
        num_variables = keypoints_per_image * keypoints_per_image
        num_learned_constraints = 2 * keypoints_per_image
        self.static_constraint_module = StaticConstraintModule(variable_range=variable_range,
                                                               num_variables=num_variables,
                                                               num_constraints=num_learned_constraints,
                                                               normalize_constraints=True)
        self.solver_module = CombOptNetModule(variable_range=variable_range)

    def forward(self, unary_costs_list):
        """
        @param unary_costs_list: list (length bs) of torch.tensors of shape [keypoints_per_image, keypoints_per_image]

        @return: torch.tensor of shape [bs, keypoints_per_image, keypoints_per_image] with 0/1 assignment
        """
        constraints = self.static_constraint_module()

        cost = torch.stack(unary_costs_list, dim=0)
        old_shape = cost.shape
        cost_vector = cost.reshape(old_shape[0], -1)

        y_denorm_flat_batch = self.solver_module(cost_vector=cost_vector, constraints=constraints)
        y_denorm_batch = y_denorm_flat_batch.reshape(*old_shape)
        return y_denorm_batch
