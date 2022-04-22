import torch
import torch.nn as nn


class MLP(torch.nn.Module):
    def __init__(self, out_features, in_features, hidden_layer_size, output_nonlinearity):
        super().__init__()
        self.fc1 = nn.Linear(in_features=in_features, out_features=hidden_layer_size)
        self.fc2 = nn.Linear(in_features=hidden_layer_size, out_features=out_features)
        self.output_nonlinearity_fn = nonlinearity_dict[output_nonlinearity]

    def forward(self, x):
        x = torch.relu(self.fc1(x.float()))
        x = self.fc2(x)
        x = self.output_nonlinearity_fn(x)
        return x


class KnapsackMLP(MLP):
    """
    Predicts normalized solution y (range [-0.5, 0.5])
    """

    def __init__(self, num_variables, reduced_embed_dim, embed_dim=4096, **kwargs):
        super().__init__(in_features=num_variables * reduced_embed_dim, out_features=num_variables,
                         output_nonlinearity='sigmoid', **kwargs)
        self.reduce_embedding_layer = nn.Linear(in_features=embed_dim, out_features=reduced_embed_dim)

    def forward(self, x):
        bs = x.shape[0]
        x = self.reduce_embedding_layer(x.float())
        x = x.reshape(shape=(bs, -1))
        x = super().forward(x)
        y_norm = x - 0.5
        return y_norm


class RandomConstraintsMLP(MLP):
    """
    Predicts normalized solution y (range [-0.5, 0.5])
    """

    def __init__(self, num_variables, **kwargs):
        super().__init__(in_features=num_variables, out_features=num_variables,
                         output_nonlinearity='sigmoid', **kwargs)

    def forward(self, x):
        x = super().forward(x)
        y_norm = x - 0.5
        return y_norm


class KnapsackExtractWeightsCostFromEmbeddingMLP(MLP):
    """
    Extracts weights and prices of vector-embedding of Knapsack instance

    @return: torch.Tensor of shape (bs, num_variables) with negative extracted prices,
             torch.Tensor of shape (bs, num_constraints, num_variables + 1) with extracted weights and negative knapsack capacity
    """

    def __init__(self, num_constraints=1, embed_dim=4096, knapsack_capacity=1.0, weight_min=0.15, weight_max=0.35,
                 cost_min=0.10, cost_max=0.45, output_nonlinearity='sigmoid', **kwargs):
        self.num_constraints = num_constraints

        self.knapsack_capacity = knapsack_capacity
        self.weight_min = weight_min
        self.weight_range = weight_max - weight_min
        self.cost_min = cost_min
        self.cost_range = cost_max - cost_min

        super().__init__(in_features=embed_dim, out_features=num_constraints + 1,
                         output_nonlinearity=output_nonlinearity, **kwargs)

    def forward(self, x):
        x = super().forward(x)
        batch_size = x.shape[0]
        cost, As = x.split([1, self.num_constraints], dim=-1)
        cost = -(self.cost_min + self.cost_range * cost[..., 0])
        As = As.transpose(1, 2)
        As = self.weight_min + self.weight_range * As
        bs = -torch.ones(batch_size, self.num_constraints).to(As.device) * self.knapsack_capacity
        constraints = torch.cat([As, bs[..., None]], dim=-1)
        return cost, constraints


nonlinearity_dict = dict(tanh=torch.tanh, relu=torch.relu, sigmoid=torch.sigmoid, identity=lambda x: x)
baseline_mlp_dict = dict(RandomConstraintsMLP=RandomConstraintsMLP, KnapsackMLP=KnapsackMLP)
