import torch
from torch import nn
from torch.nn import functional as F
from torch_scatter import scatter

from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import degree


class PathAggNet(MessagePassing):

    def __init__(self, input_dim, output_dim, num_relation, query_input_dim, message_func="distmult",
                 aggregate_func="pna", layer_norm=False, activation="relu"):
        super(PathAggNet, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_relation = num_relation
        self.query_input_dim = query_input_dim
        self.message_func = message_func
        self.aggregate_func = aggregate_func

        if layer_norm:
            self.layer_norm = nn.LayerNorm(output_dim)
        else:
            self.layer_norm = None
        if isinstance(activation, str):
            self.activation = getattr(F, activation)
        else:
            self.activation = activation

        if self.aggregate_func == "pna":
            self.linear = nn.Linear(input_dim * 13, output_dim)
        else:
            self.linear = nn.Linear(input_dim * 2, output_dim)

        self.relation_linear = nn.Linear(query_input_dim, num_relation * input_dim)


    def forward(self, input, query, initial_stat, edge_index, edge_type, size, edge_weight=None):
        batch_size = len(query)
        # layer-specific relation features as a projection of query r embeddings
        relation = self.relation_linear(query).view(batch_size, self.num_relation, self.input_dim)
       
        if edge_weight is None:
            edge_weight = torch.ones(len(edge_type), device=input.device)

        output = self.propagate(input=input, relation=relation, initial_stat=initial_stat, edge_index=edge_index,
                                edge_type=edge_type, size=size, edge_weight=edge_weight)
        
        return output

    def propagate(self, edge_index, size=None, **kwargs):
        return super(PathAggNet, self).propagate(edge_index, size, **kwargs)

    def message(self, input_j, relation, initial_stat, edge_type): 
        relation_j = relation.index_select(self.node_dim, edge_type) 

        if self.message_func == "transe":
            message = input_j + relation_j
        elif self.message_func == "distmult":
            message = input_j * relation_j
        elif self.message_func == "rotate":
            x_j_re, x_j_im = input_j.chunk(2, dim=-1)
            r_j_re, r_j_im = relation_j.chunk(2, dim=-1)
            message_re = x_j_re * r_j_re - x_j_im * r_j_im
            message_im = x_j_re * r_j_im + x_j_im * r_j_re
            message = torch.cat([message_re, message_im], dim=-1)
        else:
            raise ValueError("Unknown message function `%s`" % self.message_func)

        # augment messages with the initial_stat
        message = torch.cat([message, initial_stat], dim=self.node_dim)  # (batch_size, num_edges + num_nodes, input_dim)

        return message

    def aggregate(self, input, edge_weight, index, dim_size):
        index = torch.cat([index, torch.arange(dim_size, device=input.device)])
        edge_weight = torch.cat([edge_weight, torch.ones(dim_size, device=input.device)])
        shape = [1] * input.ndim
        shape[self.node_dim] = -1
        edge_weight = edge_weight.view(shape)

        if self.aggregate_func == "pna":
            eps = 1e-6
            mean = scatter(input * edge_weight, index, dim=self.node_dim, dim_size=dim_size, reduce="mean")
            sq_mean = scatter(input ** 2 * edge_weight, index, dim=self.node_dim, dim_size=dim_size, reduce="mean")
            max = scatter(input * edge_weight, index, dim=self.node_dim, dim_size=dim_size, reduce="max")
            min = scatter(input * edge_weight, index, dim=self.node_dim, dim_size=dim_size, reduce="min")
            std = (sq_mean - mean ** 2).clamp(min=eps).sqrt()
            features = torch.cat([mean.unsqueeze(-1), max.unsqueeze(-1), min.unsqueeze(-1), std.unsqueeze(-1)], dim=-1)
            features = features.flatten(-2)
            degree_out = degree(index, dim_size).unsqueeze(0).unsqueeze(-1)
            scale = degree_out.log()
            scale = scale / scale.mean()
            scales = torch.cat([torch.ones_like(scale), scale, 1 / scale.clamp(min=1e-2)], dim=-1)
            output = (features.unsqueeze(-1) * scales.unsqueeze(-2)).flatten(-2)
        else:
            output = scatter(input * edge_weight, index, dim=self.node_dim, dim_size=dim_size,
                             reduce=self.aggregate_func)

        return output

    def update(self, update, input):
        output = self.linear(torch.cat([input, update], dim=-1))
        if self.layer_norm:
            output = self.layer_norm(output)
        if self.activation:
            output = self.activation(output)
        return output
