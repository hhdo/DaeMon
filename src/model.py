from collections.abc import Sequence
import torch
from torch import nn, autograd
from torch.nn import functional as F
import layers

class DaeMon(nn.Module):
    def __init__(self, input_dim, hidden_dims, num_nodes, num_relation, message_func="distmult", aggregate_func="pna",
                 short_cut=False, layer_norm=False, activation="relu", num_mlp_layer=2,
                 history_len=10, mem_pass='mean'):
        super(DaeMon,self).__init__()

        self.dims = [input_dim] + list(hidden_dims)
        self.num_nodes = num_nodes
        self.num_relation = num_relation *2 # reverse rel type should be added
        self.short_cut = short_cut  # whether to use residual connections between layers
        
        self.history_len = history_len
        self.mem_pass = mem_pass

        # additional relation embedding which serves for the PAU
        # each layer has its own learnable relations matrix
        self.query = nn.Embedding(self.num_relation, input_dim)

        self.layers = nn.ModuleList()
        for i in range(len(self.dims) - 1): # num of hidden layers
            self.layers.append(layers.PathAggNet(self.dims[i], self.dims[i + 1], self.num_relation,
                                                                self.dims[0], message_func, aggregate_func, layer_norm,
                                                                activation))

        self.feature_dim = hidden_dims[-1] + input_dim

        if self.mem_pass == 'tawaregate':
            self.gate_weight = nn.Linear(input_dim, input_dim)

        self.mlp = nn.Sequential()
        mlp = []
        for i in range(num_mlp_layer - 1):
            mlp.append(nn.Linear(self.feature_dim, self.feature_dim))
            mlp.append(nn.ReLU())
        mlp.append(nn.Linear(self.feature_dim, 1))
        self.mlp = nn.Sequential(*mlp)

    def negative_sample_to_tail(self, h_index, t_index, r_index):
        # convert p(h | t, r) to p(t' | h', r')
        # h' = t, r' = r^{-1}, t' = h
        is_t_neg = (h_index == h_index[:, [0]]).all(dim=-1, keepdim=True)
        new_h_index = torch.where(is_t_neg, h_index, t_index)
        new_t_index = torch.where(is_t_neg, t_index, h_index)
        new_r_index = torch.where(is_t_neg, r_index, r_index + self.num_relation // 2)
        return new_h_index, new_t_index, new_r_index

    def memory_update(self, each_snap_g, h_index, query, first_flag, last_stat):
        batch_size = len(h_index)
        
        index = h_index.unsqueeze(-1).expand_as(query)

        # initialize all pairs states as zeros in memory
        initial_stat = torch.zeros(batch_size, each_snap_g.num_nodes(), self.dims[0], device=h_index.device)
        
        # different initial methods (variants corresponding to ablation study)
        if self.mem_pass == 'meanpool':
            # MPS/PMMP
            meanpool = last_stat.mean(dim = 1, keepdim=True)
            initial_stat.scatter_add_(1, index.unsqueeze(1), query.unsqueeze(1)*first_flag+(1-first_flag)*meanpool) 
        elif self.mem_pass == 'meanbound':
            # MPS/IPMM
            initial_stat.scatter_add_(1, index.unsqueeze(1), query.unsqueeze(1)) 
            initial_stat = first_flag*initial_stat + (1-first_flag)*torch.div((initial_stat + last_stat), 2.0) 
        elif self.mem_pass == 'tawaregate':
            # Time-aware GATE
            initial_stat.scatter_add_(1, index.unsqueeze(1), query.unsqueeze(1))
            gate_weight = torch.sigmoid(self.gate_weight(last_stat))
            initial_stat = first_flag*initial_stat + (1-first_flag)*(initial_stat*gate_weight + (1-gate_weight)*last_stat)
        elif self.mem_pass == 'outputmean':
            # MPS/MMP
            initial_stat.scatter_add_(1, index.unsqueeze(1), query.unsqueeze(1))
        else:
            raise NotImplementedError

        size = (each_snap_g.num_nodes(), each_snap_g.num_nodes())
        layer_input = initial_stat
        
        for layer in self.layers:
            # w-layers iteration
            hidden = layer(layer_input, query, initial_stat, torch.stack(each_snap_g.edges()), each_snap_g.edata['type'], size, edge_weight = None)
            if self.short_cut and hidden.shape == layer_input.shape:
                # shortcut setting
                hidden = hidden + layer_input
            layer_input = hidden

        return hidden

    def forward(self, history_g_list, query_triple):
        h_index, r_index, t_index = query_triple.unbind(-1)
        shape = h_index.shape
        batch_size = shape[0]

        # turn all triples in a batch into a tail prediction mode
        h_index, t_index, r_index = self.negative_sample_to_tail(h_index, t_index, r_index)
        assert (h_index[:, [0]] == h_index).all()
        assert (r_index[:, [0]] == r_index).all()

        # initialize queries (relation types of the given triples)
        query = self.query(r_index[:, 0])

        output = torch.zeros(batch_size, history_g_list[-1].num_nodes(), self.dims[0], device=h_index.device)
        if self.mem_pass == 'outputmean':
            # MPS/MMP
            feature_mean = torch.zeros(batch_size, history_g_list[-1].num_nodes(), self.dims[0], device=h_index.device)

        for ind, each_snap_g in enumerate(history_g_list):
            # adaptively update memory over timeline
            output = self.memory_update(each_snap_g, h_index[:, 0], query, first_flag=(ind==0), last_stat = output)
            if self.mem_pass == 'outputmean':
                # MPS/MMP
                feature_mean = torch.div((feature_mean + output), 2.0) if ind != 0 else output+feature_mean
            
        feature = output if self.mem_pass != 'outputmean' else feature_mean

        # cat original query relation embeddings for enhancing the query processing
        origin_r_emb = query.unsqueeze(1).expand(-1, history_g_list[-1].num_nodes(), -1)
        final_feature = torch.cat([feature, origin_r_emb], dim=-1)
     
        index = t_index.unsqueeze(-1).expand(-1, -1, final_feature.shape[-1])
        # extract representations of tail entities from the updated momory
        feature_t = final_feature.gather(1, index)

        # (batch_size, num_negative + 1, dim) -> (batch_size, num_negative + 1)
        score = self.mlp(feature_t).squeeze(-1)

        return score.view(shape)

    def get_loss(self, args, pred):
        
        target = torch.zeros_like(pred)
        target[:, 0] = 1
        loss = F.binary_cross_entropy_with_logits(pred, target, reduction="none")
        neg_weight = torch.ones_like(pred)
        if args.adversarial_temperature > 0:
            with torch.no_grad():
                neg_weight[:, 1:] = F.softmax(pred[:, 1:] / args.adversarial_temperature, dim=-1)
        else:
            neg_weight[:, 1:] = 1 / args.negative_num
        loss = (loss * neg_weight).sum(dim=-1) / neg_weight.sum(dim=-1)
        loss = loss.mean()

        tmp = torch.mm(self.query.weight, self.query.weight.permute(1, 0))
        orthogonal_regularizer = torch.norm(tmp - 1 * torch.diag(torch.ones(self.num_relation, device=pred.device)), 2)

        loss = loss + orthogonal_regularizer
        return loss

        



        

