import math
import torch
from torch import nn
import torch.nn.functional as F
torch.set_float32_matmul_precision('high')


class LOPModel(nn.Module):
    def __init__(self, **model_params):
        super(LOPModel, self).__init__()

        self.embedding_dim = model_params['embedding_dim']
        self.sqrt_embedding_dim = self.embedding_dim ** (1 / 2)
        self.tanh_clipping = model_params['tanh_clipping']

        # ENCODER
        self.init_node_embed = nn.Linear(2, self.embedding_dim)
        self.init_edge_embed = nn.Linear(1, self.embedding_dim)
        self.layers = nn.ModuleList([
            MHALayer(**model_params)
            for _ in range(model_params['encoder_layers'])
        ])

        # DECODER
        self.n_heads = model_params['n_heads']
        assert self.embedding_dim % self.n_heads == 0
        self.head_dim = self.embedding_dim // self.n_heads

        # Learned input symbols for first action
        self.W_placeholder = nn.Parameter(torch.Tensor(2*self.embedding_dim))
        self.W_placeholder.data.uniform_(-1, 1)

        self.W_q = nn.Linear(2*self.embedding_dim, self.embedding_dim)
        self.W_k_v = nn.Linear(self.embedding_dim, 2 * self.embedding_dim)
        self.multi_head_combine = nn.Linear(self.embedding_dim, self.embedding_dim)

    def forward(self, edges, decode_type='greedy'):
        batch_size, problem_size, _, _ = edges.shape
        nodes = torch.stack([torch.ones(batch_size, problem_size), torch.zeros(batch_size, problem_size)], dim=-1)
        mask = torch.zeros((batch_size, 1, problem_size))
        batch_idx = torch.arange(batch_size)

        prob_list = torch.zeros(size=(batch_size, 0))
        solutions = torch.zeros((batch_size, 0), dtype=torch.long)
        for step in range(problem_size - 1):
            # ENCODER
            h = self.init_node_embed(nodes)
            e = self.init_edge_embed(edges)

            for layer in self.layers:
                h = layer(h, e)
                # shape: (batch_size, problem_size, embedding_dim)

            # DECODER
            # Key and Value
            k, v = self.W_k_v(h).split(self.embedding_dim, dim=2)
            k = k.reshape(batch_size, problem_size, self.n_heads, -1).transpose(1, 2)
            v = v.reshape(batch_size, problem_size, self.n_heads, -1).transpose(1, 2)
            # shape: (batch, n_heads, problem, head_dim)

            # Query - Context
            if step == 0:
                h_context = self.W_placeholder[None, None, :].expand(batch_size, 1, self.W_placeholder.size(-1))
                #h_context = torch.cat((torch.mean(h, dim=1, keepdim=True), torch.mean(h, dim=1, keepdim=True)), dim=2)
            else:
                h_sel = torch.gather(h, 1, solutions.unsqueeze(-1).expand(-1, -1, self.embedding_dim))
                h_context = torch.cat((torch.mean(h, dim=1, keepdim=True), torch.mean(h_sel, dim=1, keepdim=True)), dim=2)

            q = self.W_q(h_context).reshape(batch_size, 1, self.n_heads, -1).transpose(1, 2)
            # shape: (batch, n_heads, 1, head_dim)

            # MHA
            score = torch.matmul(q, k.transpose(2, 3))
            score_scaled = score / torch.sqrt(torch.tensor(self.head_dim, dtype=torch.float))
            weights = nn.Softmax(dim=3)(score_scaled)
            out = torch.matmul(weights, v)
            # shape: (batch, n_heads, 1, problem) * (batch, n_heads, problem, head_dim) -> (batch, n_heads, 1, head_dim)
            out = out.transpose(1, 2).reshape(batch_size, 1, self.n_heads * self.head_dim)
            mh_atten_out = self.multi_head_combine(out)
            # shape: (batch, 1, embedding)

            #  Single-Head Attention, for probability calculation
            score = torch.matmul(mh_atten_out, h.transpose(1, 2))
            # shape: (batch, 1, embedding) * (batch, embedding, problem) -> (batch, 1, problem)
            score_scaled = score / self.sqrt_embedding_dim
            score_clipped = self.tanh_clipping * torch.tanh(score_scaled)
            score_masked = score_clipped + mask
            probs = F.softmax(score_masked, dim=2).squeeze(1)
            # shape: (batch, 1, problem)

            if decode_type == 'sample': # sampling - training
                selected = probs.multinomial(1).squeeze(dim=1)
                prob = probs[batch_idx, selected]
                prob_list = torch.cat((prob_list, prob[:, None]), dim=1)
            else:  # greedy - inference
                selected = probs.argmax(dim=1)

            # Update after selection
            solutions = torch.cat((solutions, selected[:, None]), dim=1)
            mask[batch_idx, 0, selected] = float('-inf')
            nodes = nodes.clone()
            nodes[batch_idx, selected, 0] = 0
            nodes[batch_idx, selected, 1] = 1

        # Last position given by mask: cat to solutions
        last_pos = mask.argmax(dim=2)
        solutions = torch.cat((solutions, last_pos), dim=1)
        if decode_type == 'sample':
            prob_list = torch.cat((prob_list, torch.ones((batch_size, 1))), dim=1)

        return prob_list, solutions


class MHALayer(nn.Module):
    def __init__(self, **model_params):
        super(MHALayer, self).__init__()
        self.n_heads = model_params['n_heads']
        embedding_dim = model_params['embedding_dim']
        self.head_dim = embedding_dim // self.n_heads

        self.Wq = nn.Linear(embedding_dim, self.head_dim * self.n_heads, bias=True)
        self.Wk = nn.Linear(embedding_dim, self.head_dim * self.n_heads, bias=True)
        self.Wv = nn.Linear(embedding_dim, self.head_dim * self.n_heads, bias=True)
        self.We = nn.Linear(embedding_dim, self.head_dim * self.n_heads, bias=True)

        self.attn = nn.Linear(self.head_dim, 1, bias=True)

        self.softmax = nn.Softmax(dim=1)

        self.norm1 = nn.InstanceNorm1d(embedding_dim, affine=True, track_running_stats=False)
        self.norm2 = nn.InstanceNorm1d(embedding_dim, affine=True, track_running_stats=False)

        self.mlp1 = nn.Linear(embedding_dim, model_params['ff_hidden_dim'])
        self.mlp2 = nn.Linear(model_params['ff_hidden_dim'], embedding_dim)
        self.mlp_act=NewGELU()

    def forward(self, h, edge_weights):
        batch_size, n_nodes, _ = h.shape

        # Linear transformation
        q = self.Wq(h).reshape(batch_size, n_nodes, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.Wk(h).reshape(batch_size, n_nodes, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.Wv(h).reshape(batch_size, n_nodes, self.n_heads, self.head_dim).transpose(1, 2)
        e = self.We(edge_weights).reshape(batch_size, n_nodes, n_nodes, self.n_heads, self.head_dim).permute(0, 3, 1, 2, 4).reshape(batch_size, self.n_heads, n_nodes*n_nodes, self.head_dim)

        q = q.repeat_interleave(n_nodes, dim=2)
        k = k.repeat(1, 1, n_nodes, 1)

        w = (q * k * e) / math.sqrt(self.head_dim)
        # shape: (batch, n_heads, n_nodes*n_nodes, head_dim)

        a = self.attn(w).reshape(batch_size, self.n_heads, n_nodes, n_nodes)
        # shape: (batch, n_heads, n_nodes, n_nodes)

        a = self.softmax(a)

        x = torch.matmul(a, v)
        # (batch, n_heads, n_nodes, n_nodes) * (batch, n_heads, n_nodes, head_dim) -> (batch, n_heads, n_nodes, head_dim)
        out = x.transpose(1, 2).reshape(batch_size, n_nodes, self.n_heads * self.head_dim)
        # shape: (batch, n_nodes, n_heads * head_dim)

        # Add and normalize h and out
        added = h + out
        out1 = self.norm1(added.transpose(1, 2)).transpose(1, 2)

        # MLP
        out2 = self.mlp2(self.mlp_act(self.mlp1(out1)))

        # Add and normalize out1 and out2
        added = out1 + out2
        out3 = self.norm2(added.transpose(1, 2)).transpose(1, 2)

        return out3


class NewGELU(nn.Module):
    """
    Implementation of the GELU activation function currently in Google BERT repo (identical to OpenAI GPT).
    Reference: Gaussian Error Linear Units (GELU) paper: https://arxiv.org/abs/1606.08415
    """
    def forward(self, x):
        return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))
