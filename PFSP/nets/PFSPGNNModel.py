import math
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from utils import compute_nodes
torch.set_float32_matmul_precision('high')


class PFSPModel(nn.Module):
    def __init__(self, **model_params):
        super(PFSPModel, self).__init__()
        self.embedding_dim = model_params['embedding_dim']
        self.tanh_clipping = model_params['tanh_clipping']
        self.n_heads = model_params['n_heads']
        self.head_dim = self.embedding_dim // self.n_heads
        assert self.embedding_dim % self.n_heads == 0
        self.temp = 1.0

        # Learned input symbols for first action
        self.W_placeholder = nn.Parameter(torch.Tensor(self.embedding_dim))
        self.W_placeholder.data.uniform_(-1, 1)

        # ENCODER
        self.init_node_embed = nn.Linear(2, self.embedding_dim)
        self.init_edge_embed = nn.Linear(1, self.embedding_dim)
        self.layers = nn.ModuleList([
            GNNLayer(**model_params)
            for _ in range(model_params['encoder_layers'])
        ])

        # DECODER
        self.project_fixed_context = nn.Linear(self.embedding_dim, self.embedding_dim, bias=False)
        self.project_node_embeddings = nn.Linear(self.embedding_dim, 3 * self.embedding_dim, bias=False)
        self.project_step_context = nn.Linear(self.embedding_dim, self.embedding_dim, bias=False)
        self.project_out = nn.Linear(self.embedding_dim, self.embedding_dim, bias=False)

    def forward(self, instance, edges, U, decode_type='greedy'):
        batch_size, n_machines, n_jobs = instance.shape
        #nodes = torch.stack([torch.ones(batch_size, n_jobs), torch.zeros(batch_size, n_jobs)], dim=-1)
        mask = torch.zeros((batch_size, 1, n_jobs))
        batch_idx = torch.arange(batch_size)
        S = np.zeros([batch_size, n_jobs], dtype=int)

        prob_list = torch.zeros(size=(batch_size, 0))
        solutions = torch.zeros((batch_size, 0), dtype=torch.long)
        for step in range(n_jobs):
            # transpose instance to have shape (batch_size, n_machines, n_jobs)
            nodes = compute_nodes(batch_size, n_jobs, n_machines, instance, S, U, step)

            # ENCODER
            h = self.init_node_embed(nodes)
            e = self.init_edge_embed(edges)

            for layer in self.layers:
                h, e = layer(h, e)
                # shape: (batch_size, problem_size, embedding_dim)

            # DECODER
            graph_embed = h.mean(1)
            graph_context = self.project_fixed_context(graph_embed)[:, None, :]
            key, value, final_key = self.project_node_embeddings(h[:, None, :, :]).chunk(3, dim=-1)
            key = self._make_heads(key, 1)
            value = self._make_heads(value, 1)
            final_key = final_key.contiguous()

            # Get log probabilities of next action
            if step == 0:
                step_context = self.W_placeholder[None, None, :].expand(batch_size, 1, self.W_placeholder.size(-1))
            else:
                step_context = h.gather(1, solutions[:, :, None].expand(batch_size, step, h.size(-1))).mean(1).unsqueeze(1)
            step_context = self.project_step_context(step_context)

            query = graph_context + step_context

            query = query.view(batch_size, 1, self.n_heads, 1, self.head_dim).permute(2, 0, 1, 3, 4)

            # Batch matrix multiplication to compute compatibilities (n_heads, batch_size, num_steps, graph_size)
            compatibility = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(query.size(-1))

            compatibility = compatibility + mask[None, :, :, None, :].expand_as(compatibility)

            # Batch matrix multiplication to compute heads (n_heads, batch_size, num_steps, val_size)
            heads = torch.matmul(F.softmax(compatibility, dim=-1), value)

            # Project to get glimpse/updated context node embedding (batch_size, num_steps, embedding_dim)
            final_query = self.project_out(heads.permute(1, 2, 3, 0, 4).contiguous().view(-1, 1, 1, self.n_heads * self.head_dim))

            # Batch matrix multiplication to compute logits (batch_size, num_steps, graph_size)
            logits = torch.matmul(final_query, final_key.transpose(-2, -1)).squeeze(-2) / math.sqrt(final_query.size(-1))

            # From the logits compute the probabilities by masking the graph, clipping, and masking visited
            score_clipped = self.tanh_clipping * torch.tanh(logits)
            score_masked = score_clipped + mask
            score = score_masked / self.temp
            probs = F.softmax(score, dim=2).squeeze(1)
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
            #nodes = nodes.clone()
            #nodes[batch_idx, selected, 0] = 0
            #nodes[batch_idx, selected, 1] = 1
            new_U = np.zeros([batch_size, n_jobs-step-1], dtype=int)
            for b in range(batch_size):
                sel_job = selected[b].item()
                S[b, step] = sel_job
                u = U[b, :]
                u = u[u != sel_job]
                new_U[b, :] = u
            U = new_U

        return prob_list, solutions

    def _make_heads(self, v, num_steps=None):
        assert num_steps is None or v.size(1) == 1 or v.size(1) == num_steps

        return (
            v.contiguous().view(v.size(0), v.size(1), v.size(2), self.n_heads, -1)
            .expand(v.size(0), v.size(1) if num_steps is None else num_steps, v.size(2), self.n_heads, -1)
            .permute(3, 0, 1, 2, 4)  # (n_heads, batch_size, num_steps, graph_size, head_dim)
        )


class GNNLayer(nn.Module):
    def __init__(self, **model_params):
        super(GNNLayer, self).__init__()
        embedding_dim = model_params['embedding_dim']

        self.U = nn.Linear(embedding_dim, embedding_dim, bias=True)
        self.V = nn.Linear(embedding_dim, embedding_dim, bias=True)
        self.A = nn.Linear(embedding_dim, embedding_dim, bias=True)
        self.B = nn.Linear(embedding_dim, embedding_dim, bias=True)
        self.C = nn.Linear(embedding_dim, embedding_dim, bias=True)

        self.norm_h = nn.BatchNorm1d(embedding_dim, affine=True, track_running_stats=False)
        self.norm_e = nn.BatchNorm1d(embedding_dim, affine=True, track_running_stats=False)

        self.mlp1 = nn.Linear(embedding_dim, model_params['ff_hidden_dim'])
        self.mlp2 = nn.Linear(model_params['ff_hidden_dim'], embedding_dim)
        self.mlp_act=NewGELU()

    def forward(self, h, e):
        batch_size, num_nodes, hidden_dim = h.shape
        h_in = h
        e_in = e

        # Linear transformations for node update
        Uh = self.U(h)  # B x V x H
        Vh = self.V(h).unsqueeze(1).expand(-1, num_nodes, -1, -1)  # B x V x V x H

        # Linear transformations for edge update and gating
        Ah = self.A(h)  # B x V x H
        Bh = self.B(h)  # B x V x H
        Ce = self.C(e)  # B x V x V x H

        # Update edge features and compute edge gates
        e = Ah.unsqueeze(1) + Bh.unsqueeze(2) + Ce  # B x V x V x H
        gates = torch.sigmoid(e)  # B x V x V x H

        # Update node features
        h = Uh + torch.sum(gates * Vh, dim=2) # B x V x H

        # Normalize node features
        h = self.norm_h(h.view(batch_size * num_nodes, hidden_dim)).view(batch_size, num_nodes, hidden_dim)

        # Normalize edge features
        e = self.norm_e(e.view(batch_size * num_nodes * num_nodes, hidden_dim)).view(batch_size, num_nodes, num_nodes, hidden_dim)

        # Apply non-linearity
        #h = F.relu(h)
        e = F.relu(e)

        # Make residual connection
        h = self.mlp2(self.mlp_act(self.mlp1(h)))

        h = h_in + h
        e = e_in + e
        return h, e


class NewGELU(nn.Module):
    """
    Implementation of the GELU activation function currently in Google BERT repo (identical to OpenAI GPT).
    Reference: Gaussian Error Linear Units (GELU) paper: https://arxiv.org/abs/1606.08415
    """
    def forward(self, x):
        return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))
