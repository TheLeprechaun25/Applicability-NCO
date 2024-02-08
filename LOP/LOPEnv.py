import torch
import numpy as np
torch.set_float32_matmul_precision('high')


class LOPEnv:
    def __init__(self, **env_params):
        self.env_params = env_params
        self.problem_size = env_params['problem_size']
        self.problems = None

    #@torch.compile()
    def generate_batch(self, batch_size, lolib=None):
        if lolib is None: # Random instances
            self.problems = torch.rand(batch_size, self.problem_size, self.problem_size)
        else: # LoLib instances
            self.problems = torch.zeros(batch_size, self.problem_size, self.problem_size)
            lolib_size = lolib[0].shape[0]
            for i in range(batch_size):
                # sample from a random instance of lolib
                lolib_idx = torch.randint(0, len(lolib), (1,)).item()

                random_order = torch.randperm(lolib_size)
                mesh = torch.meshgrid([random_order, random_order], indexing='ij')
                reordered_lolib = lolib[lolib_idx, :, :][mesh]
                self.problems[i, :, :] = reordered_lolib[:self.problem_size, :self.problem_size]


        ind = np.diag_indices(self.problems.shape[1])
        self.problems[:, ind[0], ind[1]] = torch.zeros(self.problem_size) # Remove diagonal elements

        # Edges
        up_tri = torch.triu(self.problems, 1) - torch.triu(self.problems.permute(0, 2, 1), 1)
        edges = up_tri - up_tri.permute(0, 2, 1)

        return edges.unsqueeze(-1)

    #@torch.compile()
    def reset(self, batch_size, test_batch=None, lolib=None):
        if test_batch is None:
            edges = self.generate_batch(batch_size, lolib)
        else:
            self.problems = test_batch[:batch_size, :, :]
            up_tri = torch.triu(self.problems, 1) - torch.triu(self.problems.permute(0, 2, 1), 1)
            edges = up_tri - up_tri.permute(0, 2, 1)
            edges = edges.unsqueeze(-1)

        return edges

    # @torch.compile()
    def get_rewards(self, solutions):
        batch_size, _ = solutions.shape
        rewards = torch.zeros([batch_size])
        for b in range(batch_size):
            mesh = torch.meshgrid([solutions[b, :], solutions[b, :]], indexing='ij')
            sol_matrix = self.problems[b, :, :][mesh]
            rewards[b] = torch.triu(sol_matrix, 1).sum()
        return rewards
