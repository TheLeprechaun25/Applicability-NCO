import torch
import numpy as np
from utils import compute_edges
torch.set_float32_matmul_precision('high')


class PFSPEnv:
    def __init__(self, **env_params):
        self.env_params = env_params
        self.n_jobs = env_params['n_jobs']
        self.n_machines = env_params['n_machines']
        self.criterion = env_params['criterion'] # 'flowtime', 'makespan'
        self.device = env_params['device']
        self.problems = None

    #@torch.compile()
    def generate_batch(self, batch_size, taillard=None):
        if taillard is None: # Random instances randint from 0 to 100
            self.problems = np.random.randint(1, 100, size=(batch_size, self.n_machines, self.n_jobs))
        else: # taillard instances
            self.problems = taillard
            """lolib_size = taillard[0].shape[0]
            for i in range(batch_size):
                # TODO: sample from a random instance of taillard
                taillard_idx = torch.randint(0, len(taillard), (1,)).item()"""

        edges = compute_edges(batch_size, self.problems, 'flowtime', self.device)

        return edges

    #@torch.compile()
    def reset(self, batch_size, test_batch=None, taillard=None):
        if test_batch is None:
            edges = self.generate_batch(batch_size, taillard)
        else: # test_batch is not None
            self.problems = test_batch[:batch_size, :, :]
            edges = compute_edges(batch_size, self.problems, 'flowtime', self.device)

        U = np.arange(0, self.n_jobs, dtype=int)
        U = np.tile(U, (batch_size, 1))
        return self.problems, edges.unsqueeze(-1), U

    # @torch.compile()
    def get_costs(self, solutions):
        batch_size = solutions.shape[0]
        costs = torch.zeros(batch_size)
        for b in range(batch_size):
            if self.criterion == "makespan":
                costs[b] = self.get_makespan(solutions[b, :], self.problems[b, :, :])
            else:  # flowtime
                costs[b] = self.get_flowtime(solutions[b, :], self.problems[b, :, :])
        return costs

    # @torch.compile()
    def get_makespan(self, solution, processing_times):
        timeTable = np.zeros(self.n_machines)
        for z in range(self.n_jobs):
            job = solution[z]
            for machine in range(self.n_machines):
                proc_time = processing_times[job, machine]
                if machine == 0:
                    timeTable[machine] += proc_time
                else:
                    if timeTable[machine - 1] < timeTable[machine]:
                        timeTable[machine] = timeTable[machine] + proc_time
                    else:
                        timeTable[machine] = timeTable[machine - 1] + proc_time
        return timeTable[self.n_machines - 1]

    def get_flowtime(self, solution, processing_times):
        timeTable = np.zeros(self.n_machines)
        first_gene = solution[0]
        timeTable[0] = processing_times[0][first_gene]
        for j in range(1, self.n_machines):
            timeTable[j] = timeTable[j - 1] + processing_times[j][first_gene]

        fitness = timeTable[self.n_machines - 1]

        for z in range(1, self.n_jobs):
            job = solution[z]
            timeTable[0] += processing_times[0][job]
            prev_machine = timeTable[0]
            for machine in range(1, self.n_machines):
                timeTable[machine] = max(prev_machine, timeTable[machine]) + processing_times[machine][job]
                prev_machine = timeTable[machine]

            fitness += timeTable[self.n_machines - 1]

        return fitness
