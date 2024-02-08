import torch
import time
import pickle
from nets.PFSPGATModel import PFSPModel as GATModel
from nets.PFSPGNNModel import PFSPModel as GNNModel
from PFSPEnv import PFSPEnv as Env
from utils import LRnm
torch.set_float32_matmul_precision('high')


class PFSPTester:
    def __init__(self, env_params, model_params, tester_params):
        self.env_params = env_params
        self.model_params = model_params
        self.tester_params = tester_params

        self.device = torch.device('cuda')
        torch.set_default_tensor_type('torch.cuda.FloatTensor')

        if self.tester_params['architecture'] == 'gat':
            self.model = GATModel(**self.model_params)
        elif self.tester_params['architecture'] == 'gnn':
            self.model = GNNModel(**self.model_params)
        else:
            raise ValueError("Architecture not supported")

        self.d_model = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        if self.tester_params['verbose']:
            print(f"Number of parameters: {self.d_model}")

        # Load model weights
        self.model.load_state_dict(torch.load(self.tester_params['model_load_path'], map_location=self.device)['model_state_dict'])
        self.env = Env(**self.env_params)

        # Multi GPU
        if torch.cuda.device_count() > 1:
            if self.tester_params['verbose']:
                print("Using", torch.cuda.device_count(), "GPUs!")
            self.model = torch.nn.DataParallel(self.model)
        elif self.tester_params['verbose']:
            print("Using a single GPU")

        self.model.to(self.device)
        if self.tester_params['compile']:
            torch.compile(self.model)

        # Eval problems
        if not self.tester_params['taillard']: # Random instances
            # Primary eval batch
            path = f"{self.env_params['n_jobs']}_{self.env_params['n_machines']}"
            self.eval_batch = torch.load(f"data/{path}/instances_{path}.pt")
            self.eval_batch = self.eval_batch[:self.tester_params['eval_batch_size']]
            self.eval_batch = self.eval_batch.transpose(1, 2).cpu().numpy()
            # load pickle file with best known values
            with open(f"data/exact_results/{path}_results.pkl", 'rb') as f:
                a = pickle.load(f)['results']
                self.opt_values = torch.zeros(self.tester_params['eval_batch_size'])
                for i in range(self.tester_params['eval_batch_size']):
                    self.opt_values[i] = a[i][-1]

        else: # Taillard instances
            from utils import load_taillard_instances
            self.taillard_batch, names = load_taillard_instances(path=f"data/taillard/{self.tester_params['taillard_type']}")
            self.taillard_batch = self.taillard_batch.cpu().numpy()
            self.eval_batch = self.taillard_batch[7:]
            with open(f"data/taillard/results_taillard.pickle", 'rb') as f:
                results = pickle.load(f)
                path = f"DEP_{self.tester_params['taillard_type']}"
                self.opt_values = torch.tensor(results[path][7:], dtype=torch.float32)

        # LRnm
        #self.lrnm_costs, _ = LRnm(self.env_params['n_jobs'], self.env_params['n_machines'], self.eval_batch)
        #self.lrnm_avg_cost = torch.tensor(self.lrnm_costs).mean().item()
        #print(f"LRnm average cost: {self.lrnm_avg_cost:.3f}. Gap: {100*(self.lrnm_avg_cost - self.opt_values.mean().item())/self.opt_values.mean().item():.3f}%")

    # Run batch inference
    def run_batch_inference(self):
        if self.tester_params['taillard']:
            taillard = self.taillard_batch[7:]
            self.tester_params['eval_batch_size'] = 3
        else:
            taillard = None

        self.model.eval()
        batch, edges, U = self.env.reset(self.tester_params['eval_batch_size'], test_batch=self.eval_batch, taillard=taillard)

        start_time = time.time()
        _, solutions = self.model(batch, edges, U, decode_type='greedy')
        elapsed_time = time.time() - start_time
        costs = self.env.get_costs(solutions)


        gaps = 100 * ((costs - self.opt_values)/self.opt_values)
        #print(f"Inference of {self.tester_params['eval_batch_size']} instances took {elapsed_time:.3f}s - {elapsed_time/self.tester_params['eval_batch_size']:.3f}s per instance")
        print(f"Average Gap: {gaps.mean().item():.3f}% - Avg Cost {costs.mean().item():.3f} - Avg best known {self.opt_values.mean().item():.3f}")

        if self.tester_params['verbose']:
            print("------------------------Instance STATS------------------------")
            for b in range(self.tester_params['eval_batch_size']):
                print(f"Instance {b} gap {gaps[b].item():.3f}% - Cost {costs[b].item():.3f} - Avg best known {self.opt_values[b].item():.3f}")
        print("-------------------------------------------------------------")
        return gaps, costs, elapsed_time

    # Run per instance inference
    def run_instance_inference(self):
        if self.tester_params['taillard']:
            taillard = self.taillard_batch[7:]
            self.tester_params['eval_batch_size'] = 3
        else:
            taillard = None

        self.model.eval()
        start_time = time.time()
        solutions = []
        for b in range(self.tester_params['eval_batch_size']):
            if self.tester_params['taillard']:
                tai = taillard[b:b+1].copy()
            else:
                tai = None

            batch, edges, U = self.env.reset(batch_size=1, test_batch=self.eval_batch[b:b+1, :, :], taillard=tai)
            _, solution = self.model(batch, edges, U, decode_type='greedy')
            solutions.append(solution)
        solutions = torch.cat(solutions, dim=0)
        elapsed_time = time.time() - start_time
        _ = self.env.reset(self.tester_params['eval_batch_size'], test_batch=self.eval_batch, taillard=taillard)
        costs = self.env.get_costs(solutions)

        gaps = 100 * ((costs - self.opt_values)/self.opt_values)
        print(f"Inference of {self.tester_params['eval_batch_size']} instances took {elapsed_time:.3f}s - {elapsed_time/self.tester_params['eval_batch_size']:.3f}s per instance")
        print(f"Average Gap: {gaps.mean().item():.3f}% - Avg Cost {costs.mean().item():.3f} - Avg best known {self.opt_values.mean().item():.3f}")

        if self.tester_params['verbose']:
            print("------------------------Instance STATS------------------------")
            for b in range(self.tester_params['eval_batch_size']):
                print(f"Instance {b} gap {gaps[b].item():.3f}% - Cost {costs[b].item():.3f} - Avg best known {self.opt_values[b].item():.3f}")
        print("-------------------------------------------------------------")
        return gaps, costs, elapsed_time



