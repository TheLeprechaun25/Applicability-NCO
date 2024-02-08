import torch
import time
from nets.LOPGATModel import LOPModel as GATModel
from nets.LOPGNNModel import LOPModel as GNNModel
from LOPEnv import LOPEnv as Env
torch.set_float32_matmul_precision('high')


class LOPTester:
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
        if not self.tester_params['lolib']: # Random instances
            self.eval_batch = torch.load(f"data/n_{self.env_params['problem_size']}/n_{self.env_params['problem_size']}_cuda.pt")
            self.eval_batch = self.eval_batch[:self.tester_params['eval_batch_size']]
            opt_values = torch.load(f"data/n_{self.env_params['problem_size']}/results_n_{self.env_params['problem_size']}.pt")
            if self.env_params['problem_size'] <= 30:
                self.opt_values = torch.tensor(opt_values['exact'][:self.tester_params['eval_batch_size']])
            else:
                self.opt_values = torch.tensor(opt_values['ma'][:self.tester_params['eval_batch_size']])
            self.opt_solutions = None

        else:  # LOLIB instances
            from utils import load_lolib_instances
            self.lolib_batch, names, norm_factors = load_lolib_instances(path=f"data/lolib/{self.tester_params['lolib_type']}")
            self.lolib_batch = self.lolib_batch.to(self.device)
            self.eval_batch = self.lolib_batch[-self.tester_params['eval_batch_size']:].clone()
            self.lolib_size = self.eval_batch.shape[1]
            opt_value_dict = torch.load(f"data/lolib/{self.tester_params['lolib_type']}/results/best_known_{self.tester_params['lolib_type']}.pt")
            self.opt_values = torch.tensor([opt_value_dict[name] for name in names[-self.tester_params['eval_batch_size']:]]) / torch.tensor(norm_factors)[-self.tester_params['eval_batch_size']:]

    # Run batch inference
    @torch.no_grad()
    def run_batch_inference(self):
        self.model.eval()

        edges = self.env.reset(self.tester_params['eval_batch_size'], test_batch=self.eval_batch.clone())
        start_time = time.time()
        _, solutions = self.model(edges, decode_type='greedy')
        elapsed_time = time.time() - start_time
        scores = self.env.get_rewards(solutions)
        gaps = 100 * ((self.opt_values - scores)/self.opt_values)
        print(f"Inference of {self.tester_params['eval_batch_size']} instances took {elapsed_time:.3f}s - {elapsed_time/self.tester_params['eval_batch_size']:.3f}s per instance")
        print(f"Average Gap: {gaps.mean().item():.3f}% +- {gaps.std(dim=0).item():.3f}% (STD) - Avg Score {scores.mean().item():.3f} - Avg best known {self.opt_values.mean().item():.3f}")

        if self.tester_params['verbose']:
            print("------------------------Instance STATS------------------------")
            for b in range(self.tester_params['eval_batch_size']):
                print(f"Instance {b} gap {gaps[b].item():.3f}% - Score {scores[b].item():.3f} - Avg best known {self.opt_values[b].item():.3f}")
        print("-------------------------------------------------------------")
        return gaps, scores, elapsed_time

    # Run per instance inference
    @torch.no_grad()
    def run_instance_inference(self):
        self.model.eval()
        start_time = time.time()
        solutions = []
        for b in range(self.tester_params['eval_batch_size']):
            edges = self.env.reset(batch_size=1, test_batch=self.eval_batch[b:b+1])
            _, solution = self.model(edges, decode_type='greedy')
            solutions.append(solution)
        solutions = torch.cat(solutions, dim=0)
        elapsed_time = time.time() - start_time
        self.env.reset(self.tester_params['eval_batch_size'], test_batch=self.eval_batch)
        scores = self.env.get_rewards(solutions)
        gaps = 100 * ((self.opt_values - scores)/self.opt_values)
        print(f"Inference of {self.tester_params['eval_batch_size']} instances took {elapsed_time:.3f}s - {elapsed_time/self.tester_params['eval_batch_size']:.3f}s per instance")
        print(f"Average Gap: {gaps.mean().item():.3f}% +- {gaps.std(dim=0).item():.3f}% (STD) - Avg Score {scores.mean().item():.3f} - Avg best known {self.opt_values.mean().item():.3f}")

        if self.tester_params['verbose']:
            print("------------------------Instance STATS------------------------")
            for b in range(self.tester_params['eval_batch_size']):
                print(f"Instance {b} gap {gaps[b].item():.3f}% - Score {scores[b].item():.3f} - Avg best known {self.opt_values[b].item():.3f}")
        print("-------------------------------------------------------------")
        return gaps, scores, elapsed_time



