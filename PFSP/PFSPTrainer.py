import torch
import time
import pickle
from nets.PFSPGATModel import PFSPModel as GATModel
from nets.PFSPGNNModel import PFSPModel as GNNModel
from PFSPEnv import PFSPEnv as Env
from utils import clip_grad_norms, configure_optimizers, LRnm
torch.set_float32_matmul_precision('high')


class PFSPTrainer:
    def __init__(self, env_params, model_params, optimizer_params, trainer_params):
        self.env_params = env_params
        self.model_params = model_params
        self.optimizer_params = optimizer_params
        self.trainer_params = trainer_params

        self.device = torch.device('cuda')
        torch.set_default_tensor_type('torch.cuda.FloatTensor')

        if self.trainer_params['architecture'] == 'gat':
            self.model = GATModel(**self.model_params)
        elif self.trainer_params['architecture'] == 'gnn':
            self.model = GNNModel(**self.model_params)
        else:
            raise ValueError("Architecture not supported")

        if trainer_params['optimizer'] == 'adam':
            self.optimizer = torch.optim.Adam([{'params': self.model.parameters(), 'lr': self.optimizer_params['lr']}])
        elif trainer_params['optimizer'] == 'adamw':
            self.optimizer = configure_optimizers(self.model, self.optimizer_params)
        else:
            raise ValueError("Optimizer not supported")

        self.d_model = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        if trainer_params['verbose']:
            print(f"Number of parameters: {self.d_model}")

        # Restore model weights
        if trainer_params['model_load']['enable']:
            checkpoint = torch.load(trainer_params['model_load']['path'], map_location=self.device)
            # check multiGPU
            if isinstance(self.model, torch.nn.DataParallel):
                self.model.module.load_state_dict(checkpoint['model_state_dict'])
            else:
                self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        # Multi GPU
        if torch.cuda.device_count() > 1:
            if trainer_params['verbose']:
                print("Using", torch.cuda.device_count(), "GPUs!")
            self.model = torch.nn.DataParallel(self.model)
        elif trainer_params['verbose']:
            print("Using a single GPU")

        self.model.to(self.device)
        if self.trainer_params['compile']:
            torch.compile(self.model)

        self.env = Env(**self.env_params)

        # Eval problems
        if not self.trainer_params['taillard']: # Random instances
            # Primary eval batch
            path = f"{self.env_params['n_jobs']}_{self.env_params['n_machines']}"
            self.eval_batch = torch.load(f"data/{path}/instances_{path}.pt")
            self.eval_batch = self.eval_batch[:self.trainer_params['eval_batch_size']]
            self.eval_batch = self.eval_batch.transpose(1, 2).cpu().numpy()
            # load pickle file with best known values
            with open(f"data/exact_results/{path}_results.pkl", 'rb') as f:
                a = pickle.load(f)['results']
                self.opt_values = torch.zeros(self.trainer_params['eval_batch_size'])
                for i in range(self.trainer_params['eval_batch_size']):
                    self.opt_values[i] = a[i][-1]

            # Secondary eval batches
            if self.trainer_params['second_eval']:
                self.secondary_eval_sizes = [[20, 5], [20, 10], [20, 20], [50, 10], [100, 10], [200, 20]]
                self.secondary_batch_sizes = [64, 64, 64, 64, 1, 1]
                # remove the problem size from the list
                if self.env_params['problem_size'] in self.secondary_eval_sizes:
                    self.secondary_eval_sizes.remove(self.env_params['problem_size'])
                self.secondary_eval_batches = []
                self.secondary_eval_opt_values = []
                for sizes, b_size in zip(self.secondary_eval_sizes, self.secondary_batch_sizes):
                    path = f"{sizes[0]}_{sizes[1]}"
                    eval_batch = torch.load(f"data/{path}/instances_{path}.pt")
                    eval_batch = eval_batch[:b_size]
                    eval_batch = eval_batch.transpose(1, 2).cpu().numpy()
                    self.secondary_eval_batches.append(eval_batch)
                    with open(f"data/exact_results/{path}_results.pkl", 'rb') as f:
                        a = pickle.load(f)['results']
                        opt_values = torch.zeros(b_size)
                        for i in range(b_size):
                            opt_values[i] = a[i][-1]
                    self.secondary_eval_opt_values.append(opt_values)
        else: # Taillard instances
            from utils import load_taillard_instances
            self.taillard_batch, names = load_taillard_instances(path=f"data/taillard/{self.trainer_params['taillard_type']}")
            self.taillard_batch = self.taillard_batch.cpu().numpy()
            self.eval_batch = self.taillard_batch[7:]
            with open(f"data/taillard/results_taillard.pickle", 'rb') as f:
                results = pickle.load(f)
                path = f"DEP_{self.trainer_params['taillard_type']}"
                self.opt_values = torch.tensor(results[path][7:])
        # LRnm
        self.lrnm_costs, _ = LRnm(self.env_params['n_jobs'], self.env_params['n_machines'], self.eval_batch)
        self.lrnm_costs = torch.tensor(self.lrnm_costs).mean().item()

    def run(self):
        non_improving_epochs = 0
        best_gap = 1e9
        for epoch in range(1, self.trainer_params['epochs']+1):
            start_time = time.time()
            print(f"\nEpoch {epoch}/{self.trainer_params['epochs']}")
            # TRAINING
            for episode in range(self.trainer_params['train_episodes']):
                self.env.n_jobs = self.env_params['n_jobs']
                self.env.n_machines = self.env_params['n_machines']
                self.train_one_batch()
            print(f"Epoch duration: {time.time() - start_time:.1f}s")

            # EVALUATION AND SAVING
            print("Evaluating model...")
            print(f"Mean cost for LRnm heuristic (flowtime): {self.lrnm_costs}.")
            # Evaluate in primary problem size
            scores = self.evaluate_model(test_batch=self.eval_batch)
            primary_gap = 100 * ((scores - self.opt_values) / self.opt_values).mean().item()
            print(f"{self.env_params['n_jobs']} jobs, {self.env_params['n_machines']} machines. Score: {scores.mean().item():.3f} {primary_gap:.3f}%")

            # Evaluate in secondary problem sizes
            secondary_gaps = []
            if self.trainer_params['second_eval']:
                for size, eval_batch, opt_values in zip(self.secondary_eval_sizes, self.secondary_eval_batches, self.secondary_eval_opt_values):
                    self.env.n_jobs = size[0]
                    self.env.n_machines = size[1]
                    scores = self.evaluate_model(test_batch=eval_batch)
                    gap = 100 * ((scores - opt_values) / opt_values).mean().item()
                    secondary_gaps.append(gap)
                    print(f"{size[0]} jobs, {size[1]} machines. Score {scores.mean().item():.3f} {gap:.3f}%")

            # Save model
            if isinstance(self.model, torch.nn.DataParallel):
                model_state_dict = self.model.module.state_dict()
            else:
                model_state_dict = self.model.state_dict()

            if self.trainer_params['taillard']:
                if self.trainer_params['model_load']['enable']:
                    path = f"results/saved_models/int_taillard/{self.trainer_params['taillard_type']}/{self.trainer_params['execution_name']}-{self.trainer_params['taillard_type']}-Epoch{epoch}-Gap{primary_gap:.2f}"
                else:
                    path = f"results/saved_models/taillard/{self.trainer_params['taillard_type']}/{self.trainer_params['execution_name']}-{self.trainer_params['taillard_type']}-Epoch{epoch}-Gap{primary_gap:.2f}"
            else:
                path = f"results/saved_models/{self.env_params['n_jobs']}_{self.env_params['n_machines']}/{self.trainer_params['execution_name']}-{self.env_params['n_jobs']}_{self.env_params['n_machines']}-Epoch{epoch}-Gap{primary_gap:.2f}"
                if self.trainer_params['second_eval']:
                    for sizes, gap in zip(self.secondary_eval_sizes, secondary_gaps):
                        path += f"-{sizes[0]}_{sizes[1]}-Gap{gap:.2f}"

            path += f"-{time.strftime('%Y%m%d%H%M')}.pt"

            torch.save({
                'model_state_dict': model_state_dict,
                'optimizer_state_dict': self.optimizer.state_dict(),
            }, path)

            # Early stopping
            if self.trainer_params['early_stopping']:
                if primary_gap > best_gap:
                    non_improving_epochs += 1
                    if non_improving_epochs > self.trainer_params['early_stopping_patience']:
                        print("Early stopping")
                        break
                else:
                    non_improving_epochs = 0
                    best_gap = primary_gap

        peak_memory_usage = torch.cuda.memory_stats()['allocated_bytes.all.peak'] / 1e9
        print(f"Peak memory usage: {peak_memory_usage:.3f} GB")
        print(f"Available memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.3f} GB")

    def train_one_batch(self):
        # Reset state: generate a batch of problems
        if self.trainer_params['taillard']:
            taillard = self.taillard_batch[:7]
        else:
            taillard = None

        batch, edges, U = self.env.reset(self.trainer_params['train_batch_size'], taillard=taillard)

        # Forward
        self.model.train()
        prob_list, solutions = self.model(batch, edges, U, decode_type='sample')
        costs = self.env.get_costs(solutions)

        # Greedy forward as baseline
        with torch.no_grad():
            self.model.eval()
            _, solutions = self.model(batch, edges, U, decode_type='greedy')
            bl_costs = self.env.get_costs(solutions)

        # Loss
        advantage = - (costs - bl_costs) # Negative cost -> reward
        log_prob = prob_list.log().sum(dim=1)
        loss = -advantage * log_prob
        loss_mean = loss.mean()

        # Backward
        self.model.zero_grad(set_to_none=True)
        loss_mean.backward()
        clip_grad_norms(self.optimizer.param_groups, self.trainer_params['max_grad_norm'])

        self.optimizer.step()

        if self.trainer_params['verbose']:
            print(f"Loss: {loss_mean.item():.3f} Cost: {costs.mean().item():.3f} Baseline cost: {bl_costs.mean().item():.3f}")

    @torch.no_grad()
    def evaluate_model(self, test_batch):
        self.model.eval()
        batch, edges, U = self.env.reset(self.trainer_params['eval_batch_size'], test_batch=test_batch)
        _, solutions = self.model(batch, edges, U, decode_type='greedy')
        # reverse the solutions (permutations)
        return self.env.get_costs(solutions)

