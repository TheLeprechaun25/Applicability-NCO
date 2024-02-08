import torch
import time
from nets.LOPGATModel import LOPModel as GATModel
from nets.LOPGNNModel import LOPModel as GNNModel
from LOPEnv import LOPEnv as Env
from utils import clip_grad_norms, configure_optimizers
torch.set_float32_matmul_precision('high')


class LOPTrainer:
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
        self.secondary_eval_sizes = [20, 30, 40]
        self.secondary_batch_sizes = [16, 16, 16]
        # remove the problem size from the list
        if self.env_params['problem_size'] in self.secondary_eval_sizes:
            idx = self.secondary_eval_sizes.index(self.env_params['problem_size'])
            self.secondary_eval_sizes.pop(idx)
            self.secondary_batch_sizes.pop(idx)

        if not self.trainer_params['lolib']: # Random instances
            # Primary eval batch
            eval_batch = torch.load(f"data/n_{self.env_params['problem_size']}/n_{self.env_params['problem_size']}_cpu.pt")
            self.eval_batch = eval_batch[:self.trainer_params['eval_batch_size']]
            opt_values = torch.load(f"data/n_{self.env_params['problem_size']}/results_n_{self.env_params['problem_size']}.pt")
            self.opt_values = torch.tensor(opt_values['ma'][:self.trainer_params['eval_batch_size']])
            # Secondary eval batches
            if self.trainer_params['second_eval']:
                self.secondary_eval_batches = []
                self.secondary_eval_opt_values = []
                for size, b_size in zip(self.secondary_eval_sizes, self.secondary_batch_sizes):
                    eval_batch = torch.load(f"data/n_{size}/n_{size}_cpu.pt")
                    eval_batch = eval_batch[:b_size]
                    self.secondary_eval_batches.append(eval_batch)
                    opt_values = torch.load(f"data/n_{size}/results_n_{size}.pt")
                    opt_values = torch.tensor(opt_values['ma'][:b_size])
                    self.secondary_eval_opt_values.append(opt_values)
        else: # LOLIB instances
            from utils import load_lolib_instances
            self.lolib_batch, names, norm_factors = load_lolib_instances(path=f"data/lolib/{self.trainer_params['lolib_type']}")
            self.lolib_batch = self.lolib_batch.to(self.device)
            self.eval_batch = self.lolib_batch[self.trainer_params['lolib_train_sizes'][self.trainer_params['lolib_type']]:]
            self.lolib_size = self.eval_batch.shape[1]
            opt_value_dict = torch.load(f"data/lolib/{self.trainer_params['lolib_type']}/results/best_known_{self.trainer_params['lolib_type']}.pt")
            self.opt_values = torch.tensor([opt_value_dict[name] for name in names[self.trainer_params['lolib_train_sizes'][self.trainer_params['lolib_type']]:]]) / torch.tensor(norm_factors)[self.trainer_params['lolib_train_sizes'][self.trainer_params['lolib_type']]:]

    def run(self):
        non_improving_epochs = 0
        best_gap = 1e9
        print(f"Training model {self.trainer_params['execution_name']}...")
        for epoch in range(1, self.trainer_params['epochs']+1):
            start_time = time.time()
            print(f"\nStarting epoch {epoch}/{self.trainer_params['epochs']}")
            # TRAINING
            for episode in range(self.trainer_params['train_episodes']):
                self.env.problem_size = self.env_params['problem_size']
                self.train_one_batch()

            # EVALUATION AND SAVING
            print(f"End of epoch {epoch}. Evaluating model...")
            # Evaluate in primary problem size
            if self.trainer_params['lolib']:
                self.env.problem_size = self.lolib_size
            scores = self.evaluate_model(test_batch=self.eval_batch)
            primary_gap = 100 * ((self.opt_values - scores) / self.opt_values).mean().item()
            print(f"Size: {self.env_params['problem_size']} - Eval-Score: {scores.mean().item():.3f} - Gap: {primary_gap:.3f}%")

            # Evaluate in secondary problem sizes
            secondary_gaps = []
            if self.trainer_params['second_eval']:
                for size, eval_batch, opt_values in zip(self.secondary_eval_sizes, self.secondary_eval_batches, self.secondary_eval_opt_values):
                    self.env.problem_size = size
                    scores = self.evaluate_model(test_batch=eval_batch)
                    gap = 100 * ((opt_values - scores) / opt_values).mean().item()
                    secondary_gaps.append(gap)
                    print(f"Size: {size}. Score {scores.mean().item():.3f} {gap:.3f}%")

            # Save model
            if isinstance(self.model, torch.nn.DataParallel):
                model_state_dict = self.model.module.state_dict()
            else:
                model_state_dict = self.model.state_dict()

            if self.trainer_params['lolib']:
                if self.trainer_params['model_load']['enable']:
                    path = f"results/saved_models/int_lolib/{self.trainer_params['lolib_type']}/{self.trainer_params['execution_name']}-{self.trainer_params['model_size']}-{self.trainer_params['lolib_type']}-n{self.env_params['problem_size']}-Epoch{epoch}-Gap{primary_gap:.2f}"
                else:
                    path = f"results/saved_models/lolib/{self.trainer_params['lolib_type']}/{self.trainer_params['execution_name']}-{self.trainer_params['model_size']}-{self.trainer_params['lolib_type']}-n{self.env_params['problem_size']}-Epoch{epoch}-Gap{primary_gap:.2f}"
            else:
                path = f"results/saved_models/n_{self.env_params['problem_size']}/{self.trainer_params['execution_name']}-{self.trainer_params['model_size']}-n{self.env_params['problem_size']}-Epoch{epoch}-Gap{primary_gap:.2f}"
                if self.trainer_params['second_eval']:
                    for size, gap in zip(self.secondary_eval_sizes, secondary_gaps):
                        path += f"-N{size}Gap{gap:.2f}"

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

            print(f"Epoch duration: {time.time() - start_time:.1f}s")

        peak_memory_usage = torch.cuda.memory_stats()['allocated_bytes.all.peak'] / 1e9
        print(f"Peak memory usage: {peak_memory_usage:.3f} GB")
        print(f"Available memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.3f} GB")

    def train_one_batch(self):
        # Reset state: generate a batch of problems
        if self.trainer_params['lolib']:
            lolib = self.lolib_batch[:self.trainer_params['lolib_train_sizes'][self.trainer_params['lolib_type']]]
        else:
            lolib = None

        edges = self.env.reset(self.trainer_params['train_batch_size'], lolib=lolib)

        # Forward
        self.model.train()
        prob_list, solutions = self.model(edges, decode_type='sample')
        reward = self.env.get_rewards(solutions)

        # Greedy forward as baseline
        with torch.no_grad():
            self.model.eval()
            _, solutions = self.model(edges, decode_type='greedy')
            bl_reward = self.env.get_rewards(solutions)

        # Loss
        advantage = reward - bl_reward
        log_prob = prob_list.log().sum(dim=1)
        loss = -advantage * log_prob
        loss_mean = loss.mean()

        # Backward
        self.model.zero_grad(set_to_none=True)
        loss_mean.backward()
        clip_grad_norms(self.optimizer.param_groups, self.trainer_params['max_grad_norm'])

        self.optimizer.step()

        if self.trainer_params['verbose']:
            print(f"Loss: {loss_mean.item():.3f} Reward: {reward.mean().item():.3f} Baseline: {bl_reward.mean().item():.3f}")

    @torch.no_grad()
    def evaluate_model(self, test_batch):
        self.model.eval()
        test_batch = test_batch.to(self.device)
        edges = self.env.reset(test_batch.shape[0], test_batch=test_batch)
        _, solutions = self.model(edges, decode_type='greedy')
        # reverse the solutions (permutations)
        return self.env.get_rewards(solutions)

