from PFSPTrainer import PFSPTrainer as Trainer
from utils import generate_word
import torch
torch.set_float32_matmul_precision('high')

env_params = {
    'n_jobs': 20,
    'n_machines': 5,
    'criterion': 'flowtime',
    'device': 'cuda',
}

model_params_small = {
    'embedding_dim': 128,
    'encoder_layers': 3,
    'n_heads': 8,
    'tanh_clipping': 10,
    'ff_hidden_dim': 2*128,
}

model_params_medium = {
    'embedding_dim': 128,
    'encoder_layers': 6,
    'n_heads': 8,
    'tanh_clipping': 10,
    'ff_hidden_dim': 2*128
}

model_params_large = {
    'embedding_dim': 256,
    'encoder_layers': 8,
    'n_heads': 16,
    'tanh_clipping': 10,
    'ff_hidden_dim': 2*256,
}

optimizer_params = {
    'lr': 1e-4,
    'betas': (0.9, 0.95),
    'weight_decay': 0.1,
}

trainer_params = {
    # General
    'architecture': 'gnn',  # 'gnn', 'gat'
    'optimizer': 'adamw', # 'adam', 'adamw'
    'model_size': 'small', # 'small', 'medium', 'large'
    'compile': False,
    'second_eval': False,

    # Data
    'taillard': False,
    'taillard_type': '20_5', # '20_10'
    # all taillard types (sizes) have 10 instances each: 7 for training and 3 for evaluation

    # Training
    'epochs': 20,
    'train_episodes': 10,
    'train_batch_size': 7,
    'max_grad_norm': 1.0,

    # Evaluation
    'eval_batch_size': 3,

    # Stopping criteria
    'early_stopping': False,
    'early_stopping_patience': 20,

    # Initialization
    'model_load': {
        'enable': False,  # enable loading pre-trained model
        'path': './results/saved_models/n_20/faqixe-small-n20-Epoch38-Gap0.13-N30Gap0.22-N40Gap0.29-N50Gap0.34-202302040344.pt',  # directory path of pre-trained model and log files saved.
    },

    'verbose': False,
}


def main():
    if trainer_params['model_size'] == 'small':
        model_params = model_params_small
    elif trainer_params['model_size'] == 'medium':
        model_params = model_params_medium
    elif trainer_params['model_size'] == 'large':
        model_params = model_params_large
    else:
        raise ValueError("model_size must be 'small', 'medium' or 'large'")

    env_params['problem_size'] = [env_params['n_jobs'], env_params['n_machines']]

    execution_name = generate_word(6)
    trainer_params['execution_name'] = execution_name

    if trainer_params['taillard']:
        trainer_params['second_eval'] = False

    if trainer_params['verbose']:
        print("Parameters:")
        print(env_params)
        print(model_params)
        print(optimizer_params)
        print(trainer_params)

    trainer = Trainer(env_params=env_params, model_params=model_params, optimizer_params=optimizer_params, trainer_params=trainer_params)
    trainer.run()


if __name__ == "__main__":
    main()
