from PFSPTester import PFSPTester as Tester
import glob
import pickle
import torch
torch.set_float32_matmul_precision('high')


model_size = '20_10'
instance_size = '20_5'
random_model = True
intens = False  # Use intensification model

n_jobs = int(instance_size.split('_')[0])
n_machines = int(instance_size.split('_')[1])

env_params = {
    'n_jobs': n_jobs,
    'n_machines': n_machines,
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

model_params_extra_large = {
    'embedding_dim': 256,
    'encoder_layers': 16,
    'n_heads': 16,
    'tanh_clipping': 10,
    'ff_hidden_dim': 3*256,
}
if random_model:
    model_path = ''
else:
    model_size = '20_10'
    if intens == True:
        model_path = '/int_taillard'
    else:
        model_path = '/taillard'

tester_params = {
    # General
    'architecture': 'gnn',  # 'gnn', 'gat'
    'model_size': 'small', # 'small', 'medium', 'large', 'extra_large'
    'compile': False,
    'run_batch_inference': False,
    'eval_batch_size': 3,
    'save_results': False,

    # Data
    'taillard': True,
    'taillard_type': instance_size, # '20_10'

    # Initialization
    'folder_path': f'results/used_models{model_path}/{model_size}',  # directory path of pre-trained model and log files saved.

    'verbose': False,
}


def main():
    if tester_params['model_size'] == 'small':
        model_params = model_params_small
    elif tester_params['model_size'] == 'medium':
        model_params = model_params_medium
    elif tester_params['model_size'] == 'large':
        model_params = model_params_large
    else: # trainer_params['model_size'] == 'extra_large':
        model_params = model_params_extra_large

    if tester_params['verbose']:
        print("Parameters:")
        print(env_params)
        print(model_params)
        print(tester_params)

    env_params['problem_size'] = [env_params['n_jobs'], env_params['n_machines']]
    size = f"{env_params['n_jobs']}_{env_params['n_machines']}"
    all_gaps = []
    all_scores = []
    all_times = []
    for path in glob.glob(tester_params['folder_path'] + '/*.pt'):
        print(" ")
        print("------------------------Average STATS------------------------")
        print("╔════════╗")
        print(f"║ {path.split('/')[-1][:6]} ║")
        print("╚════════╝")
        tester_params['model_load_path'] = path
        tester = Tester(env_params=env_params, model_params=model_params, tester_params=tester_params)
        if tester_params['run_batch_inference']:
            gaps, scores, elapsed_time = tester.run_batch_inference()
        else:
            gaps, scores, elapsed_time = tester.run_instance_inference()
        all_gaps.append(gaps)
        all_scores.append(list(scores.cpu().numpy()))
        all_times.append(elapsed_time)

    # save scores as pkl
    if tester_params['save_results']:
        results = {'scores': all_scores, 'times': all_times}
        with open(f"results/inference_results/NCO_{model_size}_tai{size}_scores.pkl", "wb") as f:
            pickle.dump(results, f)

    all_gaps = torch.stack(all_gaps)
    best_pop = torch.min(all_gaps, dim=0)[0]
    avg_gaps = torch.mean(all_gaps, dim=1)

    print(" ")
    # print mean and std of gaps and times
    print("\n------------------------Overall STATS------------------------")
    print(f"Avg gap: {avg_gaps.mean().item():.2f}%")
    print(f"Std gap: {avg_gaps.std(dim=0).item():.2f}%")

    print(f"Ensemble gap: {best_pop.mean().item():.2f}%")


if __name__ == "__main__":
    main()
