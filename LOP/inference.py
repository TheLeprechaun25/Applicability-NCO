import glob
import pickle
import torch
from LOPTester import LOPTester as Tester

# CHANGE THIS
model_size = 20
instance_size = 20

env_params = {
    'problem_size': instance_size,
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
    'encoder_layers': 8,
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

tester_params = {
    'architecture': 'gnn',  # 'gnn', 'gat'
    'model_size': 'small',
    'compile': False,

    'lolib': False,
    'lolib_type': 'IO_44',  # 'IO_44', 'RandB_50', 'SGB_75', 'MB_100', 'RandA1_150', 'RandA2_200', 'XLOLIB_250'

    'eval_batch_size': 10, # Number of eval instances
    'run_batch_inference': False,

    'save_results': False,

    # Used model directory
    'dir_path': f'results/used_models/n_{model_size}',

    'verbose': False,
}


def main():
    if tester_params['model_size'] == 'small':
        model_params = model_params_small
    elif tester_params['model_size'] == 'medium':
        model_params = model_params_medium
    elif tester_params['model_size'] == 'large':
        model_params = model_params_large
    else:
        raise ValueError("Model size not recognized")

    print("Parameters:")
    print(env_params)
    print(model_params)
    print(tester_params)

    all_gaps = []
    all_scores = []
    all_times = []
    print(tester_params['dir_path'] + '/*.pt')
    for path in glob.glob(tester_params['dir_path'] + '/*.pt'):
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
        with open(f"results/inference_results/NCO_{model_size}_N{instance_size}_scores.pkl", "wb") as f:
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
