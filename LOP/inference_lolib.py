import glob
import pickle
import torch
from LOPTester import LOPTester as Tester

# CHANGE THIS
model_type = 'IO_44'  # 'IO_44', 'RandB_50', 'SGB_75', 'RandA1_150', 'RandA2_200', 'XLOLIB_250'
lolib_type = 'IO_44'  # 'IO_44', 'RandB_50', 'SGB_75', 'RandA1_150', 'RandA2_200', 'XLOLIB_250'
intens = False  # Use intensification model


env_params = {
    'problem_size': 0,
}

model_params_small = {
    'embedding_dim': 128,
    'encoder_layers': 4,
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

model_params_extra_large = {
    'embedding_dim': 256,
    'encoder_layers': 16,
    'n_heads': 16,
    'tanh_clipping': 10,
    'ff_hidden_dim': 3*256,
}

if intens == True:
    model_path = 'int_lolib'
else:
    model_path = 'lolib'

tester_params = {
    'architecture': 'gnn',  # 'gnn', 'gat'
    'model_size': 'small',
    'compile': False,
    'eval_batch_size': 1,
    'run_batch_inference': False,
    'save_results': False,

    # Used model directory
    'dir_path': f'results/used_models/{model_path}/{model_type}',

    # LOLIB Instances
    'lolib': True,
    'lolib_type': lolib_type, # 'IO_44' (44), 'RandB_50' (50), 'SGB_75' (75), 'RandA1_150' (150), 'RandA2_200' (200), 'XLOLIB_250' (250)
    'lolib_n_instances': {'IO_44': 31, 'RandB_50': 20, 'SGB_75': 25, 'RandA1_150': 25, 'RandA2_200': 25, 'XLOLIB_250': 39},
    'lolib_train_sizes': {'IO_44': 24, 'RandB_50': 16, 'SGB_75': 20, 'RandA1_150': 20, 'RandA2_200': 20, 'XLOLIB_250': 31},

    'verbose': False,
}


def main():
    if tester_params['model_size'] == 'small':
        model_params = model_params_small
    elif tester_params['model_size'] == 'medium':
        model_params = model_params_medium
    elif tester_params['model_size'] == 'large':
        model_params = model_params_large
    else: # tester_params['model_size'] == 'extra_large':
        model_params = model_params_extra_large

    env_params['problem_size'] = int(tester_params['lolib_type'].split('_')[1])
    tester_params['eval_batch_size'] = tester_params['lolib_n_instances'][tester_params['lolib_type']] - tester_params['lolib_train_sizes'][tester_params['lolib_type']]

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
        with open(f"LOP/results/inference_results/NCO_{model_type}_LOLIB_{lolib_type}_scores.pkl", "wb") as f:
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

