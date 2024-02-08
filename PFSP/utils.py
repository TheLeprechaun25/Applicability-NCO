import torch
import numpy as np
import math
import random
import string
import os


def clip_grad_norms(param_groups, max_norm=math.inf):
    """Clips the norms for all param groups to max_norm and returns gradient norms before clipping
    """
    grad_norms = [
        torch.nn.utils.clip_grad_norm_(
            group['params'],
            max_norm if max_norm > 0 else math.inf,  # Inf so no clipping but still call to calc
            norm_type=2
        )
        for group in param_groups
    ]
    grad_norms_clipped = [min(g_norm, max_norm) for g_norm in grad_norms] if max_norm > 0 else grad_norms
    return grad_norms, grad_norms_clipped


def configure_optimizers(model, optimizer_params):
    # separate out all parameters to those that will and won't experience regularizing weight decay
    decay = set()
    no_decay = set()
    whitelist_weight_modules = (torch.nn.Linear,)
    blacklist_weight_modules = (torch.nn.BatchNorm1d, torch.nn.LayerNorm, torch.nn.InstanceNorm1d, torch.nn.Embedding)
    for mn, m in model.named_modules():
        for pn, p in m.named_parameters():
            fpn = '%s.%s' % (mn, pn) if mn else pn  # full param name
            if fpn == 'W_placeholder':
                # special case for the placeholder symbol
                no_decay.add(fpn)
            if pn.endswith('bias'):
                # all biases will not be decayed
                no_decay.add(fpn)
            elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                # weights of whitelist modules will be weight decayed
                decay.add(fpn)
            elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                # weights of blacklist modules will NOT be weight decayed
                no_decay.add(fpn)

    # validate that we considered every parameter
    param_dict = {pn: p for pn, p in model.named_parameters()}
    inter_params = decay & no_decay
    union_params = decay | no_decay
    assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params),)
    assert len(
        param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" % (
    str(param_dict.keys() - union_params),)
    # create the pytorch optimizer object
    optim_groups = [
        {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": optimizer_params['weight_decay']},
        {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
    ]
    optimizer = torch.optim.AdamW(optim_groups, lr=optimizer_params['lr'], betas=optimizer_params['betas'])
    return optimizer


def load_taillard_instances(path):
    batch = []
    names = []
    # order files alphabetically
    for filename in sorted(os.listdir(path)):
        # check that it is a file in the directory
        if not os.path.isfile(os.path.join(path, filename)):
            continue
        # check that it is a file in the first directory

        try:
            costs_file = open(path + '/' + filename, "r")
        except FileNotFoundError:
            print('Instance not found: ' + filename)
            continue
        n_jobs, n_machines, cost_matrix = read_matrix(costs_file)
        batch.append(torch.from_numpy(cost_matrix).float())
        names.append(filename)
    return torch.stack(batch), names


def read_matrix(matrix_file):
    matrix_file.readline()
    line = matrix_file.readline() # n_jobs, n_machines first two elements in line
    n_jobs, n_machines, initial_seed, upper_bound, lower_bound = [int(element) for element in line.split(sep=' ') if element != '']
    matrix = np.zeros((n_machines, n_jobs))
    matrix_file.readline()
    for i in range(n_machines):
        line = matrix_file.readline()
        assert isinstance(line, str)
        arr_row = [int(element) for element in line.split(sep=' ') if element != '']
        assert len(arr_row) == n_jobs
        matrix[i, :] = arr_row

    #assert there is no other line with numbers in the file
    line = matrix_file.readline()
    # assert line has no strings
    assert not any(char.isalpha() for char in line)
    return n_jobs, n_machines, matrix


def generate_word(length):
    VOWELS = "aeiou"
    CONSONANTS = "".join(set(string.ascii_lowercase) - set(VOWELS))
    word = ""
    for i in range(length):
        if i % 2 == 0:
            word += random.choice(CONSONANTS)
        else:
            word += random.choice(VOWELS)
    return word


def random_as_taillard():
    all_sizes = [[20, 5], [20, 10], [20, 20], [50, 10], [100, 10], [200, 20], [500, 20]]
    for sizes in all_sizes:
        print(sizes)
        n_jobs = sizes[0]
        n_machines = sizes[1]
        path = f"/home/andoni/PHD/2-Work-constructive-NCO/PFSP/data/{n_jobs}_{n_machines}/{n_jobs}_{n_machines}_cuda.pt"
        batch = torch.load(path)
        for i in range(batch.shape[0]):
            instance = batch[i, :, :].cpu().numpy()
            instance = instance * 1000000
            instance = instance.astype(int)
            save_path = f"/home/andoni/PHD/2-Work-constructive-NCO/PFSP/data/random_as_taillard/{n_jobs}_{n_machines}/{n_jobs}_{n_machines}_{i}"
            f = open(save_path, "w+")
            #f.write('number of jobs, number of machines, initial seed, upper bound and lower bound :\r\n')
            f.write(str(n_jobs) + ' ' + str(n_machines) + '\r\n')
            #f.write('processing times :\r\n')
            for j in range(n_machines):
                row = list(instance[:, j])
                line = ' '
                for c in range(n_jobs):
                    line += str(row[c]) + ' '
                f.write(str(line) + '\r\n')

            f.close()


def compute_edges(batch_size, batch, criterion, device):
    _, n_machines, n_jobs = batch.shape

    edges = np.zeros([batch_size, n_jobs, n_jobs])
    for b in range(batch_size):
        for job_i in range(n_jobs):
            for job_j in range(n_jobs):
                if job_i != job_j:
                    job_i_times = batch[b, :, job_i]
                    job_j_times = batch[b, :, job_j]

                    comp_times = np.zeros([2, n_machines])
                    for m in range(n_machines):
                        if m == 0:
                            comp_times[0, 0] = job_i_times[0]
                            comp_times[1, 0] = job_i_times[0] + job_j_times[0]
                        else:
                            comp_times[0, m] = comp_times[0, m - 1] + job_i_times[m]
                            if (comp_times[0, m] > comp_times[1, m - 1]):
                                comp_times[1, m] = comp_times[0, m] + job_j_times[m]
                            else:
                                comp_times[1, m] = comp_times[1, m - 1] + job_j_times[m]
                    if criterion == "makespan":
                        e = comp_times[1, -1]
                    else:  # self.criterion == "flowtime":
                        e = (comp_times[0, -1] + comp_times[1, -1]) / 2
                    edges[b, job_i, job_j] = e

        edges[b, :, :] = edges[b, :, :] / np.max(edges[b, :, :])
    return torch.tensor(edges, dtype=torch.float32, device=device)


def compute_nodes(batch_size, n_jobs, n_machines, batch, S, U, k):
    """
    S: Scheduled jobs. (batch_size, n_jobs)
    k: Number of scheduled jobs
    nodes: output, (batch_size, n_jobs, 3) -> IdleTime, ArtificialTime
    """
    np_nodes = np.zeros((batch_size, n_jobs, 2))

    for b in range(batch_size):
        C = batch[b, :, :]
        u = U[b, :]
        for i in range(n_jobs-k):
            job_i = U[b, i]
            if k != 0:
                IT = IdleTime(S[b, :], job_i, k, n_jobs, n_machines, C)
            else:
                IT = 0
            AT = ArtificialTime(S[b, :], job_i, k, n_jobs, n_machines, C, u)
            np_nodes[b, job_i, 0] = IT
            np_nodes[b, job_i, 1] = AT

        max_it = np.max(np_nodes[b, :, 0])
        if max_it > 0:
            np_nodes[b, :, 0] = np_nodes[b, :, 0] / max_it
        np_nodes[b, :, 1] = np_nodes[b, :, 1] / np.max(np_nodes[b, :, 1])

    """nodes = torch.stack([torch.ones([batch_size, n_jobs], device=batch.device),
                         torch.zeros([batch_size, n_jobs], device=batch.device),
                         aux], 2)"""

    return torch.tensor(np_nodes, dtype=torch.float32)



def IdleTime(S, job_i, k, n_jobs, n_machines, C):
    idle_time = 0
    aux1 = n_jobs - 2
    idle_times = IdleTimesAtPosition(S, C, k, n_machines, job_i)

    for j in range(2, n_machines):
        aux2 = k * (n_machines - j)
        wjk = n_machines / (j + (aux2/aux1))
        idle_time += wjk * max(idle_times[j-2], 0)

    return idle_time


def IdleTimesAtPosition(S, C, k, n_machines, job_i):
    idle_times = np.zeros(n_machines - 1, dtype=int)
    timeTable = np.zeros(n_machines, dtype=int)

    timeTable[0] = C[0][S[0]]

    for j in range(1, n_machines):
        timeTable[j] = timeTable[j-1] + C[j][S[0]]

    fitness = timeTable[n_machines-1]
    for z in range(1, k):
        job = S[z]
        timeTable[0] += C[0][job]
        for machine in range(1, n_machines):
            processingTime = C[machine][job]
            if timeTable[machine-1] < timeTable[machine]:
                timeTable[machine] += processingTime
            else:
                timeTable[machine] = timeTable[machine-1] + processingTime
        fitness += timeTable[n_machines-1]

    timeTable[0] = timeTable[0] + C[0][job_i]

    for machine in range(1, n_machines):
        processingTime = C[machine][job_i]
        if timeTable[machine-1] < timeTable[machine]:
            timeTable[machine] = timeTable[machine] + processingTime
        else:
            idle_times[machine-1] = timeTable[machine-1] - timeTable[machine]
            timeTable[machine] = timeTable[machine-1] + processingTime
    fitness += timeTable[n_machines-1]

    return idle_times


def ArtificialTime(S, job_i, k, n_jobs, n_machines, C, U):
    Cim = PartialEvaluation2(S, C, k, n_machines, job_i)

    artificial_job_times = np.zeros(n_machines, dtype=int)
    for i in range(1, n_jobs-k):
        for machine in range(n_machines):
            artificial_job_times[machine] = C[machine][U[i]]

    artificial_job_times = artificial_job_times / n_machines

    Cpm = PartialEvaluation1(S, C, k, n_machines, artificial_job_times)

    return Cim + Cpm


def PartialEvaluation1(S, C, size, n_machines, new_job_times):
    timeTable = np.zeros(n_machines, dtype=int)
    first_gene = S[0]
    fitness = 0
    if size != 0:
        timeTable[0] = C[0][first_gene]
        for j in range(1, n_machines):
            timeTable[j] = timeTable[j-1] + C[j][first_gene]

        fitness = timeTable[n_machines - 1]

        for z in range(1, size):
            job = S[z]
            timeTable[0] += C[0][job]
            for machine in range(1, n_machines):
                processingTime = C[machine][job]
                if timeTable[machine-1] < timeTable[machine]:
                    timeTable[machine] += processingTime
                else:
                    timeTable[machine] = timeTable[machine-1] + processingTime

            fitness += timeTable[n_machines-1]


    timeTable[0] += new_job_times[0]

    for machine in range(1, n_machines):
        processingTime = new_job_times[machine]
        if timeTable[machine-1] < timeTable[machine]:
            timeTable[machine] += processingTime
        else:
            timeTable[machine] = timeTable[machine-1] + processingTime

    fitness += timeTable[n_machines-1]

    return fitness


def PartialEvaluation2(S, C, size, n_machines, new_job):
    timeTable = np.zeros(n_machines, dtype=int)
    first_gene = S[0]
    fitness = 0
    if size != 0:
        timeTable[0] = C[0][first_gene]
        for j in range(1, n_machines):
            timeTable[j] = timeTable[j-1] + C[j][first_gene]

        fitness = timeTable[n_machines - 1]

        for z in range(1, size):
            job = S[z]
            timeTable[0] += C[0][job]
            for machine in range(1, n_machines):
                processingTime = C[machine][job]
                if timeTable[machine-1] < timeTable[machine]:
                    timeTable[machine] += processingTime
                else:
                    timeTable[machine] = timeTable[machine-1] + processingTime

            fitness += timeTable[n_machines-1]

    timeTable[0] += C[0][new_job]

    for machine in range(1, n_machines):
        processingTime = C[machine][new_job]
        if timeTable[machine-1] < timeTable[machine]:
            timeTable[machine] += processingTime
        else:
            timeTable[machine] = timeTable[machine-1] + processingTime

    fitness += timeTable[n_machines-1]

    return fitness


def LRnm(n_jobs, n_machines, batch):
    costs = []
    solutions = []
    # transpose batch
    for b in range(batch.shape[0]):
        C = batch[b, :, :]
        S = np.zeros(n_jobs, dtype=int) # Scheduled jobs
        U = np.zeros(n_jobs, dtype=int) # Unscheduled jobs
        U_index = np.zeros(n_jobs, dtype=int)
        for i in range(n_jobs):
            U[i] = i

        for k in range(n_jobs):
            for i in range(n_jobs-k):
                U_index[i] = IndexFunction(S, k, n_jobs, n_machines, U[i], U, C)
            min_idx = np.argmin(U_index)
            S[k] = U[min_idx]
            U = np.delete(U, min_idx)
            U_index = np.zeros(n_jobs-k-1, dtype=int)

        cost = get_flowtime(n_jobs, n_machines, S, C)
        costs.append(cost)
        solutions.append(S)
    return costs, solutions


def get_flowtime(n_jobs, n_machines, seq, processing_times):
    timeTable = np.zeros(n_machines)
    first_gene = seq[0]
    timeTable[0] = processing_times[0][first_gene]
    for j in range(1, n_machines):
        timeTable[j] = timeTable[j - 1] + processing_times[j][first_gene]

    fitness = timeTable[n_machines - 1]

    for z in range(1, n_jobs):
        job = seq[z]
        timeTable[0] += processing_times[0][job]
        prev_machine = timeTable[0]
        for machine in range(1, n_machines):
            timeTable[machine]= max(prev_machine, timeTable[machine]) + processing_times[machine][job]
            prev_machine=timeTable[machine]

        fitness += timeTable[n_machines-1]

    return fitness


def IndexFunction(S, k, n_jobs, n_machines, i, U, C):
    ITik = 0
    if k != 0:
        ITik = IdleTime(S, i, k, n_jobs, n_machines, C)
    ATik = ArtificialTime(S, i, k, n_jobs, n_machines, C, U)
    index = (n_jobs-k-2) * ITik + ATik
    return index


def create_random_instances():
    sizes = ['20_5', '20_10', '20_20', '50_10', '100_10', '200_20', '500_20']
    batch_size = 100
    for size in sizes:
        n_jobs, n_machines = size.split('_')
        n_jobs = int(n_jobs)
        n_machines = int(n_machines)
        # Random instances randint from 0 to 100
        instance = torch.randint(0, 100, (batch_size, n_jobs, n_machines))
        # save to file
        torch.save(instance, f"PFSP/data/{size}/instances_{size}.pt")



