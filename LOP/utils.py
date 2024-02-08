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
    assert len(param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" % (str(param_dict.keys() - union_params),)
    # create the pytorch optimizer object
    optim_groups = [
        {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": optimizer_params['weight_decay']},
        {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
    ]
    optimizer = torch.optim.AdamW(optim_groups, lr=optimizer_params['lr'], betas=optimizer_params['betas'])
    return optimizer


def load_lolib_instances(path):
    batch = []
    names = []
    norm_factors = []
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
        n, cost_matrix = read_square_matrix(costs_file)
        norm_factors.append(np.max(cost_matrix))
        cost_matrix = cost_matrix / np.max(cost_matrix)
        batch.append(torch.from_numpy(cost_matrix).float())
        names.append(filename)
    return torch.stack(batch), names, norm_factors


def read_square_matrix(matrix_file):

    line = matrix_file.readline()
    if len(line)> 5:
        line = matrix_file.readline()
        assert isinstance(line, str)
        n = [int(element) for element in line.split(sep=' ') if element != '']
        n = n[0]
        matrix_row = []
        while True:
            line = matrix_file.readline()
            assert isinstance(line, str)
            if not line.isspace():
                values = [int(element) for element in line.split(sep=' ') if element != '']
                matrix_row.extend(values)
                if len(matrix_row) == n*n:
                    break

        matrix_row = np.array(matrix_row)
        matrix = np.reshape(matrix_row, (n, n))
        np.fill_diagonal(matrix, 0)
    else:
        matrix = []
        row_counter = 0
        n = [int(element) for element in line.split(sep=' ') if element != '']
        n = n[0]
        row_counter += 1
        while row_counter < n + 1:
            line = matrix_file.readline()
            assert isinstance(line, str)
            if not line.isspace():
                row_counter += 1
                matrix_row = [int(element) for element in line.split(sep=' ') if element != '']
                matrix.append(matrix_row)

        matrix = np.array(matrix)
        np.fill_diagonal(matrix, 0)
    return n, matrix


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

