import torch
import torch.nn as nn
import itertools
from decompositions import deim
from tqdm import tqdm
from grad_dist import calnorm
import numpy as np
import math
import gc


def process_indices(indices):
    '''
    Processes indices to generate a list of cumulative indices
    '''

    l2 = indices[0]
    for i in range(len(indices) - 1):
        l2 = l2 + list(np.array(l2[-1]) + np.array(indices[i + 1]))

    return l2


def index_selection(trainloader, data3, net, clone_dict, batch_size, fraction, sel_iter, numEpochs, device, dataset_name):
    
    
    if dataset_name.lower() == 'boston':
        loss_fn = torch.nn.MSELoss(reduction='mean')
    else:
        loss_fn = torch.nn.functional.cross_entropy
    assert numEpochs > sel_iter, "Number of Epochs must be greater than sel_iter"
    indices = []
    l2 = []    
    len_ranks = batch_size * fraction
    min_range = int(len_ranks - (len_ranks * fraction))
    max_range = int(len_ranks + (len_ranks * fraction))
    
    if max_range - min_range < 1:
        ranks = np.arange((1, max_range),1, dtype=int)
        num_selections = int(numEpochs / sel_iter)
        candidates = ranks
    else:    
        ranks = np.arange(min_range, max_range, 1, dtype=int)
        num_selections = int(numEpochs / sel_iter)
        candidates = math.ceil(len(ranks) / num_selections)
    
    candidate_ranks = list(np.random.choice(list(ranks), size=candidates, replace=False))
    if len(candidate_ranks) > 3:
        candidate_ranks = list(np.random.choice(list(candidate_ranks), size=3, replace=False))
    print("current selected rank candidates:", candidate_ranks)

    
    for _, ((trainsamples, labels), V) in enumerate(tqdm(zip(trainloader, data3))):
        
        net.load_state_dict(clone_dict)
        trainsamples = trainsamples.to(device)
        labels = labels.to(device)
        
        
        A = np.reshape(trainsamples.detach().cpu().numpy(),(-1,trainsamples.shape[0]))
        out, _ = net(trainsamples, last=True, freeze=True)
        
        
        loss = loss_fn(out, labels).sum()
        l0_grad = torch.autograd.grad(loss, out)[0]
        distance_dict = {}
        for ranks in candidate_ranks:
            net.load_state_dict(clone_dict)
            idx2 = deim(V,  min(ranks, A.shape[1]))
            idx2 = list(set((itertools.chain(*idx2))))
            if dataset_name == "boston":
                out_idx, _ = net(trainsamples[idx2,:], last=True, freeze=True)
            else:
                out_idx, _ = net(trainsamples[idx2,:,:,:], last=True, freeze=True)
            loss_idx = loss_fn(out_idx, labels[idx2]).sum()
            l0_idx_grad = torch.autograd.grad(loss_idx, out_idx)[0]
            distance = calnorm(l0_idx_grad, l0_grad)
            distance_dict[tuple(idx2)] = distance 
    
        indices.append(list(min(distance_dict, key=distance_dict.get)))
    
    del clone_dict
    del net
    torch.cuda.empty_cache()    
    gc.collect()

    return process_indices(indices)
    
