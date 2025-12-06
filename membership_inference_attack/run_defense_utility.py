import torch
import itertools
import time
import argparse
import numpy as np
from torch import nn
from tqdm import tqdm
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import precision_recall_curve

from train_models import get_dataset, train_gat, shadow_target_split
from ml_util import test_model_multi_graph, predict_multi_graph, get_auprc_score, LogitsDefenseModel
from util import calculate_robustness_scores, split_graphs_by_ind

'''
Test target model utility when logits defense is applied
'''

DEVICE = ('cuda:0' if torch.cuda.is_available() else 'cpu')

def main(dataset_name, sigma, num_iter):
    # Search over models
    # Each model for each param set, we train 8 models: 4 trained on 4 folds taken from half the dataset, 
    # and another 4 for the other half of the dataset
    util_loss_fn = nn.CrossEntropyLoss() 
    dataset = get_dataset(dataset_name).to(DEVICE)
    test_size = 0.25
    t = time.perf_counter()
    
    for i in range(num_iter):
        _, _, dataset_train, dataset_test = shadow_target_split(dataset, target_test_size=test_size, shadow_test_size=test_size)
        # Train target model
        save_path = None #f'mia-models/t_model_gat_{dataset_name}_{i}.pth'
        model = train_gat(dataset_name, 'n', dataset_train, dataset_test=dataset_test, save_path=save_path, 
                             device=DEVICE, verbose=0)
        model = LogitsDefenseModel(model, sigma=sigma)
        model.eval()
        # Model performance on train/test data            
        loss, acc, f1, auc, auprc = test_model_multi_graph(model, util_loss_fn, dataset_train, device=DEVICE, auprc=True)
        print(f'{i},{dataset_name},train,{sigma},{loss},{acc},{f1},{auc},{auprc}')

        loss, acc, f1, auc, auprc = test_model_multi_graph(model, util_loss_fn, dataset_test, device=DEVICE, auprc=True)
        print(f'{i},{dataset_name},test,{sigma},{loss},{acc},{f1},{auc},{auprc}')
    print(f'Took {time.perf_counter()-t:.5f}s')
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test emodel utility under a certain strength of logits defense')
    parser.add_argument('--dataset', type=str, help='Name of dataset to train on')
    parser.add_argument('--sigma', type=float, help='Standard deviation of noise to add to logits')
    parser.add_argument('--num_iter', type=int, default=1, help='Number of times to run')
    
    args = parser.parse_args()
    print(f'Running on dataset: {args.dataset}')
    main(args.dataset, args.sigma, args.num_iter)
