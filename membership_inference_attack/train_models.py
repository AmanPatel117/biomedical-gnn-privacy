import os
import torch
import numpy as np
from sklearn.utils.class_weight import compute_class_weight
from torch import nn, optim
from torch_geometric.datasets import TUDataset, PPI
from torch_geometric.transforms import BaseTransform
from torch.utils.data import ConcatDataset
from ogb.graphproppred import PygGraphPropPredDataset

from util import onehot_transform, normalize_features, graph_train_test_split
from ml_util import CustomGATModel, CustomGraphDataset, LogitsDefenseModel, train_model_multi_graph, oversample_multi_graph, undersample_multi_graph, load_model


_DATASET_PARAMS = {
    'MUTAG': {'num_categories': 2, 'num_features': 7},
    'PROTEINS': {'num_categories': 2, 'num_features': 3},
    'PPI': {'num_categories': 121, 'num_features': 50},
    'ogbg-molhiv': {'num_categories': 9, 'num_features': 2}
}

_TRAINING_PARAMS = {
    'GAT': {
        'PROTEINS': {
            'lr': 0.001,
            'epochs': 200,
            'batch_size': 10,
            'weight_decay': 1e-4,
        },
        'MUTAG': {
            'lr': 0.001,
            'epochs': 200,
            'batch_size': 16,
            'weight_decay': 1e-4,
            'model_params': {
                'heads': 4,
                'layers': 4
            }
        },
        'PPI': {
            'lr': 0.001,
            'epochs': 100,
            'batch_size': 2,
            'weight_decay': 1e-4,
            'loss_fn': nn.BCEWithLogitsLoss,
            'model_params': {
                'readout': False
            }
        },
        'ogbg-molhiv': {
            'lr': 0.0001,
            'epochs': 50,
            'batch_size': 48,
            'weight_decay': 1e-5,
        }
        
        
    },
    
}

class TUTransform(BaseTransform):
    def __init__(self, num_categories):
        self.num_categories = num_categories
    def __call__(self, g):
        return onehot_transform(g, categories=list(range(self.num_categories)))
    def forward(self, g):
        return self(g)

class OGBMolhivTransform(BaseTransform):
    def __init__(self, min_vals, max_vals):
        self.max_vals = max_vals
        self.min_vals = min_vals
    def __call__(self, g):
        device = g.x.device
        g = normalize_features(g, mini=self.min_vals.to(device), maxi=self.max_vals.to(device))
        return onehot_transform(g, categories=[0,1])
    def forward(self, g):
        return self(g)
    
def _get_tu_transform(num_categories):
#     return lambda g: onehot_transform(g, categories=list(range(num_categories)))
    return TUTransform(num_categories)

def ogb_molhiv_transform(g):
    g.x = g.x.to(torch.float32)
    return onehot_transform(g, categories=[0,1])


def load_gat(save_path, num_feat, num_categories, model_params=None):
    if model_params is None:
        model_params = {}
        
    model = CustomGATModel(num_feat=num_feat, num_classes=num_categories, **model_params) 
    return load_model(model, save_path)


def train_gat(dataset_name, model_type, dataset_train, save_path=None, dataset_test=None, device='cpu', params=None, 
              model_params=None, resample=None, verbose=0, defense=None):
    '''
    Train Graph Attention Network (GAT)
    
    verbose: 
        0 = no logging at all 
        1 = tqdm progress bar for epochs
        2 = detailed printing during epochs
    '''
    if params is None:
        params = _TRAINING_PARAMS['GAT'][dataset_name]
    if 'model_params' in params and model_params is None:
        model_params = params['model_params']
    if model_params is None:
        model_params = {}


    num_feat = dataset_train[0].x.shape[1]
    num_categories = dataset_train[0].y.shape[1]
    
    lr, epochs, batch_size, weight_decay = params['lr'], params['epochs'], params['batch_size'], params['weight_decay']
    
    model = CustomGATModel(num_feat=num_feat, num_classes=num_categories, **model_params).to(device)
        
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                           factor=0.5,
                                                           patience=50,
                                                           min_lr=1e-6)
    
    if 'loss_fn' not in params:
        weight = compute_class_weight('balanced', 
                                      classes=np.unique(dataset_train.y.argmax(dim=1).cpu()), 
                                      y=dataset_train.y.argmax(dim=1).cpu().numpy())

        if resample is not None:
            weight = torch.tensor([1.,1.])

        loss_fn = nn.CrossEntropyLoss(weight=torch.tensor(weight).to(device))
    else:
#         class_counts = 
#         pos_weight = dataset_train.y.
        loss_fn = params['loss_fn']()
    
    
        

    train_model_multi_graph(model, optimizer, dataset_train, loss_fn, epochs, batch_size, val_dataset=dataset_test, 
                            save_path=save_path, save_freq=10, 
#                             scheduler=scheduler, 
                            device=device,
                            resample=resample,
                            verbose=verbose)
    
    return model


def get_dataset(name, folder='/home/hice1/khom9/scratch/CSE-8803-MLG-Data/'):
    '''
    Get a dataset. The dataset will be saved to the path specified by `folder`. 
    `name` can be one of {MUTAG, PROTEINS, PPI, ogbg-molhiv}.
    
    return: torch_geometric Dataset
    '''
    if name not in _DATASET_PARAMS.keys():
        raise ValueError(f'Error: Dataset name "{name}" is invalid. Valid datasets are {list(_DATASET_PARAMS.keys())}')
    
    params = _DATASET_PARAMS[name]
    num_categories = params['num_categories']
    
    root = os.path.join(folder, name)
    
    if name == 'ogbg-molhiv':
        mini = torch.tensor([0., 0., 0., 0., 0., 0., 0., 0., 0.])
        maxi = torch.tensor([91.,  0., 10.,  8.,  4.,  2.,  5.,  1.,  1.])
        dataset = PygGraphPropPredDataset(root=root, name=name, pre_transform=OGBMolhivTransform(mini, maxi))    
#         mini, maxi = dataset.x.min(dim=0)[0], dataset.x.max(dim=0)[0]
#         dataset.transform = OGBMolhivTransform(mini, maxi)
        dataset = normalize_features(dataset)
    elif name == 'PPI':
        dataset_train = PPI(root=root, split='train')
        dataset_test = PPI(root=root, split='test')
        dataset_val = PPI(root=root, split='val')
        
        dataset = CustomGraphDataset.from_datasets([dataset_train, dataset_test, dataset_val])
    else:
        dataset = TUDataset(root=root, name=name, pre_transform=TUTransform(num_categories))
#         dataset = onehot_transform(dataset, categories=list(range(num_categories)))        

    return dataset


def shadow_target_split(dataset, shadow_size=0.5, shadow_test_size=0.2, target_test_size=0.5, node_level=False):
    '''
    Split a multi-graph dataset into two, one for target model training and one for shadow model training.
    
    shadow_size: Fraction of data to put in the shadow dataset
    shadow_test_size: Fraction of the shadow data to set as test data
    target_test_size: Fraction of the target data to set as test data
    '''

    # Split dataset inhalf for the target model dataset and shadow model dataset
    t_dataset, s_dataset = graph_train_test_split(dataset, node_level=node_level, test_size=shadow_size)

    # Split each dataset into train/test splits
    if target_test_size is None or target_test_size == 0:
        t_dataset_train, t_dataset_test = t_dataset, None
    else:
        t_dataset_train, t_dataset_test = graph_train_test_split(t_dataset, node_level=node_level, test_size=target_test_size)
        
    if shadow_test_size is None or shadow_test_size == 0:
        s_dataset_train, s_dataset_test = s_dataset, None
    else:
        s_dataset_train, s_dataset_test = graph_train_test_split(s_dataset, node_level=node_level, test_size=shadow_test_size)
    
    return t_dataset_train, t_dataset_test, s_dataset_train, s_dataset_test
