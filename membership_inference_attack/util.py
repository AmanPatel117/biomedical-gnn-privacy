import torch
import numpy as np
from torch.nn import functional as F
from torch.utils.data import TensorDataset
from torch_geometric.loader import DataLoader as GDataLoader
from torch_geometric import transforms as T
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split

from ml_util import get_accuracy, predict_multi_graph


def onehot_transform(data, categories=None):
    '''
    Transform y labels into one-hot vectors. Returns the same data object (in-place) data with its y label transformed.
    
    data (torch_geometric.data.Dataset): The graph dataset to apply one hot encoding to.
    categories (list-like): List of numeric categories. For example, for 5 categories, this would be [0,1,2,3,4]
    '''
    categories = np.atleast_2d(categories).tolist() if categories is not None else 'auto'
    data.y = torch.Tensor(OneHotEncoder(categories=categories).fit_transform(data.y.reshape(-1,1)).todense())
    return data    


def graph_train_test_split(dataset, **kwargs):
    '''
    Perform a train test split on a torch_geometric Dataset. Same arguments as sklearn.model_selection.train_test_split.
    '''
    ind = np.arange(len(dataset))
    x_train, x_test = train_test_split(ind, **kwargs)
    
    d_train, d_test = dataset[x_train], dataset[x_test]
    d_train.y = dataset.y[x_train]
    d_test.y = dataset.y[x_test]
    return d_train, d_test


def query_zero_hop(model, v_feat):
    '''
    Perform a zero-hop query on a trained model. The only input is the node and its features, with a self loop
    to itself.
    
    v_feat should be a 1D vector.
    '''
    edge_index = torch.tensor([[0], [0]], dtype=int).to(DEVICE)
    return model(v_feat.unsqueeze(0).to(DEVICE), edge_index)


def zero_hop_acc(model, data, mask):
    '''Get the accuracy of the model by only inputting 0-hop graphs; i.e. only the features of the node of interest and nothing else'''
    model.eval()
    with torch.no_grad():
        pred = torch.stack([query_zero_hop(model, v_feat).cpu().flatten() for v_feat in data.x[mask]])
        y = data.y[mask].cpu()
        return get_accuracy(pred, y)
    
    
def split_in_half(data):
    '''Split a Data object in half. Severs edges between the two halves and preserves edges within each half'''
    half_split_transform = T.RandomNodeSplit(split='train_rest', num_val=0.5, num_test=0.)
    return half_split_transform(data)
    

def ind_to_mask(ind, size):
    '''Takes a list of indices and returns a boolean mask which is True at every spot specified by the indices'''
    return np.isin(np.arange(size), ind)
    
    
def target_shadow_split(data):
    '''
    Split data randomly and equally into two objects, one for target model training and one for shadow model training. Each 
    Data object is also further split by half, as their train and test sets.
    '''
    n_nodes = data.x.shape[0]
    data_halved = split_in_half(data)
    t_ind, s_ind = torch.where(data_halved.train_mask)[0], torch.where(~data_halved.train_mask)[0]
    
    # Shuffle target indices; then assign first half as train and second half as test
    t_ind_shuffle = np.random.choice(t_ind, size=len(t_ind), replace=False)
    t_ind_train, t_ind_test = np.sort(t_ind_shuffle[:len(t_ind)//2]), np.sort(t_ind_shuffle[len(t_ind)//2:])
    
    # Shuffle shadow indices; then assign first half as train and second half as test
    s_ind_shuffle = np.random.choice(s_ind, size=len(s_ind), replace=False)
    s_ind_train, s_ind_test = np.sort(s_ind_shuffle[:len(s_ind)//2]), np.sort(s_ind_shuffle[len(s_ind)//2:])
    
    # Assign masks to Data objects
    t_data = data.clone()
    t_data.train_mask = ind_to_mask(t_ind_train, n_nodes)
    t_data.test_mask = ind_to_mask(t_ind_test, n_nodes)
    t_data.val_mask = t_data.test_mask
    
    s_data = data.clone()
    s_data.train_mask = ind_to_mask(s_ind_train, n_nodes)
    s_data.test_mask = ind_to_mask(s_ind_test, n_nodes)
    s_data.val_mask = s_data.test_mask
    
    return t_data, s_data


def create_perturbed_graphs(x, num=1000, r_min=0.1, r_max=0.5, scaler=0.4, device='cpu'):
    '''
    Perturb the features (x) of a graph. Adds uniform noise to the non-zero features in x. 
    
    num: Number of perturbed graphs to make. Default: 1000.
    r_min: Minimum noise to add. Default: 0.1.
    r_max: Maximum noise to add. Default: 0.5
    scaler: Scale r_min and r_max. Noise is sample from [scaler*r_min, scaler*r_max]. Default: 0.4
    device: Device to compute on. Default: 'cpu'
    '''
#     all_x_perturb = torch.empty((num,) + tuple(x.shape), dtype=torch.float32, device=device)
    nonzero =  (x != 0).unsqueeze(0).expand(num, -1, -1)
    zero_tensor = torch.zeros(nonzero.shape, dtype=torch.float32, device=device)

    randmat = torch.FloatTensor(nonzero.shape).to(device).uniform_(scaler*r_min, scaler*r_max)
    perturbations = torch.where(nonzero, randmat, zero_tensor)
    operator = torch.where(nonzero, torch.randint(0, 2, size=nonzero.shape, device=device)*2 - 1, zero_tensor)
#         all_x_perturb[i] = x + (perturbations * operator)
    
    return x + (perturbations * operator)
#     all_x_perturb = torch.empty((num,) + tuple(x.shape), dtype=torch.float32, device=device)
#     nonzero =  (x != 0) #torch.ones_like(x, dtype=bool, device=device)
#     zero_tensor = torch.zeros(nonzero.shape, dtype=torch.float32, device=device)

#     for i in range(num):
#         randmat = torch.FloatTensor(nonzero.shape).to(device).uniform_(scaler*r_min, scaler*r_max)
#         perturbations = torch.where(nonzero, randmat, zero_tensor)
#         operator = torch.where(nonzero, torch.randint(0, 2, size=nonzero.shape, device=device)*2 - 1, zero_tensor)
#         all_x_perturb[i] = x + (perturbations * operator)
    
#     return all_x_perturb


def calculate_robustness_scores(model, dataset, n_perturb_per_graph=1000, scaler=0.4, device='cpu', metric='robustness'):
    '''
    GLO-MIA: Calculate the robustness scores of every graph in the dataset. The robustness score of a graph is
    defined as the fraction of its perturbed graphs that are still correctly labeled by the model.
    '''
    if metric not in ['robustness', 'cross_entropy']:
        raise ValueError()
    
    model.eval()
    chunksize = 512
    loader = GDataLoader(dataset, batch_size=chunksize, shuffle=False)
    
    scores = []
    with torch.no_grad():
        for i, gbatch in enumerate(loader):
            gbatch = gbatch.to(device)
            y_t = model(gbatch.x, gbatch.edge_index, gbatch.batch)
            
            x_p = create_perturbed_graphs(gbatch.x, num=n_perturb_per_graph, scaler=scaler, device=device).to(device)
            # pred has shape (number of perturbed graphs, batch size, 2)
#             pred = model(x_p.view(n_perturb_per_graph*gbatch.y.shape[0], -1))
            pred = torch.stack([model(x_pi, gbatch.edge_index, gbatch.batch).squeeze() for x_pi in x_p])
            
#             print(y_t.shape)
#             print(y_t.repeat(n_perturb_per_graph, 1).shape)
#             print(pred.flatten(end_dim=1).shape)
            
            if metric == 'robustness':
#                 y_t = F.one_hot(y_t.argmax(dim=1), num_classes=y_t.shape[1]).to(torch.float32)
                score = torch.atleast_1d(pred.argmax(dim=2) == y_t.argmax(dim=1)).to(torch.float32).mean(dim=0)
                scores.append(torch.where((y_t.argmax(dim=1) == gbatch.y.argmax(dim=1)), score, torch.zeros_like(score)))
            elif metric == 'cross_entropy':
                y_t = F.softmax(y_t, dim=1)
                score = F.cross_entropy(pred.view(-1,2), y_t.unsqueeze(0).expand(n_perturb_per_graph,-1,-1).flatten(end_dim=1), reduction='none').view(n_perturb_per_graph, -1).mean(dim=0)
                # CE between original y_t predictions, and the true labels y
                true_ce = F.cross_entropy(y_t, gbatch.y, reduction='none')
                scores.append(torch.where(y_t.argmax(dim=1) == gbatch.y.argmax(dim=1), score, true_ce))
            
    return torch.cat(scores).cpu().numpy()


def create_attack_dataset_OLD(model, data):
    '''
    Creates the attack dataset using 0-hop querying on a trained shadow model. Each data pair consists is generated from a node feature v. The x variable is the posterior
    generated by the shadow model for input v, and the label is binary- True if v is in the shadow model's train dataset, and False if it was in the test/val dataset.
    
    s_model (nn.Module): GNN trained on the shadow dataset
    s_data (torch_geometric.data.Data): Data object representing a graph, with appropriate train_mask and test_mask defined.
    '''
    
    feat = data.x[data.train_mask | data.test_mask]    
    posteriors = F.sigmoid(torch.vstack([query_zero_hop(model, v_feat).sort()[0] for v_feat in feat]).detach().cpu(), dim=1)
    membership = torch.tensor(np.array([data.train_mask[i] for i in range(data.num_nodes) if (data.train_mask[i] or data.test_mask[i])]), dtype=torch.int)
    membership = torch.Tensor(OneHotEncoder().fit_transform(membership.reshape(-1,1)).todense())
    return posteriors, membership
    
    
def create_attack_dataset(model, train_dataset, test_dataset, device='cpu'):
    '''
    Creates the attack dataset using trained shadow model. Each data pair (x,y) is generated from a node feature v. The x variable is the logit predictions
    generated by the shadow model for input v, and the label y is binary- True if v is in the shadow model's train dataset, and False if it was in the test/val dataset.
    
    model (nn.Module): Shadow GNN trained on the shadow dataset
    train_dataset (torch_geometric.data.Dataset): Dataset object containing graphs that the shadow model was trained on
    test_dataset (torch_geometric.data.Dataset): Dataset object containing graphs that the shadow model was not trained on, but is from the same distribution

    return (torch.utils.data.Dataset)
    '''
    
    train_logits = torch.Tensor(np.sort(predict_multi_graph(model, train_dataset, device=device, logits=True), axis=1))
    test_logits = torch.Tensor(np.sort(predict_multi_graph(model, test_dataset, device=device, logits=True), axis=1))
    
    labels = np.array(([1] * len(train_dataset)) + ([0] * len(test_dataset))).reshape(-1,1)
    y = OneHotEncoder(categories=[[0,1]], sparse_output=False).fit_transform(labels)
    
    return TensorDataset(torch.cat([train_logits, test_logits]), torch.Tensor(y))

    