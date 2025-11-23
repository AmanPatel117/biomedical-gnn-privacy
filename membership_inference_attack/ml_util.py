import time
import torch
import numpy as np
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Subset
from torch_geometric import nn as gnn
from torch_geometric.loader import DataLoader as GDataLoader
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

def train_model_single_graph(model, optimizer, data, loss_fn, epochs, save_freq=None, save_path=None, scheduler=None, device='cpu'):
    '''For training a GNN on a dataset comprised of only one graph. The graph might be split into parts representing training/testing/validation'''
    model.train()
    
    print(f'Learning rate: {optimizer.param_groups[0]["lr"]}')
    print(f'Scheduler: {scheduler}' if scheduler else 'No learning rate scheduling!')
#     print(f'Training for {epochs} epochs, with batch size={batch_size}')
    print(f'Using device: {device}')
    print(f'Saving model every {save_freq} epochs to {save_path}' if save_freq else 'WARNING: Will not save model!')

    data = data.to(device)
    for e in range(epochs):
#         losses = []
#         all_pred, all_true = [], []
        t = time.time()
        print(f'\n-----Epoch {e+1}/{epochs}-----')
        y = data.y[data.train_mask]
        pred = model(data.x, data.edge_index)[data.train_mask]
        loss = loss_fn(pred, y)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        train_loss = loss.item()
        all_pred = pred.cpu()
        all_true = y.cpu()
        
        elapsed = time.time() - t
        t = time.time()
        val_loss = test_model_single_graph(model, loss_fn, data, data.val_mask)
        val_pred = model(data.x, data.edge_index)[data.val_mask].cpu()
        val_acc = get_accuracy(val_pred, data.y[data.val_mask].cpu())
        train_acc = get_accuracy(all_pred, all_true)
        
        model.train()
        print(f'Loss: {train_loss} ({elapsed:.3f}s), train acc: {train_acc:.3f}, val loss: {val_loss:.3f}, val acc: {val_acc:.3f}')
                
        if scheduler is not None:
            scheduler.step()
            
        if save_freq and ((e+1) % save_freq == 0 or e == epochs-1):
            save_model(save_path, model, optimizer, epochs)
            print(f'Saved to {save_path}') 
            
            

def train_model_multi_graph(model, optimizer, dataset, loss_fn, epochs, batch_size, val_dataset=None, save_freq=None, save_path=None, scheduler=None, device='cpu'):
    '''For training a GNN on a dataset comprising of multiple graphs'''
    model.train()
    loader = GDataLoader(dataset, batch_size=batch_size, shuffle=True)
    num_batches = len(loader)
    d = len(str(num_batches))
    
    multiclass = dataset[0].y.shape[1] > 2
    validate = (val_dataset is not None)
    
    print(f'Learning rate: {optimizer.param_groups[0]["lr"]}')
    print(f'Scheduler: {scheduler}' if scheduler else 'No learning rate scheduling!')
    print(f'Training for {epochs} epochs, with batch size={batch_size}')
    print(f'Using validation data ({len(val_dataset)} samples)' if validate else 'Not using validation data!')
    print(f'Using device: {device}')
    print(f'Saving model every {save_freq} epochs to {save_path}' if (save_freq and save_path) else 'WARNING: Will not save model!')

    for e in range(epochs):
        losses = []
        all_pred, all_true = [], []
        t = time.time()
        print(f'\n-----Epoch {e+1}/{epochs}-----')
        for i, data in enumerate(loader):
            optimizer.zero_grad()
            data = data.to(device)
            pred = model(data.x, data.edge_index, data.batch)
            loss = loss_fn(pred, data.y)
            loss.backward()
            optimizer.step()
            
            losses.append(loss.item())
            all_pred.append(pred.detach().cpu())
            all_true.append(data.y.cpu())

            if len(losses) == 10 or i == len(loader)-1:
                elapsed = time.time() - t
                pred_temp, true_temp = torch.cat(all_pred).detach().cpu(), torch.cat(all_true).detach().cpu()
                train_acc = get_accuracy(pred_temp, true_temp)
                train_auc = get_auroc_score(pred_temp, true_temp, multiclass=multiclass)
                print(f'Batch {i+1:0{d}d}/{len(loader)} | loss: {np.mean(losses):.5f} ({elapsed:.3f}s) | train acc: {train_acc:.3f} | train AUC: {train_auc:.3f}')
                
                model.train()
                t = time.time()
                losses = []
        
        if validate:
            val_loss, val_acc, val_f1, val_auc = test_model_multi_graph(model, loss_fn, val_dataset, device=device, multiclass=multiclass)
            print(f'Validation: val loss: {val_loss:.3f} | val acc: {val_acc:.3f} | val F1: {val_f1:.3f} | val AUC: {val_auc:.3f}')
            model.train()
        else:
            print()
                
        if scheduler is not None:
            scheduler.step(val_loss)
            
        if save_freq and save_path and ((e+1) % save_freq == 0 or e == epochs-1):
            save_model(save_path, model, optimizer, epochs)
            print(f'Saved to {save_path}')       
            
            
def train_model(model, optimizer, dataset, loss_fn, epochs, batch_size, val_dataset=None, save_freq=None, save_path=None, scheduler=None, device='cpu'):
    '''For training a generic PyTorch model'''
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    model.train()
    
    multiclass = dataset[0][1].shape[0] > 2
    validate = (val_dataset is not None)
    
    print(f'Learning rate: {optimizer.param_groups[0]["lr"]}')
    print(f'Scheduler: {scheduler}' if scheduler else 'No learning rate scheduling!')
    print(f'Training for {epochs} epochs, with batch size={batch_size}')
    print(f'Using device: {device}')
    print(f'Saving model every {save_freq} epochs to {save_path}' if save_freq else 'WARNING: Will not save model!')

    for e in range(epochs):
        losses = []
        all_pred, all_true = [], []
        t = time.time()
        print(f'\n-----Epoch {e+1}/{epochs}-----')
        for i, (x, labels) in enumerate(loader):
            optimizer.zero_grad()
            labels = labels.to(device)
            x = x.to(device)
            pred = model(x)
            loss = loss_fn(pred, labels)
            loss.backward()
            optimizer.step()

            losses.append(loss.item())
            all_pred.append(pred.detach().cpu())
            all_true.append(labels.cpu())

            if len(losses) == 50 or i == len(loader)-1:
                elapsed = time.time() - t
                pred_temp = torch.cat(all_pred)
                true_temp = torch.cat(all_true)
                train_acc = get_accuracy(torch.cat(all_pred), torch.cat(all_true))
                train_auc = get_auroc_score(pred_temp, true_temp, multiclass=multiclass)

                print(f'Batch {i+1}/{len(loader)} | loss: {np.mean(losses)} ({elapsed:.3f}s) | train acc: {train_acc:.4f} | train auc: {train_auc:4f}', end='\n')
                    
                model.train()
                t = time.time()
                losses = []
                
        if validate:
            val_loss, val_acc, val_f1, val_auc = test_model(model, loss_fn, val_dataset, device=device, multiclass=multiclass)
            print(f'Validation: val loss: {val_loss:.3f} | val acc: {val_acc:.3f} | val F1: {val_f1:.3f} | val AUC: {val_auc:.3f}')
            model.train()
        else:
            print()
                
        if scheduler is not None:
            scheduler.step()
            
        if save_freq and ((e+1) % save_freq == 0 or e == epochs-1):
            save_model(save_path, model, optimizer, epochs)
            print(f'Saved to {save_path}')  


def test_model_single_graph(model, loss_fn, data, mask):
    model.eval()
    
    with torch.no_grad():
        pred = model(data.x, data.edge_index)[mask]
        loss = loss_fn(pred, data.y[mask])
        return loss


def test_model_multi_graph(model, loss_fn, dataset, device='cpu', multiclass=False):
    '''
    Test the loss and accuracy on a validation dataset.
    multicless: Whether there are more than 2 categories. If True, F1 and AUROC are calculated using weighted avg
                           across all categories.
    '''
    model.eval()
    batch_size = 250
    loader = GDataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    all_logits, all_targets = [], []
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            logits = model(data.x, data.edge_index, data.batch)
            targets = data.y
            
            all_logits.append(logits)
            all_targets.append(targets)
        
        all_logits = torch.cat(all_logits)
        all_targets = torch.cat(all_targets)
            
        loss = loss_fn(all_logits, all_targets)
        acc = get_accuracy(all_logits.cpu(), all_targets.cpu())

        f1 = get_f1_score(all_logits.cpu(), all_targets.cpu(), multiclass=multiclass)
        auc = get_auroc_score(all_logits.cpu(), all_targets.cpu(), multiclass=multiclass)

        return loss, acc, f1, auc
    
    
def test_model(model, loss_fn, dataset, device='cpu', multiclass=False):
    '''
    Test a generic (non-GNN) pytorch model
    '''
    model.eval()
    batch_size = 250
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    all_logits, all_targets = [], []
    with torch.no_grad():
        for X, targets in loader:
            X = X.to(device)
            targets = targets.to(device)
            logits = model(X)
            
            all_logits.append(logits)
            all_targets.append(targets)
        
        all_logits = torch.cat(all_logits)
        all_targets = torch.cat(all_targets)
            
        loss = loss_fn(all_logits, all_targets)
        acc = get_accuracy(all_logits.cpu(), all_targets.cpu())

        f1 = get_f1_score(all_logits.cpu(), all_targets.cpu(), multiclass=multiclass)
        auc = get_auroc_score(all_logits.cpu(), all_targets.cpu(), multiclass=multiclass)

        return loss, acc, f1, auc
        

    
def get_accuracy(logits, targets):
    '''
    Returns raw unweighted accuracy. Logits are any real valued of size (N, num_classes); targets are same size but one-hot encoded
    '''
    pred_labels = torch.argmax(logits, dim=1)
    n_classes = targets.shape[1]
    pred_one_hot = F.one_hot(pred_labels, num_classes=n_classes)
    
    acc = accuracy_score(targets, pred_one_hot)
    return acc


def get_f1_score(logits, targets, multiclass=False):
    '''
    Returns the F1 score. Logits are any real valued of size (N, num_classes); targets are same size but one-hot encoded
    '''
    pred_labels = torch.argmax(logits, dim=1)
    target_labels = torch.argmax(targets, dim=1)
    f1 = f1_score(target_labels, pred_labels, average=('weighted' if multiclass else 'binary'))
    return f1


def get_auroc_score(logits, targets, multiclass=False):
    '''
    Returns the area under receiver operation curve (AUROC). Logits are any real valued of size (N, num_classes); targets are same size but one-hot encoded
    '''
    if multiclass:
        pred_labels = F.softmax(logits, dim=1)
        target_labels = targets
    else:
#         pred_labels = torch.argmax(logits, dim=1)
        pred_labels = F.softmax(logits, dim=1)[:, 1]
        target_labels = torch.argmax(targets, dim=1)
    auc = roc_auc_score(target_labels, pred_labels, average=('weighted' if multiclass else 'macro'), multi_class=('ovr' if multiclass else 'raise'))
    return auc


def save_model(save_path, model, optimizer, epoch):
    '''
    Save a model to disk, with extra training parameters in order to resume training later.
    
    save_path: Path as string to location to save the model.
    model: Model to save.
    optimizer: Optimizer to save.
    epoch: The last trained epoch (for info only; not used when resuming training)
    '''
    torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, 
        save_path)
    
    
def load_model(model, save_path, strict=True):
    '''
    Load a previously saved model for inference.
    
    model: The PyTorch model to load weights into
    save_path: Path as string to the model (.pth)
    strict: Whether to strictly load weights (kwarg for load_state_dict)
    '''
    checkpoint = torch.load(save_path, weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'], strict=strict)
    return model


def predict_multi_graph(model, dataset, idxs=None, device='cpu', logits=False, return_type='np'):
    '''
    Predict the score of multiple input graphs, and return their predicted scores.
    
    :param model: The GNN model to predict. 
    :param dataset (torch_geometric.data.Dataset):
    :param idxs (list or array of int): The indexes of the data to predict for, or None to predict for the entire dataset. Default: None
    :param logits: If True, return a 1D int array containing categorical predicts; if False, return logits of size (N x num_categories)
    
    :return (np.ndarray): 1D array containing the categorical predictions for the data, in order of idxs.
    '''
    chunksize = 256
    all_pred = []
    loader = GDataLoader(dataset if idxs is None else dataset[idxs], batch_size=chunksize, shuffle=False)
    model.eval()
    with torch.no_grad():
        for gbatch in loader:
            gbatch = gbatch.to(device)
            pred = model(gbatch.x, gbatch.edge_index, gbatch.batch).squeeze().cpu().numpy()
            
            all_pred.append(pred if logits else pred.argmax(axis=1))

        return np.concatenate(all_pred) if return_type == 'np' else torch.cat([torch.Tensor(x) for x in all_pred])


def predict(model, dataset, idxs=None, device='cpu', logits=False, return_type='np'):
    '''
    Predict the score of a (non-GNN) model on a given dataset, and return the predicted categories or logits.
    
    :param model: The GNN model to predict. 
    :param dataset (torch_geometric.data.Dataset):
    :param idxs (list or array of int): The indexes of the data to predict for, or None to predict for the entire dataset. Default: None
    :param logits: If True, return a 1D int array containing categorical predicts; if False, return logits of size (N x num_categories)
    
    :return (np.ndarray): 1D array containing the categorical predictions for the data, in order of idxs.
    '''
    chunksize = 256
    all_pred = []
    loader = DataLoader(dataset if idxs is None else Subset(dataset, idxs), batch_size=chunksize, shuffle=False)
    model.eval()
    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            all_pred.append(pred if logits else pred.argmax(axis=1))
            
        return np.concatenate(all_pred) if return_type == 'np' else torch.cat(all_pred)
    
###############################################################
##################### GNN models below ########################
###############################################################


class GenericAttackModel(nn.Module):
    def __init__(self, num_feat):
        super().__init__()
        
        self.layers = nn.Sequential(
            nn.Linear(num_feat, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 2)
        )
        
        
    def forward(self, x):
        return self.layers(x)


class GCNProteinsModel(nn.Module):
    def __init__(self, num_feat, num_classes):
        super().__init__()
        
        self.gcn_layers = nn.ModuleList([
            gnn.conv.GCNConv(num_feat, 16),
            gnn.conv.GCNConv(16, num_classes)
        ])
        
        self.linear = nn.Linear(64, num_classes)
        
    
    def forward(self, x, edge_index, batch):
        out = x
        for i, gcn in enumerate(self.gcn_layers):
            out = gcn(out, edge_index)
            if i < len(self.gcn_layers)-1:
                out = out.relu()
        
        out = gnn.global_mean_pool(out, batch)
#         out = self.linear(out)
        return out
    

class GATProteinsModel(nn.Module):
    def __init__(self, num_feat, num_classes, hidden_dim=18, batch_norm=True, dropout=0.):
        super().__init__()
        
        self.heads = 8
        self.n_layers = 4
        self.hidden_dim = hidden_dim
        self.out_dim = 144
        self.batch_norm = batch_norm
        self.embedding = nn.Linear(num_feat, hidden_dim*self.heads)
        
        self.gat_layers = nn.ModuleList([
            gnn.conv.GATConv(self.hidden_dim*self.heads, hidden_dim, heads=self.heads, dropout=dropout)
            for _ in range(self.n_layers-1) 
        ])
        
        self.bns = nn.ModuleList([
            nn.BatchNorm1d(hidden_dim*self.heads)
            for _ in range(self.n_layers-1)
        ])
        
        self.gat_layers.append(gnn.conv.GATConv(self.hidden_dim*self.heads, self.out_dim, heads=1, dropout=dropout))
        self.bns.append(nn.BatchNorm1d(self.out_dim))
        self.linear = gnn.models.MLP([self.out_dim, self.out_dim//2, self.out_dim//4, num_classes])
    
    def forward(self, x, edge_index, batch):
        out = self.embedding(x)
        for i, gat in enumerate(self.gat_layers):
            out = gat(out, edge_index).relu()
            if self.batch_norm:
                out = self.bns[i](out)
        
        out = gnn.pool.global_mean_pool(out, batch)
        return self.linear(out)

    
class GATMolhivModel(nn.Module):
    def __init__(self, num_feat, num_classes):
        super().__init__()

        self.gat_layers = nn.ModuleList([
            gnn.conv.GATConv(num_feat, 16, heads=4, dropout=0.),
            gnn.conv.GATConv(16*4, 32, heads=2, dropout=0.),
            gnn.conv.GATConv(32*2, num_classes, heads=1, dropout=0.),
        ])
        
        self.batch_norms = nn.ModuleList([
            nn.BatchNorm1d(16),
            nn.BatchNorm1d(32*2),
        ])
    
    def forward(self, x, edge_index, batch):
        out = x
        for i, gat in enumerate(self.gat_layers):
            out = gat(out, edge_index)
            if i < len(self.gat_layers)-1:
                out = out.relu()
                out = self.batch_norms[i](out)
            
        out = gnn.pool.global_mean_pool(out, batch)
#         out = self.linear(out)
        return out

