import time
import torch
import numpy as np
from torch.utils.data import DataLoader
from torch.nn import functional as F
from sklearn.metrics import accuracy_score

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
        val_loss = test_model(model, loss_fn, data, data.val_mask)
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
            
            

def train_model_multi_graph(model, optimizer, dataset, loss_fn, epochs, batch_size, save_freq=None, save_path=None, scheduler=None, device='cpu'):
    '''For training a GNN on a dataset comprising of multiple graphs'''
#     loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    model.train()
    
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
        for i, data in enumerate(dataset):
#             labels = one_hot(labels, 10).to(device)
            data = data.to(device)#.squeeze().flatten(start_dim=1)
            y = data.y[data.train_mask]
            pred = model(data.x, data.edge_index)[data.train_mask]
            loss = loss_fn(pred, y)
            
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            losses.append(loss.item())
            all_pred.append(pred.cpu())
            all_true.append(y.cpu())

            if len(losses) == 10 or i == len(dataset)-1:
                elapsed = time.time() - t
                t = time.time()
                val_loss = test_model(model, loss_fn, data, data.val_mask)
                acc = get_accuracy(torch.cat(all_pred), torch.cat(all_true))
                print(acc)
                model.train()
                print(f'Batch {i+1}/{len(dataset)}, loss: {np.mean(losses)} ({elapsed:.3f}s), val loss: {val_loss:.3f}')
                losses = []
                
        if scheduler is not None:
            scheduler.step()
            
        if save_freq and ((e+1) % save_freq == 0 or e == epochs-1):
            save_model(save_path, model, optimizer, epochs)
            print(f'Saved to {save_path}')       
            
            
def train_model(model, optimizer, dataset, loss_fn, epochs, batch_size, save_freq=None, save_path=None, scheduler=None, device='cpu'):
    '''For training a generic PyTorch model'''
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    model.train()
    
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
            labels = labels.to(device)
            x = x.to(device)
            pred = model(x)
            loss = loss_fn(pred, labels)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            losses.append(loss.item())
            all_pred.append(pred.cpu())
            all_true.append(labels.cpu())

            if len(losses) == 10 or i == len(loader)-1:
                elapsed = time.time() - t
                t = time.time()
                val_acc = get_accuracy(torch.cat(all_pred), torch.cat(all_true))
                print(f'Batch {i+1}/{len(loader)}, loss: {np.mean(losses)} ({elapsed:.3f}s), train acc: {val_acc:.3f}')
                losses = []
                
        if scheduler is not None:
            scheduler.step()
            
        if save_freq and ((e+1) % save_freq == 0 or e == epochs-1):
            save_model(save_path, model, optimizer, epochs)
            print(f'Saved to {save_path}')  


def test_model(model, loss_fn, data, mask):
    model.eval()
    pred = model(data.x, data.edge_index)[mask]
    loss = loss_fn(pred, data.y[mask])
    return loss
    
    
def get_accuracy(logits, targets):
    '''Logits are any real valued of size (N, num_classes); targets are same size but one-hot encoded'''
    pred_labels = torch.argmax(logits, dim=1)
    n_classes = targets.shape[1]
    pred_one_hot = F.one_hot(pred_labels, num_classes=n_classes)
    
    acc = accuracy_score(pred_one_hot, targets)
    return acc
