import argparse
import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import TensorDataset
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import auc as AUC, roc_auc_score, f1_score, accuracy_score, precision_recall_curve, precision_score, recall_score

from train_models import train_gat, get_dataset, shadow_target_split _TRAINING_PARAMS
from util import calculate_robustness_scores
from ml_util import GenericAttackModel, LogitsDefenseModel, train_model, test_model_multi_graph, predict

DEVICE = ('cuda:0' if torch.cuda.is_available() else 'cpu')
METRIC = 'robustness'
# Number of times to train a target/shadow model on the same set of data 
NUM_RUNS = 7
NUM_PERTURB = 500

def main(dataset_name, defense, sigma=None):    
    scores_t_train, scores_t_test = [], [] # Robustness scores of the target train/test data
    scores_s_train, scores_s_test = [], [] # Robustness scores of the shadow train/test data
    t_models, s_models = [], [] # Lists of target and shadow models
    test_size = 0.25

    dataset = get_dataset(dataset_name) 
    t_dataset_train, t_dataset_test, s_dataset_train, s_dataset_test = shadow_target_split(dataset, target_test_size=test_size, shadow_test_size=test_size)
    util_loss_fn = nn.CrossEntropyLoss() 

    for i in range(NUM_RUNS):
        print(f'Run #{i+1}')
        # Train target model
        t_save_path = f'mia-models/t_model_gat_{dataset_name}_{i}.pth'
        t_model, _ = train_gat(dataset_name, 't', t_dataset_train, dataset_test=t_dataset_test, save_path=t_save_path, device=DEVICE, verbose=0)
        # Train shadow model
        s_save_path = f'mia-models/s_model_gat_{dataset_name}_{i}.pth'
        s_model, _ = train_gat(dataset_name, 's', s_dataset_train, dataset_test=s_dataset_test, save_path=s_save_path, device=DEVICE, verbose=0)
        
        if defense:
            print(f'Applying logits noise defense')
            t_model = LogitsDefenseModel(t_model, sigma=sigma)
            s_model = LogitsDefenseModel(s_model, sigma=sigma)
            
        t_model.eval()
        s_model.eval()
        # Target model performance on train/test data
        _, acc, f1, auc = test_model_multi_graph(t_model, util_loss_fn, t_dataset_train, device=DEVICE)
        print(f'Target model (train): acc: {acc:.4f}, F1: {f1:.4f}, AUROC: {auc:.4f}')

        _, acc, f1, auc = test_model_multi_graph(t_model, util_loss_fn, t_dataset_test, device=DEVICE)
        print(f'Target model (test): acc: {acc:.4f}, F1: {f1:.4f}, AUROC: {auc:.4f}')

        # Shadow model performance on test data
        _, acc, f1, auc = test_model_multi_graph(s_model, util_loss_fn, s_dataset_train, device=DEVICE)
        print(f'Shadow model (train): acc: {acc:.4f}, F1: {f1:.4f}, AUROC: {auc:.4f}')

        _, acc, f1, auc = test_model_multi_graph(s_model, util_loss_fn, s_dataset_test, device=DEVICE)
        print(f'Shadow model (test): acc: {acc:.4f}, F1: {f1:.4f}, AUROC: {auc:.4f}')

        # Get robustness scores for shadow model training/test data
    #     scalers, thresholds, all_acc, all_auroc, s_scores_train, s_scores_test = search_scaler(s_model, s_dataset_train, s_dataset_test,
    #                                                                                            n_perturb_per_graph=1000, scaler_max=5.5, ds=0.3, metric=METRIC)
        scalers = np.arange(0.1, 5., 0.25)
        s_scores_train = np.array([calculate_robustness_scores(s_model, s_dataset_train, n_perturb_per_graph=NUM_PERTURB, scaler=scaler, device=DEVICE, metric=METRIC) for scaler in (scalers)])
        s_scores_test = np.array([calculate_robustness_scores(s_model, s_dataset_test, n_perturb_per_graph=NUM_PERTURB, scaler=scaler, device=DEVICE, metric=METRIC) for scaler in (scalers)])

        # Get robustness scores for target model training/test data
        t_scores_train = np.array([calculate_robustness_scores(t_model, t_dataset_train, n_perturb_per_graph=NUM_PERTURB, scaler=scaler, device=DEVICE, metric=METRIC) for scaler in (scalers)])
        t_scores_test = np.array([calculate_robustness_scores(t_model, t_dataset_test, n_perturb_per_graph=NUM_PERTURB, scaler=scaler, device=DEVICE, metric=METRIC) for scaler in (scalers)])

        # Label shadow data as member/non-member to be used as attack model train set
        X = np.concatenate([s_scores_train.T, s_scores_test.T])
        labels = np.array(([1] * len(s_dataset_train)) + ([0] * len(s_dataset_test))).reshape(-1,1)
        y = OneHotEncoder(categories=[[0,1]], sparse_output=False).fit_transform(labels)
        att_dataset = TensorDataset(torch.Tensor(X), torch.Tensor(y))

        # Label target data as member/non-member to be used as attack model test set
        X_test = np.concatenate([t_scores_train.T, t_scores_test.T])
        labels_test = np.array(([1] * len(t_dataset_train)) + ([0] * len(t_dataset_test))).reshape(-1,1)
        y_test = OneHotEncoder(categories=[[0,1]], sparse_output=False).fit_transform(labels_test)
        att_dataset_test = TensorDataset(torch.Tensor(X_test), torch.Tensor(y_test))

        # Train attack model
        lr = 0.001
        epochs = 200
        batch_size = 16
        weight_decay = 1e-3

        att_model = GenericAttackModel(num_feat=len(scalers), dropout=0.4).to(DEVICE)
        optimizer = optim.Adam(att_model.parameters(), lr=lr, weight_decay=weight_decay)
        weight = compute_class_weight('balanced', classes=np.unique(y.argmax(axis=1)), y=y.argmax(axis=1))
        loss_fn = nn.CrossEntropyLoss(weight=torch.tensor(weight).to(DEVICE))
        train_model(att_model, optimizer, att_dataset, loss_fn, epochs, batch_size, val_dataset=None, device=DEVICE, verbose=0)
        att_model.eval()

        # Evaluate attack model on the target and shadow datasets
        att_pred = predict(att_model, att_dataset, device=DEVICE, logits=True, return_type='pt')
        att_pred_test = predict(att_model, att_dataset_test, device=DEVICE, logits=True, return_type='pt')

        # Get AUROC on target and shadow datasets
        print(f'Shadow (train) AUROC: {roc_auc_score(y.argmax(axis=1), att_pred[:,1].cpu())}')
        print(f'Target (test) AUROC: {roc_auc_score(y_test.argmax(axis=1), att_pred_test[:,1].cpu())}')

        scores_t_train.append(t_scores_train)
        scores_t_test.append(t_scores_test)
        scores_s_train.append(s_scores_train)
        scores_s_test.append(s_scores_test)
        t_models.append(t_model.to('cpu'))
        s_models.append(s_model.to('cpu'))
    
    
    
    # Label shadow data as member/non-member to be used as attack model train set
    # X = np.concatenate([s_scores_train.T, s_scores_test.T])
    s_scores_train = np.stack(scores_s_train).mean(axis=0)
    s_scores_test = np.stack(scores_s_test).mean(axis=0)
    t_scores_train = np.stack(scores_t_train).mean(axis=0)
    t_scores_test = np.stack(scores_t_test).mean(axis=0)
    # t_scores_train = scores_t_train[0]
    # t_scores_test = scores_t_test[0]

    X = np.concatenate([s_scores_train.T, s_scores_test.T])
    labels = np.array(([1] * len(s_dataset_train)) + ([0] * len(s_dataset_test))).reshape(-1,1)
    y = OneHotEncoder(categories=[[0,1]], sparse_output=False).fit_transform(labels)
    att_dataset = TensorDataset(torch.Tensor(X), torch.Tensor(y))

    # Label target data as member/non-member to be used as attack model test set
    X_test = np.concatenate([t_scores_train.T, t_scores_test.T])
    labels_test = np.array(([1] * len(t_dataset_train)) + ([0] * len(t_dataset_test))).reshape(-1,1)
    y_test = OneHotEncoder(categories=[[0,1]], sparse_output=False).fit_transform(labels_test)
    att_dataset_test = TensorDataset(torch.Tensor(X_test), torch.Tensor(y_test))
    
    lr = 0.001
    epochs = 250
    batch_size = 16
    weight_decay = 1e-4

    att_model = GenericAttackModel(num_feat=len(scalers), dropout=0.3).to(DEVICE)
    optimizer = optim.Adam(att_model.parameters(), lr=lr, weight_decay=weight_decay)
    weight = compute_class_weight('balanced', classes=np.unique(y.argmax(axis=1)), y=y.argmax(axis=1))

    loss_fn = nn.CrossEntropyLoss(weight=torch.tensor(weight).to(DEVICE))

    train_model(att_model, optimizer, att_dataset, loss_fn, epochs, batch_size, 
                val_dataset=att_dataset_test, device=DEVICE, verbose=0)
    att_model.eval()
    
    att_pred = predict(att_model, att_dataset, device=DEVICE, logits=True, return_type='pt')
    att_pred_test = predict(att_model, att_dataset_test, device=DEVICE, logits=True, return_type='pt')
    
    # Final results
    train_auroc = roc_auc_score(y.argmax(axis=1), att_pred[:,1].cpu())
    accuracy = accuracy_score(y_test.argmax(axis=1), att_pred_test.cpu().argmax(axis=1))
    auroc = roc_auc_score(y_test.argmax(axis=1), att_pred_test[:,1].cpu())
    prec = precision_score(y_test.argmax(axis=1), att_pred_test.cpu().argmax(axis=1))
    rec = recall_score(y_test.argmax(axis=1), att_pred_test.cpu().argmax(axis=1))
    precision, recall, thresholds = precision_recall_curve(y_test.argmax(axis=1), att_pred_test[:,1].cpu())
    auprc = AUC(recall, precision)
    f1 = f1_score(y_test.argmax(axis=1), att_pred_test.cpu().argmax(dim=1))
    print(f'glo_mia,{"logits" if defense else "none"},{sigma},{dataset_name.lower()},{test_size},{accuracy},{auroc},{train_auroc},{auprc},{f1}')
    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Perform GLO-MIA variation attack')
    parser.add_argument('--dataset', type=str, help='Name of dataset to train on')
    parser.add_argument('--defense', action='store_true', help='Whether to apply logits defense')
    parser.add_argument('--sigma', type=float, default=None, help='Standard deviation of Gaussian noise applied to logits during defense')
    parser.add_argument('--iter', type=int, help='Number of times to run')
    args = parser.parse_args()
    
    for i in range(args.iter):
        print(f'----------------ITERATION {i+1}/{args.iter} (dataset={args.dataset}, defense={args.defense}, sigma={args.sigma})----------------')
        main(args.dataset, args.defense, args.sigma)
        print()
