import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score

from util import calculate_robustness_scores




def search_scaler(s_model, s_dataset_train, s_dataset_test, n_perturb_per_graph=1000, scaler_min=0.1, ds=0.2, scaler_max=1, device='cpu', metric='robustness'):
    scalers = []
    all_t = []
    all_acc = []
    all_auroc = []
    s_scores_train = []
    s_scores_test = []
    for scaler in tqdm(np.arange(scaler_min, scaler_max, ds)):
        scores_train = calculate_robustness_scores(s_model, s_dataset_train, n_perturb_per_graph=n_perturb_per_graph, scaler=scaler, device=device, metric=metric)
        scores_test = calculate_robustness_scores(s_model, s_dataset_test, n_perturb_per_graph=n_perturb_per_graph, scaler=scaler, device=device, metric=metric)
        
        is_member = np.concatenate((np.ones_like(scores_train), np.zeros_like(scores_test)))
        robust_scores = np.concatenate([scores_train, scores_test])
        fpr, tpr, thresholds = roc_curve(is_member, robust_scores)

        t = max(thresholds, key=lambda x: roc_auc_score(is_member, robust_scores>x))
        
        pred_member = (robust_scores > t).astype(int)
        acc = accuracy_score(is_member, pred_member)
        auroc = roc_auc_score(is_member, robust_scores)
        f1 = f1_score(is_member, pred_member)
        
#         print(f's={scaler:.3f}:  t={t:.4f}, acc={acc:.4f}, AUC={auroc:.4f}, F1={f1:.4f}')
        
        scalers.append(scaler)
        all_acc.append(acc)
        all_auroc.append(auroc)
        all_t.append(t)
        s_scores_train.append(scores_train)
        s_scores_test.append(scores_test)
        
    return scalers, all_t, all_acc, all_auroc, np.array(s_scores_train), np.array(s_scores_test)