# run_ppi_experiment.py

import torch
import numpy as np
import torch.nn.functional as F
import torch.optim as optim
from models.gcn import GCN, embedding_GCN
from models.graphsage import graphsage, embedding_graphsage
from models.gat import GAT, embedding_gat
from topology_attack import PGDAttack
from utils import *
# from dataset import Dataset   # <- not used anymore
from molhiv_dataset_compat import MolHIVArchiveDataset
import argparse
from sklearn.metrics import roc_curve, auc, average_precision_score
import scipy.io as sio
import random
import os

# ------------------------- Helpers you already had -------------------------

def test(adj, features, labels, victim_model):
    adj, features, labels = to_tensor(adj, features, labels, device=device)

    victim_model.eval()
    adj_norm = normalize_adj_tensor(adj)
    output = victim_model(features, adj_norm)  # logits of shape (N, 2) for binary task

    loss_test = F.nll_loss(F.log_softmax(output[idx_test], dim=1), labels[idx_test].long())
    acc_test = accuracy(output[idx_test], labels[idx_test])
    print("Test set results:", "loss= {:.4f}".format(loss_test.item()), "accuracy= {:.4f}".format(acc_test.item()))

    return output.detach()

def dot_product_decode(Z):
    Z = F.normalize(Z, p=2, dim=1)
    Z = torch.matmul(Z, Z.t())
    adj = torch.relu(Z - torch.eye(Z.shape[0], device=Z.device))
    return adj

def preprocess_Adj(adj, feature_adj):
    n = len(adj)
    cnt = 0
    adj = adj.numpy()
    feature_adj = feature_adj.numpy()
    for i in range(n):
        for j in range(n):
            if feature_adj[i][j] > 0.14 and adj[i][j] == 0.0:
                adj[i][j] = 1.0
                cnt += 1
    print(cnt)
    return torch.FloatTensor(adj)

def transfer_state_dict(pretrained_dict, model_dict):
    state_dict = {}
    for k, v in pretrained_dict.items():
        if k in model_dict.keys():
            state_dict[k] = v
        else:
            print("Missing key(s) in state_dict :{}".format(k))
    return state_dict

def _to_numpy_dense(x):
    # accepts torch.Tensor (dense), torch sparse (to_dense), or scipy.sparse
    import numpy as _np
    import scipy.sparse as _sp
    import torch as _torch
    if _sp.issparse(x):
        return x.toarray().astype(_np.float32)
    if isinstance(x, _torch.Tensor):
        if x.is_sparse:
            x = x.to_dense()
        return x.detach().cpu().numpy().astype(_np.float32)
    return _np.asarray(x, dtype=_np.float32)

def metric(ori_adj, inference_adj, idx):
    """
    ROC-AUC & AP for link inference on a node subset:
    - Binarize ground-truth adj to {0,1} (handles any 2s).
    - Use upper triangle (no diagonal) to avoid double counting (i,j)/(j,i).
    - Balance negatives to #positives for AP stability (like your original intent).
    """
    import numpy as np
    from sklearn.metrics import roc_curve, auc, average_precision_score

    A = _to_numpy_dense(ori_adj)
    S = _to_numpy_dense(inference_adj)

    # restrict to attacked subgraph
    A = A[np.ix_(idx, idx)]
    S = S[np.ix_(idx, idx)]

    # make sure GT is exactly {0,1}
    A = (A > 0).astype(np.uint8)

    # score clipping is optional but keeps things tidy
    S = np.clip(S, 0.0, 1.0)

    # take upper triangle only (exclude diagonal)
    triu_i, triu_j = np.triu_indices(A.shape[0], k=1)
    y_true = A[triu_i, triu_j]
    y_score = S[triu_i, triu_j]

    # balance negatives to positives (if you want comparability with earlier code)
    pos_idx = np.where(y_true == 1)[0]
    neg_idx = np.where(y_true == 0)[0]
    if len(pos_idx) == 0:
        print("No positive edges in the selected subset; cannot compute ROC/AP.")
        return
    if len(neg_idx) > len(pos_idx):
        rng = np.random.default_rng(0)
        neg_idx = rng.choice(neg_idx, size=len(pos_idx), replace=False)

    keep = np.concatenate([pos_idx, neg_idx])
    y_true_b = y_true[keep]
    y_score_b = y_score[keep]

    fpr, tpr, _ = roc_curve(y_true_b, y_score_b)
    roc = auc(fpr, tpr)
    ap = average_precision_score(y_true_b, y_score_b)
    print(f"Inference attack AUC: {roc:.4f}  AP: {ap:.4f}")

def Auc(ori_adj, modified_adj, idx):
    real_edge = []
    pred_edge = []
    for i in idx:
        for j in idx:
            if i != j:
                real_edge.append(ori_adj[i][j])
                pred_edge.append(modified_adj[i][j])
    fpr, tpr, threshold = roc_curve(real_edge, pred_edge)
    print(auc(fpr, tpr))
    return auc(fpr, tpr)

# ------------------------- Args -------------------------

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=15, help='Random seed.')
parser.add_argument('--epochs', type=int, default=100, help='Number of epochs to optimize in GraphMI attack.')
parser.add_argument('--lr', type=float, default=0.01, help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=16, help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate (1 - keep probability).')

# Force PPI usage; keep the flag for compatibility
parser.add_argument('--dataset', type=str, default='ppi', choices=['ppi'], help='dataset')

# Pick which PPI target (GO term) to use: single label index -> binary classification
parser.add_argument('--ppi_target', type=int, default=0, help='PPI label index (0..120) for one-vs-rest.')

parser.add_argument('--density', type=float, default=1.0, help='Edge density estimation')
parser.add_argument('--model', type=str, default='PGD', choices=['PGD', 'min-max'], help='model variant')
parser.add_argument('--nlabel', type=float, default=0.1, help='Fraction of nodes to attack')

args = parser.parse_args()

# ------------------------- Repro/Device -------------------------

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if device.type == 'cuda':
    torch.cuda.manual_seed(args.seed)

# ------------------------- Load PPI with chosen target -------------------------
# target is an int -> labels become (N,) in {0,1}
# data = PPIDataset(
#     root="~/data",
#     setting="gcn",
#     target=0,
#     require_mask=True,

#     # NEW: keep exactly one full-sized PPI graph (no downsizing)
#     single_graph=('train', 0),          # choose split {'train','val','test'} and graph index

#     # Optional: create a within-graph 10/10/80 split so val/test aren't empty
#     # (omit this line to keep pure 'gcn' semantics where only one split is populated)
#     local_split_if_single='10_10_80',

#     # Ignored in single-graph mode:
#     max_graphs_per_split=None,
#     nodes_per_graph=None,
# )


##### MOLHIV LOADER ######
data = MolHIVArchiveDataset(
    archive_zip="../Graph_Model_Inversion_Attack/archive.zip",
    extract_dir="../Graph_Model_Inversion_Attack/archive_extracted",
    require_mask=True,
    seed=22,
    max_graphs_per_split=(10, 10, 10),  # per-split caps
    max_graphs_total=1600,                  # global cap (applied after per-split)
    graphs_select_mode="random",            # or "head"
)

print(data)
adj = data.adj            # csr (N, N)
features = data.features  # csr (N, 50)
labels = data.labels      # (N,) int64 {0,1}

# Use the official PPI node splits that come with PPIDataset
idx_train, idx_val, idx_test = data.idx_train, data.idx_val, data.idx_test

# choose subset of nodes to attack
idx_attack = np.array(random.sample(range(adj.shape[0]), int(adj.shape[0] * args.nlabel)))

# Estimate number of edges to flip (same formula you had)
num_edges = int(0.5 * args.density * adj.sum() / adj.shape[0]**2 * len(idx_attack)**2)

# Preprocess to tensors (keep features as dense float, labels as long indices)
adj, features, labels = preprocess(adj, features, labels, preprocess_adj=False, onehot_feature=False)
print(adj, adj.shape)
print(features, features.shape)
print(labels)

# Feature-based adjacency (if you still want it)
#feature_adj = dot_product_decode(features)

# Initial adj set to zeros (as you had)
init_adj = torch.zeros_like(adj).to(torch.float32).cpu()  # same shape as adj; starts at 0

# ------------------------- Victim model (binary classification) -------------------------

# For one-vs-rest on a single PPI label: nclass = 2
# nfeat = features.shape[1]
# nclass = 2

# victim_model = GCN(nfeat=nfeat, nclass=nclass, nhid=args.hidden,
#                    dropout=args.dropout, weight_decay=args.weight_decay, device=device)
# victim_model = victim_model.to(device)

# # Fit on official train/val split (CrossEntropy, expecting labels in {0,1})
# victim_model.fit(features, adj, labels, idx_train, idx_val, train_iters = 200)

# # Build embedding model and load overlapping parameters
# embedding = embedding_GCN(nfeat=nfeat, nhid=args.hidden, device=device)
# embedding.load_state_dict(transfer_state_dict(victim_model.state_dict(), embedding.state_dict()))
# embedding = embedding.to(device)

# # ------------------------- Victim model (GAT, binary classification) -------------------------

# # For one-vs-rest on a single PPI label: nclass = 2
# nfeat = features.shape[1]
# nclass = 2

# # Instantiate GAT victim model
# victim_model = GAT(
#     nfeat=nfeat,
#     nhid=args.hidden,
#     nclass=nclass,
#     dropout=args.dropout,
#     alpha=0.2,      # e.g. 0.2 if not in args
#     nheads=8,    # e.g. 8   if not in args
#     device=device
# ).to(device)

# # Fit on official train/val split (NLLLoss, expecting labels in {0,1})
# victim_model.fit(
#     features,
#     adj,
#     labels,
#     idx_train,
#     idx_val,
#     train_iters=20
# )

# # ------------------------- GAT embedding model -------------------------

# embedding = embedding_gat(
#     nfeat=nfeat,
#     nhid=args.hidden,
#     nclass=nclass,
#     dropout=args.dropout,
#     alpha=0.2,      # same as above
#     nheads=8,    # same as above
#     device=device
# )

# # Transfer overlapping parameters from trained GAT victim to embedding GAT
# embedding.load_state_dict(
#     transfer_state_dict(victim_model.state_dict(), embedding.state_dict())
# )
# embedding = embedding.to(device)

# ------------------------- Victim model (GraphSAGE, binary classification) -------------------------

nfeat = features.shape[1]
nclass = 2

victim_model = graphsage(
    nfeat=nfeat,
    nhid=args.hidden,
    nclass=nclass,
    dropout=args.dropout,
    lr=getattr(args, "lr", 0.01),         # fallback to 0.01 if args.lr not defined
    weight_decay=args.weight_decay,
    with_relu=True,
    with_bias=False,
    device=device
).to(device)

victim_model.fit(
    features,
    adj,
    labels,
    idx_train,
    idx_val,          # can be None if you don't have a val split
    train_iters=200,  # or whatever you want
    initialize=True,
    verbose=True,
    normalize=True,
    patience=100
)
# ------------------------- GraphSAGE embedding model -------------------------

embedding = embedding_graphsage(
    nfeat=nfeat,
    nhid=args.hidden,
    with_bias=False,
    device=device
)

# Transfer overlapping parameters from trained GraphSAGE victim model
embedding.load_state_dict(
    transfer_state_dict(victim_model.state_dict(), embedding.state_dict())
)
sage_embedding = embedding.to(device)

# ------------------------- Attack model -------------------------

attack = PGDAttack(model=victim_model, embedding=embedding, nnodes=adj.shape[0], loss_type='CE', device=device)
attack = attack.to(device)

def main():
    # Run the attack to infer links among targeted nodes
    print('Beginning attack...')
    num_edges = max(2000, int(0.005 * len(idx_attack)**2))
    attack.attack(features, init_adj, labels, idx_attack, num_edges, epochs=200)#args.epochs)
    inference_adj = attack.modified_adj.cpu()

    print('=== testing GCN on original (clean) graph ===')
    _ = test(adj, features, labels, victim_model)

    print('=== calculating link inference AUC & AP on attacked subset ===')
    metric(adj, inference_adj, idx_attack)

if __name__ == '__main__':
    main()