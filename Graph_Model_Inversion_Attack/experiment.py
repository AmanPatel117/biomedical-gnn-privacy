# experiment.py

import argparse
import random
import os

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from models.gcn import GCN, embedding_GCN
from models.graphsage import graphsage, embedding_graphsage
from models.gat import GAT, embedding_gat
from topology_attack import PGDAttack

from molhiv_dataset_compat import MolHIVArchiveDataset
from proteins_dataset_compat import ProteinsArchiveDataset
from mutag_dataset_compat import MutagArchiveDataset

from utils import (
    to_tensor,
    normalize_adj_tensor,
    accuracy,
    preprocess,
)


# ------------------------- Small helpers -------------------------


def transfer_state_dict(pretrained_dict, model_dict):
    """Copy parameters that exist in both state_dicts."""
    state_dict = {}
    for k, v in pretrained_dict.items():
        if k in model_dict.keys():
            state_dict[k] = v
        else:
            print(f"Missing key(s) in state_dict: {k}")
    return state_dict


def _to_numpy_dense(x):
    """Accept torch.Tensor (dense/sparse) or scipy.sparse and return dense np.array."""
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
    ROC-AUC, AP, and best-F1 for link inference on a node subset:
    - Binarize ground-truth adj to {0,1}.
    - Use upper triangle (no diagonal).
    - Balance negatives to #positives.
    """
    from sklearn.metrics import roc_curve, auc, average_precision_score, precision_recall_curve

    A = _to_numpy_dense(ori_adj)
    S = _to_numpy_dense(inference_adj)

    # restrict to attacked subgraph
    A = A[np.ix_(idx, idx)]
    S = S[np.ix_(idx, idx)]

    # ensure {0,1}
    A = (A > 0).astype(np.uint8)
    S = np.clip(S, 0.0, 1.0)

    # upper triangle, no diag
    triu_i, triu_j = np.triu_indices(A.shape[0], k=1)
    y_true = A[triu_i, triu_j]
    y_score = S[triu_i, triu_j]

    pos_idx = np.where(y_true == 1)[0]
    neg_idx = np.where(y_true == 0)[0]

    if len(pos_idx) == 0:
        print("No positive edges in the selected subset; cannot compute ROC/AP/F1.")
        return float("nan"), float("nan"), float("nan")

    if len(neg_idx) > len(pos_idx):
        rng = np.random.default_rng(0)
        neg_idx = rng.choice(neg_idx, size=len(pos_idx), replace=False)

    keep = np.concatenate([pos_idx, neg_idx])
    y_true_b = y_true[keep]
    y_score_b = y_score[keep]

    # ROC-AUC + AP
    fpr, tpr, _ = roc_curve(y_true_b, y_score_b)
    roc = auc(fpr, tpr)
    ap = average_precision_score(y_true_b, y_score_b)

    # Best F1 over thresholds via precision-recall curve
    precisions, recalls, thresholds = precision_recall_curve(y_true_b, y_score_b)
    # last point in PR curve corresponds to threshold -> +inf, skip for F1
    precisions = precisions[:-1]
    recalls = recalls[:-1]
    f1_scores = 2 * precisions * recalls / (precisions + recalls + 1e-12)
    best_f1 = float(np.nanmax(f1_scores)) if f1_scores.size > 0 else float("nan")

    print(f"Inference attack AUC: {roc:.4f}  AP: {ap:.4f}  best-F1: {best_f1:.4f}")
    return float(roc), float(ap), best_f1


def evaluate_victim(adj, features, labels, idx_test, victim_model, device):
    """Evaluate victim model on clean graph and return (loss, acc, logits)."""
    adj_t, feat_t, labels_t = to_tensor(adj, features, labels, device=device)
    victim_model.eval()
    adj_norm = normalize_adj_tensor(adj_t)

    # All your models (GCN / GAT / GraphSAGE) already output log_softmax
    log_probs = victim_model(feat_t, adj_norm)

    loss_test = F.nll_loss(log_probs[idx_test], labels_t[idx_test].long())
    acc_test = accuracy(log_probs[idx_test], labels_t[idx_test])

    print(
        "Test set results:",
        "loss= {:.4f}".format(loss_test.item()),
        "accuracy= {:.4f}".format(acc_test.item()),
    )

    return float(loss_test.item()), float(acc_test.item()), log_probs.detach()


# ------------------------- Model builder -------------------------


def build_victim_and_embedding(arch, nfeat, nclass, args, device):
    """Factory for (victim_model, embedding_model) given an architecture name."""
    if arch == "gcn":
        victim_model = GCN(
            nfeat=nfeat,
            nclass=nclass,
            nhid=args.hidden,
            dropout=args.dropout,
            weight_decay=args.weight_decay,
            device=device,
        ).to(device)

        embedding = embedding_GCN(
            nfeat=nfeat,
            nhid=args.hidden,
            device=device,
        )

    elif arch == "gat":
        victim_model = GAT(
            nfeat=nfeat,
            nhid=args.hidden,
            nclass=nclass,
            dropout=args.dropout,
            alpha=0.2,
            nheads=8,
            device=device,
        ).to(device)

        embedding = embedding_gat(
            nfeat=nfeat,
            nhid=args.hidden,
            nclass=nclass,
            dropout=args.dropout,
            alpha=0.2,
            nheads=8,
            device=device,
        )

    elif arch == "graphsage":
        victim_model = graphsage(
            nfeat=nfeat,
            nhid=args.hidden,
            nclass=nclass,
            dropout=args.dropout,
            lr=args.lr,
            weight_decay=args.weight_decay,
            with_relu=True,
            with_bias=False,
            device=device,
        ).to(device)

        embedding = embedding_graphsage(
            nfeat=nfeat,
            nhid=args.hidden,
            with_bias=False,
            device=device,
        )

    else:
        raise ValueError(f"Unknown architecture: {arch}")

    # parameter transfer
    embedding.load_state_dict(
        transfer_state_dict(victim_model.state_dict(), embedding.state_dict())
    )
    embedding = embedding.to(device)

    return victim_model, embedding


# ------------------------- Single experiment -------------------------


def run_single_experiment(
    dataset_name,
    arch,
    run_id,
    adj,
    features,
    labels,
    idx_train,
    idx_val,
    idx_test,
    idx_attack,
    init_adj,
    args,
    device,
):
    """Train victim, run attack, return metrics for a single run of one architecture."""
    # seed that depends on dataset + arch + run
    base = hash(dataset_name + arch) % (10**6)
    seed = args.seed + run_id + base
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed(seed)

    print(f"\n=== Dataset: {dataset_name} | Arch: {arch} | Run: {run_id} | Seed: {seed} ===")

    nfeat = features.shape[1]
    nclass = 2

    victim_model, embedding = build_victim_and_embedding(
        arch, nfeat, nclass, args, device
    )

    # train victim
    if arch == "graphsage":
        victim_model.fit(
            features,
            adj,
            labels,
            idx_train,
            idx_val,
            train_iters=50,
            initialize=True,
            verbose=True,
            normalize=True,
            patience=100,
        )
    else:
        victim_model.fit(
            features,
            adj,
            labels,
            idx_train,
            idx_val,
            train_iters=50,
        )

    # test victim
    test_loss, test_acc, _ = evaluate_victim(
        adj, features, labels, idx_test, victim_model, device
    )

    # attack
    num_edges = max(2000, int(0.005 * len(idx_attack) ** 2))
    attack = PGDAttack(
        model=victim_model,
        embedding=embedding,
        nnodes=adj.shape[0],
        loss_type="CE",
        device=device,
    ).to(device)

    print("Beginning attack...")
    attack.attack(
        features,
        init_adj,
        labels,
        idx_attack,
        num_edges,
        epochs=args.epochs,
    )
    inference_adj = attack.modified_adj.cpu()

    attack_auc, attack_ap, attack_f1 = metric(adj, inference_adj, idx_attack)

    return {
        "dataset": dataset_name,
        "arch": arch,
        "run": run_id,
        "seed": seed,
        "test_loss": test_loss,
        "test_acc": test_acc,
        "attack_auc": attack_auc,
        "attack_ap": attack_ap,
        "attack_f1": attack_f1,
    }


# ------------------------- Dataset loader -------------------------


def load_dataset(dataset_name: str):
    """
    Small factory to load MolHIV / PROTEINS / MUTAG in a unified way.
    For PROTEINS/MUTAG we use torch_geometric.TUDataset under the hood via
    ProteinsArchiveDataset / MutagArchiveDataset, but expose the same
    fields as MolHIVArchiveDataset.
    """
    dataset_name = dataset_name.lower()
    if dataset_name == "molhiv":
        return MolHIVArchiveDataset(
            archive_zip="../Graph_Model_Inversion_Attack/archive.zip",
            extract_dir="../Graph_Model_Inversion_Attack/archive_extracted",
            require_mask=True,
            seed=22,
            max_graphs_per_split=(10, 10, 10),
            max_graphs_total=1600,
            graphs_select_mode="random",
        )
    elif dataset_name == "proteins":
        return ProteinsArchiveDataset(
            require_mask=True,
            seed=22,
            max_graphs_per_split=(10, 10, 10),
            max_graphs_total=1600,
            graphs_select_mode="random",
        )
    elif dataset_name == "mutag":
        return MutagArchiveDataset(
            require_mask=True,
            seed=22,
            max_graphs_per_split=(10, 10, 10),
            max_graphs_total=1600,
            graphs_select_mode="random",
        )
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")


# ------------------------- Main experiment loop -------------------------


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=15, help="Random seed.")
    parser.add_argument(
        "--epochs", type=int, default=100, help="Attack optimization epochs."
    )
    parser.add_argument("--lr", type=float, default=0.01, help="LR for GraphSAGE.")
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--hidden", type=int, default=16)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument(
        "--nlabel",
        type=float,
        default=0.1,
        help="Fraction of nodes to attack (for idx_attack).",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=10,
        help="Number of trials per architecture.",
    )
    parser.add_argument(
        "--archs",
        type=str,
        default="gcn,gat,graphsage",
        help="Comma-separated list of architectures: gcn,gat,graphsage",
    )
    parser.add_argument(
        "--datasets",
        type=str,
        default="proteins,mutag,molhiv",
        help="Comma-separated list of datasets: molhiv,proteins,mutag",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default=".",
        help="Directory to save CSV results.",
    )

    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed(args.seed)

    arch_list = [a.strip().lower() for a in args.archs.split(",") if a.strip()]
    dataset_list = [d.strip().lower() for d in args.datasets.split(",") if d.strip()]

    results = []

    for dataset_name in dataset_list:
        print("\n#############################")
        print(f"### Loading dataset: {dataset_name}")
        print("#############################")

        # --------- load data ---------
        data = load_dataset(dataset_name)
        print(data)

        adj = data.adj            # csr (N, N)
        features = data.features  # csr (N, F)
        labels = data.labels      # (N,) int {0,1}
        idx_train, idx_val, idx_test = data.idx_train, data.idx_val, data.idx_test

        # choose subset of nodes to attack
        idx_attack = np.array(
            random.sample(
                range(adj.shape[0]),
                int(adj.shape[0] * args.nlabel),
            )
        )

        # convert to torch tensors (no adj normalization here; that happens inside models)
        adj_t, features_t, labels_t = preprocess(
            adj,
            features,
            labels,
            preprocess_adj=False,
            onehot_feature=False,
        )

        # initial (fake) adjacency for attack
        init_adj = torch.zeros_like(adj_t).to(torch.float32).cpu()

        # run experiments for each architecture
        for arch in arch_list:
            for run_id in range(args.runs):
                res = run_single_experiment(
                    dataset_name=dataset_name,
                    arch=arch,
                    run_id=run_id,
                    adj=adj_t,
                    features=features_t,
                    labels=labels_t,
                    idx_train=idx_train,
                    idx_val=idx_val,
                    idx_test=idx_test,
                    idx_attack=idx_attack,
                    init_adj=init_adj,
                    args=args,
                    device=device,
                )
                results.append(res)

    df = pd.DataFrame(results)
    print("\n=== All runs (all datasets) ===")
    print(df)

    summary = (
        df.groupby(["dataset", "arch"])[["test_acc", "attack_auc", "attack_ap", "attack_f1"]]
        .agg(["mean", "std"])
    )
    print("\n=== Summary by (dataset, architecture) (mean ± std) ===")
    print(summary)

    os.makedirs(args.outdir, exist_ok=True)
    raw_path = os.path.join(args.outdir, "graphmi_experiments_raw.csv")
    summary_path = os.path.join(args.outdir, "graphmi_experiments_summary.csv")
    df.to_csv(raw_path, index=False)
    summary.to_csv(summary_path)
    print(f"\nSaved raw results to: {raw_path}")
    print(f"Saved summary to:     {summary_path}")


if __name__ == "__main__":
    main()