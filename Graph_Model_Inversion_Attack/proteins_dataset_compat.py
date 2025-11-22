# proteins_dataset_compat.py
#
# Stitched-node view of TUDataset("PROTEINS") with a MolHIVArchiveDataset-like API.

import numpy as np
import scipy.sparse as sp

try:
    import torch
    from torch_geometric.datasets import TUDataset
except ImportError as e:
    raise ImportError(
        "proteins_dataset_compat requires torch and torch_geometric. "
        "Install them via `pip install torch torch-geometric` (plus "
        "the appropriate extra index URL for your CUDA/CPU setup)."
    ) from e


class _TUDStitchedDataset:
    """
    Base class that turns a torch_geometric TUDataset into a single
    stitched graph, mimicking the MolHIVArchiveDataset interface:

        .adj       : csr (N, N)
        .features  : csr (N, F)
        .labels    : np.ndarray (N,)  -- node labels (graph label broadcast)
        .idx_train : np.ndarray of node indices
        .idx_val   : ...
        .idx_test  : ...
        .node2graph: np.ndarray (N,)  -- which original graph each node came from
        .graph_ptr : list of cumulative node offsets
        .graph_label : np.ndarray (G,) graph-level labels

    It creates 80/10/10 graph-level splits, then assigns all nodes
    from each graph to that split.
    """

    def __init__(
        self,
        name,
        root="datasets/TUD",            # where TUDataset will be stored
        require_mask=False,
        seed=None,
        dtype_float=np.float32,
        max_graphs_total=None,
        max_graphs_per_split=None,      # (train, val, test)
        graphs_select_mode="random",    # {'head', 'random'}
        **kwargs,                       # absorbs archive_zip, extract_dir, etc.
    ):
        self.name = str(name)
        self.root = root
        self.require_mask = bool(require_mask)
        self.dtype_float = dtype_float

        self.max_graphs_total = (
            None if max_graphs_total is None else int(max_graphs_total)
        )
        self.max_graphs_per_split = (
            None
            if max_graphs_per_split is None
            else tuple(int(x) for x in max_graphs_per_split)
        )
        self.graphs_select_mode = str(graphs_select_mode).lower()
        assert self.graphs_select_mode in {"head", "random"}

        self.rng = np.random.default_rng(seed if seed is not None else 0)

        # 1) Load TUDataset
        self._load_tud()

        # 2) Build graph-level splits
        self._build_splits()

        # 3) Apply graph caps (if any)
        self._apply_graph_caps()

        # 4) Build stitched union graph
        self._build_stitched_union()

        # 5) (Optional) masks like MolHIVArchiveDataset
        if self.require_mask:
            self._build_masks()

    # ------------------------------------------------------------------
    # Loading and splits
    # ------------------------------------------------------------------

    def _load_tud(self):
        self.dataset = TUDataset(root=self.root, name=self.name)
        if len(self.dataset) == 0:
            raise RuntimeError(f"TUDataset('{self.name}') is empty.")
        self.num_graphs = len(self.dataset)

        # collect graph labels as 1D ints
        glabels = []
        for data in self.dataset:
            y = data.y
            if isinstance(y, torch.Tensor):
                y = y.view(-1)[0].item()
            glabels.append(int(y))
        self.graph_label = np.array(glabels, dtype=np.int64)

    def _build_splits(self):
        """80/10/10 split at graph level."""
        G = self.num_graphs
        idx = np.arange(G, dtype=np.int64)
        self.rng.shuffle(idx)

        n_tr = int(round(0.8 * G))
        n_va = int(round(0.1 * G))

        gids_train = np.sort(idx[:n_tr])
        gids_val = np.sort(idx[n_tr:n_tr + n_va])
        gids_test = np.sort(idx[n_tr + n_va:])

        self.gids_train = gids_train
        self.gids_valid = gids_val
        self.gids_test = gids_test

    def _pick(self, arr, k):
        if k is None or k >= len(arr):
            return arr
        if self.graphs_select_mode == "head":
            return arr[:k]
        # random
        return np.sort(self.rng.choice(arr, size=k, replace=False))

    def _apply_graph_caps(self):
        # Per-split caps
        if self.max_graphs_per_split is not None:
            kt, kv, ke = self.max_graphs_per_split
            self.gids_train = self._pick(self.gids_train, kt)
            self.gids_valid = self._pick(self.gids_valid, kv)
            self.gids_test = self._pick(self.gids_test, ke)

        # Global cap
        if self.max_graphs_total is not None:
            union = np.concatenate(
                [self.gids_train, self.gids_valid, self.gids_test]
            )
            if len(union) > self.max_graphs_total:
                labels = (
                    ["train"] * len(self.gids_train)
                    + ["val"] * len(self.gids_valid)
                    + ["test"] * len(self.gids_test)
                )
                pool = list(zip(union.tolist(), labels))
                if self.graphs_select_mode == "head":
                    pool = pool[: self.max_graphs_total]
                else:
                    idx = self.rng.choice(
                        len(pool), size=self.max_graphs_total, replace=False
                    )
                    pool = [pool[i] for i in idx]

                tr = [g for g, s in pool if s == "train"]
                va = [g for g, s in pool if s == "val"]
                te = [g for g, s in pool if s == "test"]

                self.gids_train = np.array(sorted(tr), dtype=np.int64)
                self.gids_valid = np.array(sorted(va), dtype=np.int64)
                self.gids_test = np.array(sorted(te), dtype=np.int64)

    # ------------------------------------------------------------------
    # Stitched union
    # ------------------------------------------------------------------

    def _build_stitched_union(self):
        set_tr = set(self.gids_train.tolist())
        set_va = set(self.gids_valid.tolist())
        set_te = set(self.gids_test.tolist())
        keep_any = set_tr | set_va | set_te

        rows, cols = [], []
        x_list = []
        node_labels_list = []
        node2graph = []
        graph_ptr = [0]
        idx_train, idx_val, idx_test = [], [], []

        offset = 0
        for g_idx in range(self.num_graphs):
            if g_idx not in keep_any:
                continue

            data = self.dataset[g_idx]
            n = int(data.num_nodes)

            # --- edges ---
            if data.edge_index is not None and data.edge_index.numel() > 0:
                ei = data.edge_index
                if isinstance(ei, torch.Tensor):
                    ei = ei.cpu().numpy()
                u, v = ei[0].astype(np.int64), ei[1].astype(np.int64)

                # undirected + dedup (no self-loops)
                lo = np.minimum(u, v)
                hi = np.maximum(u, v)
                mask = lo != hi
                lo, hi = lo[mask], hi[mask]
                if lo.size > 0:
                    pairs = np.stack([lo, hi], axis=1)
                    pairs = np.unique(pairs, axis=0)
                    u = pairs[:, 0]
                    v = pairs[:, 1]

                    # add both directions
                    src = np.concatenate([u, v]) + offset
                    dst = np.concatenate([v, u]) + offset
                    rows.append(src)
                    cols.append(dst)
            # else: edgeless graph, nothing to add

            # --- features ---
            if getattr(data, "x", None) is not None:
                Xg = data.x
                if isinstance(Xg, torch.Tensor):
                    Xg = Xg.cpu().numpy()
                Xg = Xg.astype(self.dtype_float, copy=False)
            else:
                # fallback: degree scalar
                deg = np.zeros(n, dtype=self.dtype_float)
                if rows:
                    # We don't have local edges here, so just zeros / ones.
                    deg[:] = 1.0
                Xg = deg.reshape(n, 1)

            x_list.append(Xg)

            # --- node labels from graph label ---
            glab = int(self.graph_label[g_idx])
            node_labels_list.append(np.full(n, glab, dtype=np.int64))
            node2graph.append(np.full(n, g_idx, dtype=np.int64))

            # --- node indices into stitched tensor ---
            span = np.arange(offset, offset + n, dtype=np.int64)
            if g_idx in set_tr:
                idx_train.extend(span)
            elif g_idx in set_va:
                idx_val.extend(span)
            elif g_idx in set_te:
                idx_test.extend(span)
            else:
                idx_train.extend(span)

            offset += n
            graph_ptr.append(offset)

        if offset == 0:
            raise RuntimeError("No nodes constructed (selection caps too strict?).")

        rows = np.concatenate(rows, axis=0) if rows else np.array([], dtype=np.int64)
        cols = np.concatenate(cols, axis=0) if cols else np.array([], dtype=np.int64)
        N = offset

        adj = sp.coo_matrix(
            (np.ones_like(rows, dtype=self.dtype_float), (rows, cols)),
            shape=(N, N),
            dtype=self.dtype_float,
        ).tocsr()
        adj.setdiag(0)
        adj.eliminate_zeros()

        X = np.vstack(x_list).astype(self.dtype_float, copy=False)
        self.features = sp.csr_matrix(X, dtype=self.dtype_float)
        self.labels = np.concatenate(node_labels_list, axis=0)
        self.adj = adj

        self.idx_train = np.asarray(idx_train, dtype=np.int64)
        self.idx_val = np.asarray(idx_val, dtype=np.int64)
        self.idx_test = np.asarray(idx_test, dtype=np.int64)

        self.node2graph = np.concatenate(node2graph, axis=0)
        self.graph_ptr = graph_ptr
        self.graph_label = self.graph_label.astype(np.int64, copy=False)

    # ------------------------------------------------------------------
    # Masks (optional)
    # ------------------------------------------------------------------

    def _build_masks(self):
        N = self.labels.shape[0]

        def m(idx):
            z = np.zeros(N, dtype=bool)
            z[idx] = True
            return z

        self.train_mask = m(self.idx_train)
        self.val_mask = m(self.idx_val)
        self.test_mask = m(self.idx_test)

        def take(idx):
            z = np.zeros_like(self.labels)
            z[idx] = self.labels[idx]
            return z

        self.y_train = take(self.idx_train)
        self.y_val = take(self.idx_val)
        self.y_test = take(self.idx_test)

    # ------------------------------------------------------------------

    def __repr__(self):
        N, F = self.features.shape
        return (
            f"{self.name}-tud(adj_shape={self.adj.shape}, "
            f"feature_shape=({N},{F}), labels_shape={self.labels.shape}, "
            f"splits=({self.idx_train.size},{self.idx_val.size},{self.idx_test.size}), "
            f"graphs={len(self.graph_ptr)-1})"
        )


class ProteinsArchiveDataset(_TUDStitchedDataset):
    """
    Thin wrapper for PROTEINS dataset.
    Accepts the same knobs as MolHIVArchiveDataset, but ignores archive-specific ones.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(name="PROTEINS", *args, **kwargs)
