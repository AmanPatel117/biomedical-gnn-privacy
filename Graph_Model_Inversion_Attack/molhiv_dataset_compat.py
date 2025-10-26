# molhiv_archive_dataset_compat.py
# Adds graph selection controls:
#   - max_graphs_total: int | None        -> cap total stitched graphs (train+val+test combined)
#   - max_graphs_per_split: (int,int,int) -> cap per split (train, val, test)
#   - graphs_select_mode: {'head','random'}  (deterministic if seed is set)
#
# NOTE: We use the ORIGINAL archive order to build a cumulative node pointer (gptr_all),
# so we can slice node-feat.csv correctly even when subsetting graphs.

import os
import os.path as osp
import zipfile
import numpy as np
import pandas as pd
import scipy.sparse as sp

def _maybe_read_csv(path, **kw):
    if osp.isfile(path):
        return pd.read_csv(path, **kw)
    gz = path + ".gz"
    if osp.isfile(gz):
        return pd.read_csv(gz, **kw, compression="gzip")
    return None


class MolHIVArchiveDataset:
    """
    MolHIV loader that mirrors your PPIDataset-style API but uses local archive files.

    Required archive layout (under extract_dir/raw/):
      edge.csv, num-node-list.csv, num-edge-list.csv
    Optional:
      node-feat.csv, node-label.csv, graph-label.csv, split/split_dict.npy

    New graph selection knobs
    -------------------------
    max_graphs_total : int | None
        If set, keep at most this many graphs across all splits (order controlled by graphs_select_mode).
    max_graphs_per_split : tuple[int,int,int] | None
        If set, caps graphs from (train, val, test) *individually*.
        Applied before max_graphs_total (the latter can further reduce the union).
    graphs_select_mode : {'head','random'}, default='random'
        'head'   -> take earliest graphs in each split as listed by the split file (or fallback).
        'random' -> sample without replacement using the provided seed.

    Other key args (unchanged)
    --------------------------
    single_graph, local_split_if_single, require_mask, seed, dedup_undirected, dtype_float
    """

    def __init__(
        self,
        archive_zip,
        extract_dir="archive_extracted",
        require_mask=False,
        seed=None,
        single_graph=None,
        local_split_if_single=None,
        dedup_undirected=True,
        dtype_float=np.float32,
        max_graphs_total=None,
        max_graphs_per_split=None,      # e.g., (3000, 500, 500)
        graphs_select_mode="random",
    ):
        self.archive_zip = osp.expanduser(archive_zip)
        self.extract_dir = osp.expanduser(extract_dir)
        self.raw_dir = osp.join(self.extract_dir, "raw")

        self.require_mask = bool(require_mask)
        self.rng = np.random.default_rng(seed if seed is not None else 0)
        self.single_graph = None if single_graph is None else (str(single_graph[0]).lower(), int(single_graph[1]))
        if self.single_graph is not None and self.single_graph[0] not in {"train", "val", "test"}:
            raise ValueError("single_graph split must be one of {'train','val','test'}")
        self.local_split_if_single = local_split_if_single
        self.dedup_undirected = bool(dedup_undirected)
        self.dtype_float = dtype_float

        # selection caps
        self.max_graphs_total = None if max_graphs_total is None else int(max_graphs_total)
        self.max_graphs_per_split = None if max_graphs_per_split is None else tuple(int(x) for x in max_graphs_per_split)
        self.graphs_select_mode = str(graphs_select_mode).lower()
        assert self.graphs_select_mode in {"head", "random"}

        # 1) Ensure extracted
        self._ensure_extracted()

        # 2) Load essentials
        self._load_counts_and_edges()    # sets: n_counts, e_counts, num_graphs, edges_np_all
        self._build_gptr_all()           # cumulative node pointer over ALL graphs (original order)
        self._load_features_if_any()     # sets: node_features or None
        self._load_graph_labels_if_any() # sets: graph_label (G,)
        self._load_splits_or_make_fallback()  # sets: gids_train, gids_valid, gids_test

        # 3) Apply graph caps (split-wise and/or global) — operates on graph IDs
        if self.single_graph is None:
            self._apply_graph_caps()

        # 4) Build view
        if self.single_graph is not None:
            self._build_single_graph()
        else:
            self._build_stitched_union()

        if self.require_mask:
            self._build_masks()

    # ----------------------------- I/O helpers -----------------------------

    def _ensure_extracted(self):
        if not osp.isdir(self.raw_dir):
            if not osp.isfile(self.archive_zip):
                raise FileNotFoundError(f"Couldn't find {self.archive_zip}")
            os.makedirs(self.extract_dir, exist_ok=True)
            with zipfile.ZipFile(self.archive_zip, "r") as z:
                z.extractall(self.extract_dir)
            assert osp.isdir(self.raw_dir), f"Couldn't find {self.raw_dir} after unzip"

    def _load_counts_and_edges(self):
        edges_csv = osp.join(self.raw_dir, "edge.csv")
        n_nodes_csv = osp.join(self.raw_dir, "num-node-list.csv")
        n_edges_csv = osp.join(self.raw_dir, "num-edge-list.csv")

        edges_df = _maybe_read_csv(edges_csv, header=None, names=["src", "dst"])
        if edges_df is None:
            raise FileNotFoundError(f"Missing {edges_csv}(.gz)")
        n_nodes_df = _maybe_read_csv(n_nodes_csv, header=None)
        n_edges_df = _maybe_read_csv(n_edges_csv, header=None)
        if n_nodes_df is None or n_edges_df is None:
            raise FileNotFoundError("Missing num-node-list.csv(.gz) and/or num-edge-list.csv(.gz)")

        self.n_counts = n_nodes_df.iloc[:, 0].astype(int).tolist()
        self.e_counts = n_edges_df.iloc[:, 0].astype(int).tolist()
        if len(self.n_counts) != len(self.e_counts):
            raise ValueError("num-node-list and num-edge-list have different lengths")

        self.num_graphs = len(self.n_counts)
        self.edges_np_all = edges_df.to_numpy(dtype=np.int64)
        if self.edges_np_all.shape[1] != 2:
            raise ValueError("edge.csv must have exactly two columns (src,dst) with NO header")

        total_e = len(self.edges_np_all)
        if sum(self.e_counts) != total_e and sum(self.e_counts) * 2 != total_e:
            raise ValueError(
                f"num-edge-list sums to {sum(self.e_counts)} but edge.csv has {total_e} rows "
                "(not compatible with one- or two-direction storage)."
            )
        self.edges_list_is_bidirectional = (sum(self.e_counts) * 2 == total_e)

    def _build_gptr_all(self):
        """Cumulative node offsets over ALL graphs in archive order."""
        self.gptr_all = [0]
        for n in self.n_counts:
            self.gptr_all.append(self.gptr_all[-1] + int(n))

    def _load_features_if_any(self):
        feat_csv = osp.join(self.raw_dir, "node-feat.csv")
        feat_df = _maybe_read_csv(feat_csv, header=None)
        if feat_df is not None:
            self.node_features = feat_df.to_numpy(dtype=self.dtype_float, copy=False)
        else:
            nlabel_csv = osp.join(self.raw_dir, "node-label.csv")
            nlabel_df = _maybe_read_csv(nlabel_csv, header=None)
            if nlabel_df is not None:
                labels = nlabel_df.iloc[:, 0].astype(int).to_numpy()
                K = labels.max() + 1
                self.node_features = np.eye(K, dtype=self.dtype_float)[labels]
            else:
                self.node_features = None

    def _load_graph_labels_if_any(self):
        glabel_csv = osp.join(self.raw_dir, "graph-label.csv")
        glabel_df = _maybe_read_csv(glabel_csv, header=None)
        if glabel_df is None:
            self.graph_label = np.zeros(self.num_graphs, dtype=np.int64)
        else:
            y = glabel_df.iloc[:, 0].to_numpy()
            if y.dtype.kind in "f":
                y = (y >= 0.5).astype(np.int64)
            else:
                u = np.unique(y)
                if set(u.tolist()) == {-1, 1}:
                    y = ((y + 1) // 2).astype(np.int64)
                else:
                    y = y.astype(np.int64)
            if y.shape[0] != self.num_graphs:
                raise ValueError("graph-label.csv length does not match number of graphs")
            self.graph_label = y

    def _load_splits_or_make_fallback(self):
        split_dir = osp.join(self.extract_dir, "split")
        split_npy = osp.join(split_dir, "split_dict.npy")
        gids_train = gids_valid = gids_test = None

        if osp.isfile(split_npy):
            d = np.load(split_npy, allow_pickle=True).item()
            def _np(x): return np.array(d.get(x, []), dtype=np.int64)
            gids_train = _np("train")
            gids_valid = _np("valid")
            gids_test  = _np("test")

        if gids_train is None or gids_valid is None or gids_test is None:
            idx = np.arange(self.num_graphs, dtype=np.int64)
            self.rng.shuffle(idx)
            n_tr = int(round(0.8 * self.num_graphs))
            n_va = int(round(0.1 * self.num_graphs))
            gids_train = np.sort(idx[:n_tr])
            gids_valid = np.sort(idx[n_tr:n_tr + n_va])
            gids_test  = np.sort(idx[n_tr + n_va:])

        self.gids_train = gids_train
        self.gids_valid = gids_valid
        self.gids_test  = gids_test

    # ----------------------- Graph selection (NEW) ------------------------

    def _pick(self, arr, k):
        if k is None or k >= len(arr):
            return arr
        if self.graphs_select_mode == "head":
            return arr[:k]
        # random
        return np.sort(self.rng.choice(arr, size=k, replace=False))

    def _apply_graph_caps(self):
        # cap per split
        if self.max_graphs_per_split is not None:
            kt, kv, ke = self.max_graphs_per_split
            self.gids_train = self._pick(self.gids_train, kt)
            self.gids_valid = self._pick(self.gids_valid, kv)
            self.gids_test  = self._pick(self.gids_test,  ke)

        # cap total across splits (preserving rough split proportions)
        if self.max_graphs_total is not None:
            union = np.concatenate([self.gids_train, self.gids_valid, self.gids_test])
            if len(union) > self.max_graphs_total:
                # Build labeled pool and sample deterministically
                labels = (['train'] * len(self.gids_train) +
                          ['val']   * len(self.gids_valid) +
                          ['test']  * len(self.gids_test))
                pool = list(zip(union.tolist(), labels))
                if self.graphs_select_mode == "head":
                    pool = pool[:self.max_graphs_total]
                else:
                    idx = self.rng.choice(len(pool), size=self.max_graphs_total, replace=False)
                    pool = [pool[i] for i in idx]

                # Rebuild splits
                tr = [g for g, s in pool if s == 'train']
                va = [g for g, s in pool if s == 'val']
                te = [g for g, s in pool if s == 'test']
                self.gids_train = np.array(sorted(tr), dtype=np.int64)
                self.gids_valid = np.array(sorted(va), dtype=np.int64)
                self.gids_test  = np.array(sorted(te), dtype=np.int64)

    # ---------------------------- Build graphs -----------------------------

    def _iter_graph_slices(self):
        """Yield (g, n, edges_local) for each graph in order, slicing edge.csv by e_counts."""
        start = 0
        for g in range(self.num_graphs):
            n = int(self.n_counts[g])
            e = int(self.e_counts[g])
            sub = self.edges_np_all[start:start + e]
            start += e
            if sub.size == 0:
                yield g, n, np.empty((2, 0), dtype=np.int64)
                continue

            u = sub[:, 0]; v = sub[:, 1]
            if self.dedup_undirected:
                lo = np.minimum(u, v)
                hi = np.maximum(u, v)
                uv = np.stack([lo, hi], axis=1)
                uv = uv[uv[:, 0] != uv[:, 1]]
                if uv.size > 0:
                    uv = np.unique(uv, axis=0)
                    u = np.concatenate([uv[:, 0], uv[:, 1]])
                    v = np.concatenate([uv[:, 1], uv[:, 0]])
                else:
                    u = np.array([], dtype=np.int64)
                    v = np.array([], dtype=np.int64)
            yield g, n, (np.stack([u, v], axis=0) if u.size > 0 else np.empty((2, 0), dtype=np.int64))

    def _build_stitched_union(self):
        # Build sets for fast membership checks
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
        for g, n, ei_local in self._iter_graph_slices():
            if g not in keep_any:
                continue

            # map local edges -> global with offset
            if ei_local.size > 0:
                src = ei_local[0] + offset
                dst = ei_local[1] + offset
                rows.append(src); cols.append(dst)

            # slice features by ORIGINAL global pointer (gptr_all)
            if self.node_features is not None:
                s, t = self.gptr_all[g], self.gptr_all[g + 1]
                Xg = self.node_features[s:t]
            else:
                # fallback: degree scalar
                deg = np.zeros(n, dtype=self.dtype_float)
                if ei_local.size > 0:
                    np.add.at(deg, ei_local[0], 1)
                    np.add.at(deg, ei_local[1], 1)
                Xg = deg.reshape(n, 1)

            x_list.append(Xg.astype(self.dtype_float, copy=False))

            # node labels inherit graph label
            glab = int(self.graph_label[g]) if self.graph_label is not None else 0
            node_labels_list.append(np.full(n, glab, dtype=np.int64))
            node2graph.append(np.full(n, g, dtype=np.int64))

            # node indices into stitched tensor
            span = np.arange(offset, offset + n, dtype=np.int64)
            if g in set_tr:
                idx_train.extend(span)
            elif g in set_va:
                idx_val.extend(span)
            elif g in set_te:
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

        adj = sp.coo_matrix((np.ones_like(rows, dtype=self.dtype_float), (rows, cols)),
                            shape=(N, N), dtype=self.dtype_float).tocsr()
        adj.setdiag(0); adj.eliminate_zeros()

        X = np.vstack(x_list).astype(self.dtype_float, copy=False)
        self.features = sp.csr_matrix(X, dtype=self.dtype_float)
        self.labels = np.concatenate(node_labels_list, axis=0)
        self.adj = adj

        self.idx_train = np.asarray(idx_train, dtype=np.int64)
        self.idx_val   = np.asarray(idx_val,   dtype=np.int64)
        self.idx_test  = np.asarray(idx_test,  dtype=np.int64)

        self.node2graph = np.concatenate(node2graph, axis=0)
        self.graph_ptr = graph_ptr
        self.graph_label = self.graph_label.astype(np.int64, copy=False)

    def _build_single_graph(self):
        which, gi = self.single_graph
        if which == "train":
            gids = self.gids_train
        elif which == "val":
            gids = self.gids_valid
        else:
            gids = self.gids_test
        if gi < 0 or gi >= len(gids):
            raise IndexError(f"single_graph index {gi} out of range for split '{which}'")

        g_pick = int(gids[gi])

        # Recover local edges and features for that graph
        # Find edge slice
        start = 0
        for g in range(self.num_graphs):
            n = int(self.n_counts[g]); e = int(self.e_counts[g])
            end = start + e
            if g == g_pick:
                sub = self.edges_np_all[start:end]
                if sub.size > 0:
                    u, v = sub[:, 0], sub[:, 1]
                    if self.dedup_undirected:
                        lo = np.minimum(u, v); hi = np.maximum(u, v)
                        pairs = np.stack([lo, hi], axis=1)
                        pairs = pairs[pairs[:, 0] != pairs[:, 1]]
                        if pairs.size > 0:
                            pairs = np.unique(pairs, axis=0)
                            u = np.concatenate([pairs[:, 0], pairs[:, 1]])
                            v = np.concatenate([pairs[:, 1], pairs[:, 0]])
                        else:
                            u = np.array([], dtype=np.int64); v = np.array([], dtype=np.int64)
                else:
                    u = np.array([], dtype=np.int64); v = np.array([], dtype=np.int64)

                adj = sp.coo_matrix((np.ones_like(u, dtype=self.dtype_float), (u, v)),
                                    shape=(n, n), dtype=self.dtype_float).tocsr()
                adj.setdiag(0); adj.eliminate_zeros()

                # features via original pointer
                if self.node_features is not None:
                    s, t = self.gptr_all[g], self.gptr_all[g + 1]
                    Xg = self.node_features[s:t]
                else:
                    deg = np.zeros(n, dtype=self.dtype_float)
                    if u.size > 0:
                        np.add.at(deg, u, 1); np.add.at(deg, v, 1)
                    Xg = deg.reshape(n, 1)
                features = sp.csr_matrix(Xg.astype(self.dtype_float, copy=False))

                glab = int(self.graph_label[g]) if self.graph_label is not None else 0
                labels = np.full(n, glab, dtype=np.int64)

                if (self.local_split_if_single == "10_10_80"):
                    idx_train, idx_val, idx_test = self._random_node_split(n, 0.10, 0.10, 0.80)
                else:
                    if which == "train":
                        idx_train = np.arange(n, dtype=np.int64); idx_val = np.array([], dtype=np.int64); idx_test = np.array([], dtype=np.int64)
                    elif which == "val":
                        idx_train = np.array([], dtype=np.int64); idx_val = np.arange(n, dtype=np.int64); idx_test = np.array([], dtype=np.int64)
                    else:
                        idx_train = np.array([], dtype=np.int64); idx_val = np.array([], dtype=np.int64); idx_test = np.arange(n, dtype=np.int64)

                self.adj = adj
                self.features = features
                self.labels = labels
                self.idx_train, self.idx_val, self.idx_test = idx_train, idx_val, idx_test
                self.node2graph = np.zeros(n, dtype=np.int64)
                self.graph_ptr = [0, n]
                self.graph_label = np.array([glab], dtype=np.int64)
                return
            start = end

        raise RuntimeError("single_graph index resolved to a graph that could not be sliced")

    # ---------------------------- Utilities -----------------------------

    def _random_node_split(self, N, p_tr, p_va, p_te):
        assert abs(p_tr + p_va + p_te - 1.0) < 1e-6
        idx = np.arange(N, dtype=np.int64)
        self.rng.shuffle(idx)
        n_tr = int(round(p_tr * N)); n_va = int(round(p_va * N))
        idx_tr = np.sort(idx[:n_tr]); idx_va = np.sort(idx[n_tr:n_tr + n_va]); idx_te = np.sort(idx[n_tr + n_va:])
        return idx_tr, idx_va, idx_te

    def _build_masks(self):
        N = self.labels.shape[0]
        def m(idx):
            z = np.zeros(N, dtype=bool); z[idx] = True; return z
        self.train_mask = m(self.idx_train)
        self.val_mask   = m(self.idx_val)
        self.test_mask  = m(self.idx_test)

        def take(idx):
            z = np.zeros_like(self.labels); z[idx] = self.labels[idx]; return z
        self.y_train = take(self.idx_train)
        self.y_val   = take(self.idx_val)
        self.y_test  = take(self.idx_test)

    def __repr__(self):
        N, F = self.features.shape
        return (f"molhiv-archive(adj_shape={self.adj.shape}, feature_shape=({N},{F}), "
                f"labels_shape={self.labels.shape}, "
                f"splits=({self.idx_train.size},{self.idx_val.size},{self.idx_test.size}), "
                f"graphs={len(self.graph_ptr)-1})")