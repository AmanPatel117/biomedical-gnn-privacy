# GNN_Privacy_for_Biomedical_Data

## LINKTELLER

[LINKTELLER](https://arxiv.org/abs/2108.06504), a query-based link re-identification attack against Graph Neural Networks (GNNs). The core idea: if node v is connected to node u, then slightly up-weighting the features of v should noticeably change u’s prediction. By probing a black-box GNN API and measuring these influence signals, we can rank node pairs and recover private edges.


### 🧠 How LINKTELLER Works (Short)

1. **Query the API:**  
   Query the black-box GNN API on a set of nodes \( V^{(I)} \) to obtain logits \( P \).

2. **Perturb each node:**  
   For each node \( v \), slightly up-weight its features:  
   $
   X' = X; \quad X'_v \leftarrow (1 + \Delta) X_v
   $

3. **Re-query and compute influence:**  
   Query the API again to get \( P' \), and compute the *influence matrix*:  
   \[
   I_v = \frac{P' - P}{\Delta}
   \]

4. **Measure pairwise influence:**  
   Use the \( \ell_2 \)-norm of the influence row as the signal from node \( v \) to node \( u \):  
   \[
   \| I_v[u, :] \|_2
   \]

5. **Symmetrize the influence:**  
   For each unordered pair \( \{u, v\} \), take either the **max** or **sum** of the bidirectional influences.

6. **Rank and select top edges:**  
   Sort all node pairs by influence magnitude and take the top  
   \[
   m = \hat{k} \, \frac{n(n-1)}{2}
   \]  
   as the predicted edges, where \( \hat{k} \) is the **density belief** controlling how dense the reconstructed graph should be.
