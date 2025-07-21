# Matrix Methods: Theoretical Foundations of UltraGCN and Its Variants

This README focuses exclusively on the theoretical details covered in the project report. All implementation, empirical evaluations, and reproducibility instructions are contained in the accompanying Jupyter notebook.

---

## 1. Introduction to Matrix Methods for Recommendation

Collaborative filtering in recommendation systems often leverages low-rank matrix factorization or graph-based propagation to capture latent preference structures. UltraGCN unifies these approaches by interpreting user–item interactions as a bipartite graph and deriving an infinite-depth graph convolution operator with closed-form propagation.

## 2. UltraGCN Theoretical Framework

### 2.1 Graph Representation

* **Bipartite Graph**
  Users \$U\$ and items \$I\$ form nodes; edges \$E\$ represent observed interactions.
* **Adjacency Matrices**
  \$R\in\mathbb{R}^{|U|\times|I|}\$ captures user–item ratings; normalized adjacency \$\tilde{A}=D^{-1/2}\[\begin{smallmatrix}0 & R \ R^\top & 0\end{smallmatrix}]D^{-1/2}\$.

### 2.2 Infinite-Layer Propagation

* **Standard GCN** stacks \$K\$ layers of propagation:
  \$H^{(k+1)}=\tilde{A}H^{(k)}W^{(k)}\$.
* **UltraGCN** takes \$K\to\infty\$ and derives a *closed-form* propagated embedding:
  $H^*=\sum_{k=0}^\infty\alpha_k\,\tilde{A}^kX = f(\tilde{A})\,X$
  where \${\alpha\_k}\$ are decaying coefficients and \$f(\tilde{A})\$ admits an analytical matrix function.

### 2.3 Graph-Constraint Regularization

* **Motivation**: enforce that embeddings respect observed edges more strongly than unobserved ones.
* **Loss Term**:
  $L_{graph}=\sum_{(u,i)\in E}\log\sigma\bigl(\mathbf{e}_u^\top \mathbf{e}_i\bigr) + \lambda\sum_{(u,i)\notin E}\log\bigl(1-\sigma(\mathbf{e}_u^\top \mathbf{e}_i)\bigr)$
* **Properties**: encourages large dot-products for positive edges and small for negatives; \$\sigma\$ denotes the sigmoid function.

---

## 3. Variants and Theoretical Analyses

### 3.1 UltraGCN without Graph-Constraint Loss

* **Modification**: set \$L\_{graph}=0\$.
* **Effect**: reduces to pure infinite-depth smoothing; theoretical analysis shows decreased edge discrimination.

### 3.2 Removal of Log-Sigmoid Activation

* **Modification**: replace \$\log\sigma(x)\$ with a linear penalty \$w,x\$.
* **Analysis**: gradients become constant w\.r.t.\ \$x\$, impacting convergence and reducing regularization strength on hard negatives.

---

## 4. Theoretical Insights

* **Connection to Truncated SVD**
  Truncated SVD solves \$\min\_{\mathrm{rank}(M)=r}|R-M|\_F^2\$; UltraGCN can be viewed as a weighted spectral filter on \$R\$’s singular values.

* **Spectral Smoothing vs.\ Oversmoothing**
  Infinite-depth propagation increases embedding smoothness; prove conditions under which node embeddings remain distinguishable (avoiding oversmoothing).

* **Generalization Bounds**
  Under mild assumptions on graph sparsity and spectral decay, derive upper bounds on recommendation error as a function of regularization parameters.

---

## 5. Conclusion and Future Directions

This report establishes a rigorous theoretical foundation for UltraGCN and its variants, highlighting how closed-form infinite-layer propagation and graph-constraint losses influence embedding quality. Future work includes tighter generalization guarantees and exploration of novel spectral filters.
