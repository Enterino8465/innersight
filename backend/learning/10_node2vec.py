"""
Node2Vec on a toy community graph.

Demonstrates how biased random walks (p, q parameters) shape the learned
embeddings — then explains why these choices matter for insider-threat detection.
"""

import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torch_geometric.data import Data
from torch_geometric.nn import Node2Vec
from sklearn.manifold import TSNE
import numpy as np

# ── 1. Build toy graph with 3 communities ────────────────────────────────────

def make_community_graph(community_size=10, p_intra=0.85, p_inter=0.03, seed=42):
    """
    Three communities of `community_size` nodes each.
    Edges are sampled with high probability within a community (p_intra)
    and low probability between communities (p_inter).
    """
    rng = np.random.default_rng(seed)
    n = community_size * 3
    labels = torch.tensor([i // community_size for i in range(n)], dtype=torch.long)

    src, dst = [], []
    for i in range(n):
        for j in range(i + 1, n):
            same = (labels[i] == labels[j]).item()
            prob = p_intra if same else p_inter
            if rng.random() < prob:
                src += [i, j]
                dst += [j, i]

    edge_index = torch.tensor([src, dst], dtype=torch.long)
    return Data(edge_index=edge_index, num_nodes=n), labels


data, labels = make_community_graph()
print(f"Nodes: {data.num_nodes}  |  Edges (undirected): {data.num_edges // 2}")


# ── 2. Train Node2Vec ─────────────────────────────────────────────────────────

def train_node2vec(data, p=1.0, q=1.0, epochs=100, seed=0):
    torch.manual_seed(seed)
    model = Node2Vec(
        edge_index=data.edge_index,
        embedding_dim=32,
        walk_length=20,
        context_size=10,
        walks_per_node=10,
        num_negative_samples=1,
        p=p,
        q=q,
        sparse=True,
        num_nodes=data.num_nodes,
    )
    loader = model.loader(batch_size=64, shuffle=True, num_workers=0)
    optimizer = torch.optim.SparseAdam(list(model.parameters()), lr=0.01)

    model.train()
    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        for pos_rw, neg_rw in loader:
            optimizer.zero_grad()
            loss = model.loss(pos_rw, neg_rw)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        if epoch % 20 == 0:
            print(f"  epoch {epoch:3d}  loss {total_loss:.4f}")

    model.eval()
    with torch.no_grad():
        embeddings = model()
    print(f"  embedding shape: {embeddings.shape}")
    return embeddings.detach().numpy()


# ── 3. t-SNE helper ──────────────────────────────────────────────────────────

COLORS = ["#e6194b", "#3cb44b", "#4363d8"]
COMMUNITY_NAMES = ["Community A", "Community B", "Community C"]


def tsne_plot(ax, embeddings, labels, title):
    tsne = TSNE(n_components=2, perplexity=5, random_state=42, init="pca")
    coords = tsne.fit_transform(embeddings)
    for c in range(3):
        mask = labels.numpy() == c
        ax.scatter(
            coords[mask, 0], coords[mask, 1],
            c=COLORS[c], label=COMMUNITY_NAMES[c],
            s=80, edgecolors="k", linewidths=0.4,
        )
    ax.set_title(title, fontsize=10, fontweight="bold")
    ax.legend(fontsize=8)
    ax.set_xticks([])
    ax.set_yticks([])


# ── 4. Single run (p=1, q=1) ─────────────────────────────────────────────────

print("\n=== p=1.0, q=1.0 (unbiased) ===")
emb_unbiased = train_node2vec(data, p=1.0, q=1.0)

fig, ax = plt.subplots(figsize=(5, 5))
tsne_plot(ax, emb_unbiased, labels, "Node2Vec (p=1, q=1) — unbiased walk")
fig.tight_layout()
fig.savefig(
    "/Users/michaelkuksov/Developer/innersight/backend/learning/node2vec_toy.png",
    dpi=150,
)
plt.close(fig)
print("Saved: node2vec_toy.png")


# ── 5. Comparison: q=0.5 (DFS) vs q=2.0 (BFS) ───────────────────────────────

print("\n=== p=1.0, q=0.5 (DFS-like) ===")
emb_dfs = train_node2vec(data, p=1.0, q=0.5)

print("\n=== p=1.0, q=2.0 (BFS-like) ===")
emb_bfs = train_node2vec(data, p=1.0, q=2.0)

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
tsne_plot(axes[0], emb_unbiased, labels, "p=1, q=1\nUnbiased — general similarity")
tsne_plot(axes[1], emb_dfs,      labels, "p=1, q=0.5\nDFS-like — global roles")
tsne_plot(axes[2], emb_bfs,      labels, "p=1, q=2.0\nBFS-like — local neighborhoods")
fig.suptitle("Node2Vec: effect of the q parameter on community separation", fontsize=12)
fig.tight_layout()
fig.savefig(
    "/Users/michaelkuksov/Developer/innersight/backend/learning/node2vec_comparison.png",
    dpi=150,
)
plt.close(fig)
print("\nSaved: node2vec_comparison.png")


# ── 6. Explanation ───────────────────────────────────────────────────────────

print("""
─────────────────────────────────────────────────────────────────────────────
Node2Vec parameter intuition
─────────────────────────────────────────────────────────────────────────────
p=1, q=1 : unbiased random walk — general structural similarity
p=1, q<1 : DFS-like — captures global roles (e.g., all managers cluster)
p=1, q>1 : BFS-like — captures local neighborhoods (e.g., same-team users)

For insider threat (CERT graph):
  We want q slightly < 1 (e.g., 0.5–0.8).
  This lets the walk venture across departments while still honouring local
  team structure — so an anomalous user who starts behaving like a different
  department's role will land in the wrong cluster and stand out as an outlier.
─────────────────────────────────────────────────────────────────────────────
""")
