"""
Phase 5 — Message Passing & GraphSAGE
======================================
Goal: understand how information flows through a graph before wiring up
the real GraphSAGE model in InnerSight UEBA.

Graph structure (6 nodes, 8 edges):

  0 -- 1 -- 2        (chain / triangle base)
  |  /               (edge 0-1, 1-2, 0-2 = triangle)
  3 -- 4 -- 5        (chain hanging off node 3)
  |
  (edge 0-3, 3-4, 4-5, 1-3, 2-4)

Full edges: 0-1, 1-2, 0-2, 0-3, 1-3, 3-4, 4-5, 2-4
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, HeteroData
from torch_geometric.nn import SAGEConv, HeteroConv

DIVIDER = "=" * 60

# ---------------------------------------------------------------------------
# 1. Build the toy graph by hand
# ---------------------------------------------------------------------------
print(DIVIDER)
print("1. GRAPH CONSTRUCTION")
print(DIVIDER)

# Hand-picked 3-dim feature vectors — easy to trace through the math
x = torch.tensor([
    [1.0, 0.0, 0.0],   # node 0
    [0.0, 1.0, 0.0],   # node 1
    [0.0, 0.0, 1.0],   # node 2
    [1.0, 1.0, 0.0],   # node 3
    [0.0, 1.0, 1.0],   # node 4
    [1.0, 0.0, 1.0],   # node 5
], dtype=torch.float)

# Edges listed as (src, dst) — undirected means we include both directions
edges = [
    (0, 1), (1, 0),
    (1, 2), (2, 1),
    (0, 2), (2, 0),
    (0, 3), (3, 0),
    (1, 3), (3, 1),
    (3, 4), (4, 3),
    (4, 5), (5, 4),
    (2, 4), (4, 2),
]
edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()  # shape (2, 16)

data = Data(x=x, edge_index=edge_index)
print(f"Nodes : {data.num_nodes}")
print(f"Edges : {data.num_edges} (undirected pairs, each listed twice)")
print(f"x shape : {x.shape}")
print()
print("Node features:")
for i, feat in enumerate(x):
    print(f"  node {i}: {feat.tolist()}")

# ---------------------------------------------------------------------------
# 2. Manual message passing (no PyG layers)
# ---------------------------------------------------------------------------
print()
print(DIVIDER)
print("2. MANUAL MESSAGE PASSING")
print(DIVIDER)

def get_neighbors(node_id, edge_index):
    """Return list of neighbor indices for node_id."""
    mask = edge_index[0] == node_id
    return edge_index[1][mask].tolist()

# Build adjacency view
print("\nNeighbor lists:")
for n in range(data.num_nodes):
    nbrs = get_neighbors(n, edge_index)
    print(f"  node {n}: neighbors = {nbrs}")

# Aggregate: mean of neighbor features
print("\nNeighbor feature means:")
neighbor_means = []
for n in range(data.num_nodes):
    nbrs = get_neighbors(n, edge_index)
    if nbrs:
        nbr_feats = x[nbrs]          # shape (k, 3)
        mean_feat = nbr_feats.mean(dim=0)
    else:
        mean_feat = torch.zeros(3)
    neighbor_means.append(mean_feat)
    print(f"  node {n}: {mean_feat.tolist()}")

neighbor_means = torch.stack(neighbor_means)  # (6, 3)

# Concatenate own features with neighbor mean  → (6, 6)
h_manual = torch.cat([x, neighbor_means], dim=1)  # (6, 6)
print(f"\nAfter concat [self | mean_nbr]: shape = {h_manual.shape}")

# Apply learned linear + ReLU
torch.manual_seed(42)
linear = nn.Linear(6, 8, bias=True)
h_manual_out = F.relu(linear(h_manual))  # (6, 8)

print(f"After Linear(6→8) + ReLU: shape = {h_manual_out.shape}")
print("\nNode 0 new representation:")
print(f"  {h_manual_out[0].detach().tolist()}")
print()
print("Interpretation for node 0:")
print("  Node 0 neighbors: 1, 2, 3")
print("  Its features [1,0,0] are now MIXED with avg of [0,1,0],[0,0,1],[1,1,0]")
print("  = avg neighbor: [0.333, 0.667, 0.333]")
print("  Concat: [1, 0, 0, 0.333, 0.667, 0.333]  → then learned weights shape it")

# ---------------------------------------------------------------------------
# 3. PyG SAGEConv — one layer
# ---------------------------------------------------------------------------
print()
print(DIVIDER)
print("3. PyG SAGEConv — SINGLE LAYER")
print(DIVIDER)

torch.manual_seed(42)
conv1_single = SAGEConv(in_channels=3, out_channels=8)
out_sage = conv1_single(x, edge_index)    # (6, 8)

print(f"SAGEConv(3 → 8) output shape: {out_sage.shape}")
print("\nPer-node 8-dim representations:")
for n in range(data.num_nodes):
    vals = [f"{v:.3f}" for v in out_sage[n].detach().tolist()]
    print(f"  node {n}: [{', '.join(vals)}]")

print()
print("Each node now has an 8-dim vector encoding itself + its 1-hop neighbors.")

# ---------------------------------------------------------------------------
# 4. Two stacked layers — 2-hop neighborhood
# ---------------------------------------------------------------------------
print()
print(DIVIDER)
print("4. STACKED SAGEConv — TWO LAYERS (2-hop)")
print(DIVIDER)

# 2-hop influence on node 0:
#   Hop-1 neighbors of 0: {1, 2, 3}
#   Hop-2 neighbors of 0: neighbors of {1,2,3} minus 0 itself
#     nbrs(1) = {0,2,3}
#     nbrs(2) = {0,1,4}
#     nbrs(3) = {0,1,4,5}? let's recheck: nbrs(3)={0,1,4}
#     union minus 0 = {1,2,3,4}  plus the hop-1 set
#   So after 2 layers node 0 sees: {0,1,2,3,4} — all except node 5

torch.manual_seed(42)
conv_l1 = SAGEConv(3, 8)
conv_l2 = SAGEConv(8, 4)

h = F.relu(conv_l1(x, edge_index))     # (6, 8)  — each node sees 1-hop
out_2hop = conv_l2(h, edge_index)      # (6, 4)  — each node now sees 2-hop

print(f"Layer 1 output shape : {h.shape}   (1-hop awareness)")
print(f"Layer 2 output shape : {out_2hop.shape}   (2-hop awareness)")
print()
print("2-hop influence diagram for node 0:")
print()
print("  Layer 1 aggregates:          Layer 2 aggregates:")
print("    0 ← {1, 2, 3}               0 ← {1, 2, 3}")
print("    1 ← {0, 2, 3}               but now h[1] already saw {0,2,3}")
print("    2 ← {0, 1, 4}               and h[2] already saw {0,1,4}")
print("    3 ← {0, 1, 4}               and h[3] already saw {0,1,4}")
print()
print("  => After 2 layers, node 0 has seen: {0, 1, 2, 3, 4}")
print("     Node 5 is 3 hops away — not yet visible.")
print()
print("Node 0 after 2-hop:")
vals = [f"{v:.3f}" for v in out_2hop[0].detach().tolist()]
print(f"  {vals}")

# ---------------------------------------------------------------------------
# 5. Heterogeneous version
# ---------------------------------------------------------------------------
print()
print(DIVIDER)
print("5. HETEROGENEOUS GRAPH — HeteroConv")
print(DIVIDER)

# Mini UEBA scenario: 3 users, 2 PCs
# user features: 4-dim  |  pc features: 3-dim
hdata = HeteroData()

hdata['user'].x = torch.tensor([
    [1.0, 0.0, 0.5, 0.1],   # user 0
    [0.0, 1.0, 0.2, 0.9],   # user 1
    [0.5, 0.5, 0.8, 0.3],   # user 2
], dtype=torch.float)

hdata['pc'].x = torch.tensor([
    [1.0, 0.0, 0.0],   # pc 0
    [0.0, 1.0, 0.0],   # pc 1
], dtype=torch.float)

# logon edges: user → pc
#   user0→pc0, user1→pc0, user1→pc1, user2→pc1
hdata['user', 'logon', 'pc'].edge_index = torch.tensor([
    [0, 1, 1, 2],
    [0, 0, 1, 1],
], dtype=torch.long)

# reverse edges: pc → user  (needed for users to aggregate from PCs)
hdata['pc', 'rev_logon', 'user'].edge_index = torch.tensor([
    [0, 0, 1, 1],
    [0, 1, 1, 2],
], dtype=torch.long)

print("HeteroData nodes:")
print(f"  user: {hdata['user'].x.shape}  (3 users, 4 features each)")
print(f"  pc  : {hdata['pc'].x.shape}   (2 PCs,   3 features each)")
print("Edges:")
print(f"  user→pc (logon)    : {hdata['user','logon','pc'].edge_index.shape}")
print(f"  pc→user (rev_logon): {hdata['pc','rev_logon','user'].edge_index.shape}")

torch.manual_seed(42)
hetero_conv = HeteroConv({
    ('user', 'logon', 'pc'):      SAGEConv((-1, -1), 8),
    ('pc', 'rev_logon', 'user'):  SAGEConv((-1, -1), 8),
}, aggr='mean')

x_dict = {'user': hdata['user'].x, 'pc': hdata['pc'].x}
edge_index_dict = {
    ('user', 'logon', 'pc'):      hdata['user', 'logon', 'pc'].edge_index,
    ('pc', 'rev_logon', 'user'):  hdata['pc', 'rev_logon', 'user'].edge_index,
}

out_dict = hetero_conv(x_dict, edge_index_dict)

print()
print("HeteroConv output shapes:")
for node_type, emb in out_dict.items():
    print(f"  '{node_type}': {emb.shape}")

print()
print("User embeddings (each user saw its own PCs):")
for i, emb in enumerate(out_dict['user'].detach()):
    vals = [f"{v:.3f}" for v in emb.tolist()]
    print(f"  user {i}: [{', '.join(vals)}]")

print()
print("PC embeddings (each PC saw its logon users):")
for i, emb in enumerate(out_dict['pc'].detach()):
    vals = [f"{v:.3f}" for v in emb.tolist()]
    print(f"  pc   {i}: [{', '.join(vals)}]")

print()
print("user1 logged onto both pc0 and pc1, so its embedding mixes features from both.")
print("pc0 was accessed by user0 and user1, so its embedding reflects both users.")

# ---------------------------------------------------------------------------
# 6. Summary
# ---------------------------------------------------------------------------
print()
print(DIVIDER)
print("6. SUMMARY")
print(DIVIDER)
print()
print("Message passing: collect neighbors -> aggregate -> update")
print("SAGEConv does this with learned weights")
print("HeteroConv runs separate SAGEConv per edge type")
print("Stack 2-3 layers = 2-3 hop neighborhood awareness")
print()
print("In InnerSight UEBA terms:")
print("  - Nodes = users, PCs, processes, files")
print("  - Edges = logon, exec, access, network")
print("  - After 2 layers: a user node sees all PCs it touched,")
print("    plus all OTHER users who shared those PCs")
print("  - That 2-hop context is what lets the model spot lateral movement")
