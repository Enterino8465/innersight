import torch
import torch_geometric
from torch_geometric.data import HeteroData
from torch_geometric.loader import DataLoader

print(f"PyG version: {torch_geometric.__version__}")
print(f"Torch version: {torch.__version__}")
print()

# Build a small heterogeneous graph
data = HeteroData()

data['user'].x = torch.randn(10, 5)   # 10 user nodes, 5 features
data['pc'].x   = torch.randn(20, 3)   # 20 pc nodes,   3 features

# 30 random ('user', 'logon', 'pc') edges
src = torch.randint(0, 10, (30,))
dst = torch.randint(0, 20, (30,))
data['user', 'logon', 'pc'].edge_index = torch.stack([src, dst])

print(data)
print()
print(f"user nodes : {data['user'].num_nodes}  |  features: {data['user'].x.shape}")
print(f"pc nodes   : {data['pc'].num_nodes}  |  features: {data['pc'].x.shape}")
print(f"logon edges: {data['user', 'logon', 'pc'].num_edges}")
print()

# DataLoader — wraps HeteroData in a list so it batches as a single graph
loader = DataLoader([data], batch_size=1)
for batch in loader:
    print(f"Batch type : {type(batch).__name__}")
    print(f"  user nodes in batch : {batch['user'].num_nodes}")
    print(f"  pc nodes in batch   : {batch['pc'].num_nodes}")
    print(f"  logon edges in batch: {batch['user', 'logon', 'pc'].num_edges}")

print()
print("PyG is working correctly")
