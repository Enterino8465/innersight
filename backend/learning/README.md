# Learning Scripts (Archived)

These scripts were written during the initial learning phases of the InnerSight project. They document the progression from raw PyTorch basics to graph neural networks.

## Status

**These are NOT runnable production code.** Several scripts import from modules that have been deleted during the Phase 0 cleanup (b3_network, b4_loss, b5_backprop).

## Scripts

| # | File | Topic | Runnable? |
|---|------|-------|-----------|
| 01 | device_check.py | CUDA/MPS/CPU detection | Yes |
| 02 | tensors.py | Tensor basics + Network import | No (imports deleted Network) |
| 03 | autograd.py | Autograd + backprop demo | No (imports deleted modules) |
| 04 | nn_module.py | nn.Module patterns | No (imports deleted Network) |
| 05 | loss_and_optimizer.py | Loss functions + Adam | Yes |
| 06 | dataloader.py | DataLoader patterns | Yes (with INNERSIGHT_DATA_DIR set) |
| 07 | full_training_loop.py | End-to-end MLP training | Yes (with INNERSIGHT_DATA_DIR set) |
| 08 | save_load.py | Checkpoint save/load | Yes |
| 09 | pyg_verify.py | PyG installation check | Yes |
| 10 | node2vec.py | Node2Vec on toy graph | Yes |
| 11 | message_passing.py | GNN message passing | Yes |

## Note

The production model code lives in `backend/models/`. These scripts are kept for reference only.
