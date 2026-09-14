# Social Influence Prediction — GNN-Based Modeling of Peer Behavior

A small GNN pipeline that predicts whether a user will perform an action
(e.g. adopt a product, repost, view an ad) based on their social
neighborhood's behavior — following the DeepInf-style social-influence
prediction setup.


## Setup

```bash
pip install -r requirements.txt
```

## Run

```bash
python3 main.py
```

This will:
1. Build a synthetic social graph and simulate a cascade to generate labels.
2. Sample ego networks and extract handcrafted features.
3. Train three models: Logistic Regression, Plain GCN, and the Full model
   (GCN + handcrafted features).
4. Print AUC-ROC / F1 / Precision / Recall for each on a held-out test set.
5. Save the trained full model's weights to `model_weights.npz`.
6. Save a training-loss plot to `training_curve.png`.

## Loading saved weights

```python
from model import GNNModel

model = GNNModel(f_in=2, hand_dim=9, use_handcrafted=True)
model.load_weights("model_weights.npz")
```

## What it does (pipeline)

| Step | File | Description |
|---|---|---|
| (a) Ego-network sampling | `data.py` | BFS up to 2 hops around each user, fixed-size (16 nodes) |
| (b) Graph encoding | `model.py` | 2-layer GCN, hand-written in NumPy (own forward + backward pass) |
| (c) Instance normalization | `model.py` | Per-instance node-embedding normalization before readout |
| (d) Handcrafted features | `data.py` | Demographics, activity, degree, clustering, neighbor-influence ratio |
| (e) Prediction head | `model.py` | 1-hidden-layer MLP + sigmoid |
| (f)/(g) Training | `model.py` | Binary cross-entropy, backprop end-to-end, Adam optimizer |

## Baselines

- **Logistic Regression** — handcrafted features only, no graph structure.
- **Plain GCN** — same encoder as the full model, but without handcrafted-feature fusion.
- **Full model** — GCN embedding + handcrafted features (proposed approach).

## Typical results (held-out test set)

| Model | AUC-ROC | F1 |
|---|---|---|
| Logistic Regression | 0.79 | 0.36 |
| Plain GCN | 0.99 | 0.89 |
| Full model | 0.99 | 0.94 |

GNN-based models substantially outperform logistic regression because
message passing lets a node directly read its neighbors' action states —
the exact signal an Independent Cascade process propagates through the
network. Handcrafted features give a further, smaller boost on top.

## Dataset note

The original DeepInf datasets (OAG / Digg / Weibo) require downloading
an external repository and weren't available in the environment this
was built in, so a **synthetic graph with a simulated cascade** is used
instead — the fallback the problem statement explicitly allows. A
Barabási–Albert graph mimics a realistic social network, and an
Independent Cascade model produces each node's binary action label.

## Scope notes

Built without internet access, so PyTorch/PyG/DGL weren't available —
the GCN layer, instance norm, and optimizer are implemented from scratch
in NumPy with hand-derived backward passes. Kept to 3 files, 3 models
(no DeepWalk baseline, no formal ablation studies) to fit a short
project timeline. Swapping in `torch_geometric.nn.GCNConv` later would
be a straightforward drop-in replacement for the `GNNModel` class.