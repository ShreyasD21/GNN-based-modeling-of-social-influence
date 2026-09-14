import numpy as np
import networkx as nx

SEED = 0
rng = np.random.default_rng(SEED)



#DATASET: synthetic social graph + simulated IC cascade
def build_dataset(n_nodes=1200, m_attach=4, n_seeds=15, base_p=0.06):
    G = nx.barabasi_albert_graph(n_nodes, m_attach, seed=SEED)

    deg = np.array([d for _, d in G.degree()], dtype=float)
    deg_norm = (deg - deg.min()) / (deg.max() - deg.min() + 1e-9)
    attrs = {
        "age_group": rng.integers(0, 4, size=n_nodes),
        "activity": np.clip(rng.beta(2, 2, size=n_nodes) * 0.7 + 0.3 * deg_norm, 0, 1),
        "hist_freq": np.clip(rng.beta(2, 3, size=n_nodes) * 0.6 + 0.4 * deg_norm, 0, 1),
    }

    y = np.zeros(n_nodes, dtype=int)
    seeds = rng.choice(n_nodes, size=n_seeds, replace=False)
    y[seeds] = 1
    frontier, tried = list(seeds), set()
    while frontier:
        nxt = []
        for u in frontier:
            for v in G.neighbors(u):
                if y[v] or (u, v) in tried:
                    continue
                tried.add((u, v))
                p = np.clip(base_p * (1 + attrs["activity"][v]) * (1 + attrs["hist_freq"][v]), 0, 0.9)
                if rng.random() < p:
                    y[v] = 1
                    nxt.append(v)
        frontier = nxt

    return G, attrs, y



#Ego network sampling: BFS up to k hops keeping N fixed
def sample_ego_network(G, v, y, k_hops=2, n_nodes=16):
    nodes, frontier, seen = [v], [v], {v}
    for _ in range(k_hops):
        nxt = []
        for u in frontier:
            for w in G.neighbors(u):
                if w not in seen:
                    seen.add(w); nxt.append(w); nodes.append(w)
                    if len(nodes) >= n_nodes:
                        break
            if len(nodes) >= n_nodes:
                break
        frontier = nxt
        if not frontier or len(nodes) >= n_nodes:
            break
    nodes = nodes[:n_nodes]
    n_real = len(nodes)
    idx = {u: i for i, u in enumerate(nodes)}

    A = np.zeros((n_real, n_real))
    for u, w in G.subgraph(nodes).edges():
        A[idx[u], idx[w]] = A[idx[w], idx[u]] = 1

    max_deg = max(dict(G.degree()).values())
    X = np.array([[float(y[u]), G.degree(u) / max_deg] for u in nodes])  

    # symmetric normalization
    A_hat = A + np.eye(n_real)
    d = A_hat.sum(1)
    d_inv_sqrt = np.zeros_like(d)
    d_inv_sqrt[d > 0] = d[d > 0] ** -0.5
    D = np.diag(d_inv_sqrt)
    A_norm = D @ A_hat @ D

    N = n_nodes
    A_pad = np.zeros((N, N)); A_pad[:n_real, :n_real] = A_norm
    X_pad = np.zeros((N, X.shape[1])); X_pad[:n_real] = X
    return {"A": A_pad, "X": X_pad, "n_real": n_real}



# Handcrafted features
def handcrafted_features(G, v, y, attrs, clustering):
    neighbors = list(G.neighbors(v))
    degree = len(neighbors)
    n_active = sum(y[u] for u in neighbors)
    influence_ratio = n_active / degree if degree else 0.0
    return np.array([
        *np.eye(4)[attrs["age_group"][v]],   
        attrs["activity"][v],                
        attrs["hist_freq"][v],               
        degree / 50.0,                       
        clustering[v],                       
        influence_ratio,                     
    ])
