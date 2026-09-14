from model import GNNModel, train
from data import build_dataset, sample_ego_network, handcrafted_features, rng
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
import matplotlib
matplotlib.use("Agg")


def evaluate(y_true, y_proba, threshold=0.5):
    y_pred = (np.array(y_proba) >= threshold).astype(int)
    return {
        "AUC-ROC": roc_auc_score(y_true, y_proba),
        "F1": f1_score(y_true, y_pred, zero_division=0),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
    }


def main():
    G, attrs, y = build_dataset()
    clustering = nx.clustering(G)

    eligible = [v for v in G.nodes() if G.degree(v) >= 2]
    rng.shuffle(eligible)
    n_tr, n_va = int(.7 * len(eligible)), int(.15 * len(eligible))
    train_v, val_v, test_v = eligible[:n_tr], eligible[n_tr:n_tr +
                                                       n_va], eligible[n_tr + n_va:]
    y_train = np.array([y[v] for v in train_v])
    y_val = np.array([y[v] for v in val_v])
    y_test = np.array([y[v] for v in test_v])
    print(f"nodes={G.number_of_nodes()} train={len(train_v)} val={len(val_v)} test={len(test_v)} "
          f"positive_rate={y.mean():.2f}")

    def build(vs):
        inst = [sample_ego_network(G, v, y) for v in vs]
        hand = np.array([handcrafted_features(
            G, v, y, attrs, clustering) for v in vs])
        return inst, hand

    train_inst, train_hand = build(train_v)
    val_inst, val_hand = build(val_v)
    test_inst, test_hand = build(test_v)

    lr_clf = LogisticRegression(max_iter=1000).fit(train_hand, y_train)
    lr_proba = lr_clf.predict_proba(test_hand)[:, 1]
    lr_metrics = evaluate(y_test, lr_proba)

    plain = GNNModel(
        f_in=2, hand_dim=train_hand.shape[1], use_handcrafted=False)
    train(plain, train_inst, train_hand, y_train, epochs=15)
    plain_proba = [plain.forward(i, h)[0]
                   for i, h in zip(test_inst, test_hand)]
    plain_metrics = evaluate(y_test, plain_proba)

    full = GNNModel(f_in=2, hand_dim=train_hand.shape[1], use_handcrafted=True)
    losses = train(full, train_inst, train_hand, y_train, epochs=20)
    full_proba = [full.forward(i, h)[0] for i, h in zip(test_inst, test_hand)]
    full_metrics = evaluate(y_test, full_proba)

    full.save_weights("model_weights.npz")

    for name, m in [("Logistic Regression", lr_metrics),
                    ("Plain GCN", plain_metrics),
                    ("Full model (GCN + handcrafted)", full_metrics)]:
        print(f"{name:35s} " + "  ".join(f"{k}={v:.3f}" for k, v in m.items()))

    plt.figure(figsize=(5, 3.5))
    plt.plot(losses)
    plt.xlabel("epoch")
    plt.ylabel("train BCE loss")
    plt.title("Full model training curve")
    plt.tight_layout()
    plt.savefig("training_curve.png", dpi=130)


if __name__ == "__main__":
    main()
