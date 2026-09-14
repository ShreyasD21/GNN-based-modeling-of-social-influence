import numpy as np

SEED = 0
rng = np.random.default_rng(SEED)


class GNNModel:
    def __init__(self, f_in, hand_dim, hidden=16, use_handcrafted=True):
        self.use_handcrafted = use_handcrafted
        def lim(a, b): return np.sqrt(6 / (a + b))
        self.W1 = rng.uniform(-lim(f_in, hidden),
                              lim(f_in, hidden), (f_in, hidden))
        self.b1 = np.zeros(hidden)
        self.W2 = rng.uniform(-lim(hidden, hidden),
                              lim(hidden, hidden), (hidden, hidden))
        self.b2 = np.zeros(hidden)
        head_in = hidden * 2 + (hand_dim if use_handcrafted else 0)
        self.W3 = rng.uniform(-lim(head_in, hidden),
                              lim(head_in, hidden), (head_in, hidden))
        self.b3 = np.zeros(hidden)
        self.W4 = rng.uniform(-lim(hidden, 1), lim(hidden, 1), (hidden, 1))
        self.b4 = np.zeros(1)
        self._adam = {k: {"m": 0, "v": 0, "t": 0} for k in
                      ["W1", "b1", "W2", "b2", "W3", "b3", "W4", "b4"]}

    def save_weights(self, path):
        """Save all trained parameters to a .npz file (deliverable: trained model weights)."""
        np.savez(path, W1=self.W1, b1=self.b1, W2=self.W2, b2=self.b2,
                 W3=self.W3, b3=self.b3, W4=self.W4, b4=self.b4)

    def load_weights(self, path):
        """Load previously-saved parameters produced by save_weights()."""
        d = np.load(path)
        self.W1, self.b1 = d["W1"], d["b1"]
        self.W2, self.b2 = d["W2"], d["b2"]
        self.W3, self.b3 = d["W3"], d["b3"]
        self.W4, self.b4 = d["W4"], d["b4"]

    @staticmethod
    def _relu(x): return np.maximum(x, 0)

    @staticmethod
    def _instance_norm(H, n_real, eps=1e-5):
        Hr = H[:n_real]
        mean, var = Hr.mean(0, keepdims=True), Hr.var(0, keepdims=True)
        std = np.sqrt(var + eps)
        Hn = H.copy()
        Hn[:n_real] = (Hr - mean) / std
        return Hn, (Hr, mean, std, n_real)

    def forward(self, inst, hand_feat):
        A, X, n_real = inst["A"], inst["X"], inst["n_real"]
        AX1 = A @ X
        Z1 = AX1 @ self.W1 + self.b1
        H1 = self._relu(Z1)
        AX2 = A @ H1
        Z2 = AX2 @ self.W2 + self.b2
        H2 = self._relu(Z2)
        Hn, norm_cache = self._instance_norm(H2, n_real)

        ego = Hn[0]
        neigh = Hn[1:n_real].mean(0) if n_real > 1 else np.zeros_like(ego)
        readout = np.concatenate([ego, neigh])
        fused = np.concatenate([readout, hand_feat]
                               ) if self.use_handcrafted else readout

        z3 = fused @ self.W3 + self.b3
        a3 = self._relu(z3)
        z4 = a3 @ self.W4 + self.b4
        p = 1 / (1 + np.exp(-np.clip(z4[0], -30, 30)))

        cache = dict(A=A, X=X, AX1=AX1, Z1=Z1, H1=H1, AX2=AX2, Z2=Z2, H2=H2,
                     norm_cache=norm_cache, readout=readout, fused=fused,
                     z3=z3, a3=a3, p=p, n_real=n_real)
        return p, cache

    def backward(self, cache, y_true, lr):
        dz4 = np.array([cache["p"] - y_true])
        dW4 = np.outer(cache["a3"], dz4)
        db4 = dz4
        da3 = dz4 @ self.W4.T
        dz3 = da3 * (cache["z3"] > 0)
        dW3 = np.outer(cache["fused"], dz3)
        db3 = dz3
        dfused = dz3 @ self.W3.T

        hidden = len(cache["readout"]) // 2
        dreadout = dfused[:len(cache["readout"])]
        dego, dneigh = dreadout[:hidden], dreadout[hidden:]

        dHn = np.zeros_like(cache["H2"])
        dHn[0] = dego
        n_real = cache["n_real"]
        if n_real > 1:
            dHn[1:n_real] = dneigh / (n_real - 1)

        Hr, mean, std, N = cache["norm_cache"]
        dHnr = dHn[:N]
        x_mu = Hr - mean
        dvar = np.sum(dHnr * x_mu * -0.5 * std ** -3, axis=0, keepdims=True)
        dmean = np.sum(dHnr * -1.0 / std, axis=0, keepdims=True) + \
            dvar * np.mean(-2.0 * x_mu, axis=0, keepdims=True)
        dH2 = np.zeros_like(cache["H2"])
        dH2[:N] = dHnr / std + dvar * 2.0 * x_mu / N + dmean / N

        dZ2 = dH2 * (cache["Z2"] > 0)
        dW2 = cache["AX2"].T @ dZ2
        db2 = dZ2.sum(0)
        dH1 = (cache["A"].T @ (dZ2 @ self.W2.T))

        dZ1 = dH1 * (cache["Z1"] > 0)
        dW1 = cache["AX1"].T @ dZ1
        db1 = dZ1.sum(0)

        grads = dict(W1=dW1, b1=db1, W2=dW2, b2=db2,
                     W3=dW3, b3=db3, W4=dW4, b4=db4)
        for k, g in grads.items():
            setattr(self, k, self._adam_step(
                getattr(self, k), g, self._adam[k], lr))

    @staticmethod
    def _adam_step(param, grad, state, lr, b1=0.9, b2=0.999, eps=1e-8):
        state["t"] += 1
        state["m"] = b1 * state["m"] + (1 - b1) * grad
        state["v"] = b2 * state["v"] + (1 - b2) * grad ** 2
        m_hat = state["m"] / (1 - b1 ** state["t"])
        v_hat = state["v"] / (1 - b2 ** state["t"])
        return param - lr * m_hat / (np.sqrt(v_hat) + eps)


def bce(p, y, eps=1e-9):
    p = np.clip(p, eps, 1 - eps)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def train(model, instances, hand_feats, labels, epochs=20, lr=0.01):
    losses = []
    for ep in range(epochs):
        order = rng.permutation(len(instances))
        ep_loss = []
        for i in order:
            p, cache = model.forward(instances[i], hand_feats[i])
            ep_loss.append(bce(p, labels[i]))
            model.backward(cache, labels[i], lr)
        losses.append(np.mean(ep_loss))
    return losses
