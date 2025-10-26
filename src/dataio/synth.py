
import numpy as np
import os, json

def gen_dag(d, exp_edges=2, seed=0):
    rng = np.random.default_rng(seed)
    A = np.zeros((d,d))
    for i in range(d):
        for j in range(i+1, d):
            if rng.random() < exp_edges / d:
                A[i,j] = 1.0
    return A

def simulate_scm(A, T=512, lags=1, seed=0):
    rng = np.random.default_rng(seed)
    d = A.shape[0]
    S = np.zeros((T, d))
    for t in range(lags, T):
        base = rng.normal(0, 0.1, size=d)
        for j in range(d):
            parents = np.where(A[:,j]>0)[0]
            val = 0.0
            for p in parents:
                val += np.tanh(S[t-1, p])
            S[t,j] = 0.8*S[t-1,j] + 0.5*val + base[j]
    return S

def apply_corruptions(S, regimes=2, miss_rate=0.4, snr_db=10, seed=0):
    rng = np.random.default_rng(seed)
    T, d = S.shape
    # regime per time block
    block = T // regimes
    e = np.zeros(T, dtype=int)
    for r in range(regimes):
        e[r*block:(r+1)*block] = r
    e[(regimes*block):] = regimes-1
    # biases per regime
    B = rng.lognormal(mean=0.0, sigma=0.15, size=(regimes, d))
    b = rng.normal(0, 0.1, size=(regimes, d))
    X = np.zeros_like(S)
    for t in range(T):
        rr = e[t]
        mult = B[rr]
        add = b[rr]
        clean = S[t]*mult + add
        # heteroscedastic noise
        sig = 0.1 + 0.1*np.abs(clean)
        noise = rng.normal(0, sig, size=d)
        X[t] = clean + noise
    # missingness (MCAR simple)
    M = (rng.random(size=X.shape) > miss_rate).astype(float)
    X_miss = X.copy()
    X_miss[M==0] = 0.0
    return X_miss, M, e, (B,b)

def window_data(X, M, e, Tw=128, Ts=64):
    T, d = X.shape
    windows = []
    masks = []
    envs = []
    for start in range(0, T - Tw + 1, Ts):
        end = start + Tw
        windows.append(X[start:end])
        masks.append(M[start:end])
        # regime label by majority
        rr = np.bincount(e[start:end]).argmax()
        envs.append(rr)
    return np.stack(windows), np.stack(masks), np.array(envs)

def build_synth(out_root, d=10, T=512, regimes=2, Tw=128, Ts=64, seed=0):
    A = gen_dag(d, exp_edges=2, seed=seed)
    S = simulate_scm(A, T=T, lags=1, seed=seed)
    X, M, e, Bb = apply_corruptions(S, regimes=regimes, miss_rate=0.4, snr_db=10, seed=seed+1)
    Xw, Mw, ew = window_data(X, M, e, Tw=Tw, Ts=Ts)
    Sw, _, _ = window_data(S, np.ones_like(S), e, Tw=Tw, Ts=Ts)
    os.makedirs(out_root, exist_ok=True)
    np.save(os.path.join(out_root, "X.npy"), Xw)
    np.save(os.path.join(out_root, "M.npy"), Mw)
    np.save(os.path.join(out_root, "e.npy"), ew)
    np.save(os.path.join(out_root, "S.npy"), Sw)
    np.save(os.path.join(out_root, "A_true.npy"), A)
    meta = {"d":int(d),"T":int(T),"regimes":int(regimes),"Tw":int(Tw),"Ts":int(Ts)}
    with open(os.path.join(out_root, "meta.json"), "w") as f:
        json.dump(meta, f)
    return out_root
