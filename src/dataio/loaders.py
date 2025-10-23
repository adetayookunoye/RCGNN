
import numpy as np, os, torch

class SynthDataset(torch.utils.data.Dataset):
    def __init__(self, root, split="train", split_fracs=(0.6,0.2,0.2), seed=0):
        self.X = np.load(os.path.join(root, "X.npy"))
        self.M = np.load(os.path.join(root, "M.npy"))
        self.e = np.load(os.path.join(root, "e.npy"))
        self.S = np.load(os.path.join(root, "S.npy"))
        n = self.X.shape[0]
        rng = np.random.default_rng(seed)
        # split by regimes
        regimes = np.unique(self.e)
        rng.shuffle(regimes)
        ntr = int(len(regimes)*split_fracs[0])
        nva = int(len(regimes)*split_fracs[1])
        train_R = regimes[:ntr]
        val_R = regimes[ntr:ntr+nva]
        test_R = regimes[ntr+nva:]
        mask = {"train":np.isin(self.e, train_R),
                "val":np.isin(self.e, val_R),
                "test":np.isin(self.e, test_R)}[split]
        self.idx = np.where(mask)[0]

    def __len__(self): return len(self.idx)

    def __getitem__(self, i):
        ii = self.idx[i]
        X = torch.tensor(self.X[ii], dtype=torch.float32)
        M = torch.tensor(self.M[ii], dtype=torch.float32)
        e = torch.tensor(self.e[ii], dtype=torch.long)
        S = torch.tensor(self.S[ii], dtype=torch.float32)
        return {"X":X, "M":M, "e":e, "S":S}

def load_synth(root, split="train", seed=0):
    return SynthDataset(root, split=split, seed=seed)
