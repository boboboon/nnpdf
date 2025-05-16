"""T3 Toy Model Fit Script
Combines elements from the author’s GGI notebooks to:
  1) Load prepared BCDMS Fₚ–F_d data
  2) Define and train a PDF neural network
  3) Perform replica fits and cross-validation
  4) (Optional) Include a SMEFT Wilson coefficient parameter

Usage:
  python t3_toy_model.py

"""

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import Adam

# %%
# 1) Load prepared data
# ---------------------
# y: central values (F_p - F_d), shape (248,)
# Cy: covariance matrix,        shape (248,248)
# kin: kinematic points        shape (248,2)
# FK: FK convolution table,    shape (248,50)
# fk_grid: x-grid              shape (50,)
# NNPDF40: reference PDF vector shape (n_replicas*50,)

data_path = "data/prepared_data/"

y = np.load(data_path + "data.npy")  # (248,)
Cy = np.load(data_path + "Cy.npy")  # (248,248)
kin = np.load(data_path + "kin.npy")  # (248,2)
FK = np.load(data_path + "FK.npy")  # (248,50)
xgrid = np.load(data_path + "fk_grid.npy")  # (50,)
NNPDF40 = np.load(data_path + "NNPDF40.npy")  # (nreplicas*50,)

# slice out T3 reference from central replica
T3_ref = NNPDF40[6 * 50 : 7 * 50]

# %%
# 2) Kinematic coverage plot
# ---------------------------
plt.figure()
plt.scatter(kin[:, 0], kin[:, 1] ** 2, marker="*", label="BCDMS F$_p$-F$_d$")
plt.xscale("log")
plt.yscale("log")
plt.xlabel("x")
plt.ylabel("Q$^2$ [GeV$^2$]")
plt.legend()
plt.grid(True)
plt.title("Kinematic Coverage")
plt.show()

# %%
# 3) Invert covariance matrix
# ----------------------------
w, V = np.linalg.eigh(Cy)
invCy = V @ np.diag(1.0 / w) @ V.T  # (248,248)

# convert to torch tensors
y_torch = torch.tensor(y, dtype=torch.float32)
FK_torch = torch.tensor(FK, dtype=torch.float32)
x_torch = torch.tensor(xgrid, dtype=torch.float32).unsqueeze(1)
Cinv_torch = torch.tensor(invCy, dtype=torch.float32)


# %%
# 4) Network & Convolution definitions
# -------------------------------------
class PDF_NN(nn.Module):
    """Simple feed-forward net for PDF values"""

    def __init__(self):
        super().__init__()
        self.l1 = nn.Linear(1, 64)
        self.l2 = nn.Linear(64, 64)
        self.l3 = nn.Linear(64, 32)
        self.l4 = nn.Linear(32, 1)

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = F.relu(self.l3(x))
        return self.l4(x)  # (nx,1)


class ComputeConv(nn.Module):
    """Convolution with precomputed FK table"""

    def __init__(self, FK_np):
        super().__init__()
        self.FK = nn.Parameter(torch.tensor(FK_np, dtype=torch.float32), requires_grad=False)

    def forward(self, pdf_vals):
        # FK: (ndata,nx), pdf_vals: (nx,1)
        return (self.FK @ pdf_vals).squeeze(1)  # (ndata,)


class Observable(nn.Module):
    """Combines PDF_NN and ComputeConv"""

    def __init__(self, FK_np, include_smeft=False):
        super().__init__()
        self.pdf_net = PDF_NN()
        self.conv = ComputeConv(FK_np)
        # optional SMEFT Wilson coefficient (multiplicative)
        if include_smeft:
            self.c = nn.Parameter(torch.tensor(0.0))
        else:
            self.c = None

    def forward(self, x):
        pdf_vals = self.pdf_net(x)  # (nx,1)
        y_pred = self.conv(pdf_vals)  # (ndata,)
        if self.c is not None:
            # toy SMEFT effect: scale prediction
            y_pred = y_pred * (1.0 + self.c)
        return y_pred


# %%
# 5) χ² loss function
# -------------------
def chi2_loss(y_true, y_pred, invC):
    d = y_true - y_pred  # (ndata,)
    chi2 = d @ (invC @ d)  # scalar
    return chi2 / y_true.numel()


# %%
# 6) Single-model training
# ------------------------
model = Observable(FK, include_smeft=True)
opt = Adam(model.parameters(), lr=1e-3)
epochs = 500
patience = 10
best_val = float("inf")
wait = 0
hist = []

for epoch in range(1, epochs + 1):
    model.train()
    opt.zero_grad()
    y_pred = model(x_torch)
    loss = chi2_loss(y_torch, y_pred, Cinv_torch)
    loss.backward()
    opt.step()

    val = loss.item()
    hist.append(val)
    if val < best_val:
        best_val, wait = val, 0
        torch.save(model.state_dict(), "t3_best.pt")
    else:
        wait += 1
        if wait >= patience:
            print(f"Early stopping at epoch {epoch}")
            break
    if epoch % 50 == 0:
        print(f"Epoch {epoch}: χ²/N = {val:.4f}")

# training history plot
plt.figure()
plt.plot(hist)
plt.xlabel("Epoch")
plt.ylabel("χ²/N")
plt.title("Single-model Training")
plt.show()

# %%
# 7) Replica fits
# ----------------
from copy import deepcopy


def fit_replica(y, Cy, invC, FK_np, xgrid, epochs=500, noise=True):
    # draw replica
    y_rep = np.random.multivariate_normal(y, Cy) if noise else y.copy()
    y_rep_t = torch.tensor(y_rep, dtype=torch.float32)
    # new model instance
    mdl = Observable(FK_np, include_smeft=False)
    optim = Adam(mdl.parameters(), lr=1e-3)
    history = []
    for ep in range(epochs):
        optim.zero_grad()
        yp = mdl(torch.tensor(xgrid, dtype=torch.float32).unsqueeze(1))
        loss = chi2_loss(y_rep_t, yp, torch.tensor(invC, dtype=torch.float32))
        loss.backward()
        optim.step()
        history.append(loss.item())
    return mdl, history


n_replicas = 10
replica_models = []
replica_histories = []
for i in range(n_replicas):
    print(f"Fitting replica {i}")
    mdl_i, hist_i = fit_replica(y, Cy, invCy, FK, xgrid)
    replica_models.append(mdl_i)
    replica_histories.append(hist_i)

# plot replica training curves
plt.figure()
for h in replica_histories:
    plt.plot(h, alpha=0.5)
plt.xlabel("Epoch")
plt.ylabel("χ²/N")
plt.title("Replica Training")
plt.show()

# %%
# 8) Cross-validation (Solution Ex 5)
# ----------------------------------
n_data = y.shape[0]
idx = np.random.permutation(n_data)
half = n_data // 2
fitting_idx, test_idx = idx[:half], idx[half:]

y_fit = y[fitting_idx]
y_test = y[test_idx]
Cy_fit = Cy[np.ix_(fitting_idx, fitting_idx)]
Cy_test = Cy[np.ix_(test_idx, test_idx)]
invC_fit = np.linalg.inv(Cy_fit)
invC_test = np.linalg.inv(Cy_test)
FK_fit = FK[fitting_idx]
FK_test = FK[test_idx]

# fit replicas on fitting set, record best epoch via look-back
best_epochs = []
best_models = []
n_rep_cv = 20
for rep in range(n_rep_cv):
    print(f"CV replica {rep}")
    # train on fitting set
    mdl_cv = Observable(FK_fit, include_smeft=False)
    optim = Adam(mdl_cv.parameters(), lr=1e-3)
    train_losses, val_losses = [], []
    # split fitting set into train/val 75/25
    perm = np.random.permutation(len(fitting_idx))
    split = int(0.75 * len(fitting_idx))
    train_idx = perm[:split]
    val_idx = perm[split:]
    for epoch in range(epochs):
        # train step
        optim.zero_grad()
        yp_tr = mdl_cv(torch.tensor(xgrid, dtype=torch.float32).unsqueeze(1))
        loss_tr = chi2_loss(
            torch.tensor(y_fit, dtype=torch.float32),
            yp_tr,
            torch.tensor(invC_fit, dtype=torch.float32),
        )
        loss_tr.backward()
        optim.step()
        train_losses.append(loss_tr.item())
        # validation
        yp_val = mdl_cv(torch.tensor(xgrid, dtype=torch.float32).unsqueeze(1))
        loss_val = chi2_loss(
            torch.tensor(y_fit, dtype=torch.float32),
            yp_val,
            torch.tensor(invC_fit, dtype=torch.float32),
        )
        val_losses.append(loss_val.item())
    # select best epoch on val
    best_epoch = int(np.argmin(val_losses))
    best_epochs.append(best_epoch)
    # store model state
    mdl_best = deepcopy(mdl_cv)
    best_models.append(mdl_best)

# histogram of best epochs
plt.figure()
plt.hist(best_epochs, bins=10)
plt.xlabel("Best Epoch")
plt.ylabel("Count")
plt.title("Cross-Validation Best Epochs")
plt.show()

# %%
# 9) Evaluation on test set: bias and variance
# --------------------------------------------
y_th_reps = np.stack(
    [
        mdl.conv(mdl.pdf_net(torch.tensor(xgrid, dtype=torch.float32).unsqueeze(1)))
        .detach()
        .numpy()[test_idx]
        for mdl in best_models
    ],
)
y_th_avg = np.mean(y_th_reps, axis=0)
bias = (y_th_avg - y_test) @ invC_test @ (y_th_avg - y_test)
variance = np.mean(
    [(y_th_reps[i] - y_th_avg) @ invC_test @ (y_th_reps[i] - y_th_avg) for i in range(n_rep_cv)],
)
print(f"Bias/N: {bias / variance:.3f}")

# %%
