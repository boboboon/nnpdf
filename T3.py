# %%
"""T3 Script (Closure/Pseudo-data version, clean and documented).

Fits a neural-network PDF to pseudo-data generated from
W @ T3_ref (NNPDF4.0 central) plus correlated Gaussian noise.

"""

from pathlib import Path

import lhapdf
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as torch_func
from loguru import logger
from matplotlib.lines import Line2D
from sklearn.model_selection import train_test_split
from torch import nn
from torch.optim import Adam
from validphys.api import API
from validphys.fkparser import load_fktable
from validphys.loader import Loader

# --- Set matplotlib to use LaTeX for all text
plt.rc("text", usetex=True)
plt.rc("font", family="serif")

images_dir = Path("images")
images_dir.mkdir(exist_ok=True)

model_state_dir = Path("model_states")
model_state_dir.mkdir(exist_ok=True)

# %% ---- Data Preparation ----
# Load BCDMS F2_p, F2_d and form difference y = F2_p - F2_d
inp_p = {
    "dataset_input": {"dataset": "BCDMS_NC_NOTFIXED_P_EM-F2", "variant": "legacy"},
    "use_cuts": "internal",
    "theoryid": 208,
}
inp_d = {
    "dataset_input": {"dataset": "BCDMS_NC_NOTFIXED_D_EM-F2", "variant": "legacy"},
    "use_cuts": "internal",
    "theoryid": 208,
}

lcd_p = API.loaded_commondata_with_cuts(**inp_p)
lcd_d = API.loaded_commondata_with_cuts(**inp_d)

df_p = lcd_p.commondata_table.rename(
    columns={"kin1": "x", "kin2": "Q2", "kin3": "y", "data": "F2_p", "stat": "error"},
)
df_d = lcd_d.commondata_table.rename(
    columns={"kin1": "x", "kin2": "Q2", "kin3": "y", "data": "F2_d", "stat": "error"},
)

df_p["idx_p"] = np.arange(len(df_p))
df_d["idx_d"] = np.arange(len(df_d))


mp = 0.938
mp2 = mp**2

merged_df = (
    df_p.merge(df_d, on=["x", "Q2"], suffixes=("_p", "_d")).assign(
        y_val=lambda df: df["F2_p"] - df["F2_d"],
        W2=lambda df: df["Q2"] * (1 - df["x"]) / df["x"] + mp2,
    )  # difference
)
# %% ---- Raw BCDMS F2 (x vs Q² heatmap) ----
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), sharex=True, sharey=True)

# Proton
sc1 = ax1.scatter(df_p["x"], df_p["Q2"], c=df_p["F2_p"], cmap="viridis", s=25, alpha=0.8)
ax1.set_xscale("log")
ax1.set_yscale("log")
ax1.set_xlabel(r"$x$")
ax1.set_ylabel(r"$Q^2\,[\mathrm{GeV}^2]$")
ax1.set_title(r"BCDMS $F_2^p$")
plt.colorbar(sc1, ax=ax1, label=r"$F_2^p$")

# Deuteron
sc2 = ax2.scatter(df_d["x"], df_d["Q2"], c=df_d["F2_d"], cmap="plasma", s=25, alpha=0.8)
ax2.set_xscale("log")
ax2.set_yscale("log")
ax2.set_xlabel(r"$x$")
ax2.set_title(r"BCDMS $F_2^d$")
plt.colorbar(sc2, ax=ax2, label=r"$F_2^d$")

fig.suptitle("Raw BCDMS DIS Data: Proton and Deuteron", y=1.02)
fig.tight_layout()
fig.savefig(images_dir / "raw_F2p_F2d_heatmap.png", dpi=300)
plt.show()


# %% ---- FK Table Construction ----
loader = Loader()
fk_p = load_fktable(loader.check_fktable(setname="BCDMSP", theoryID=208, cfac=()))
fk_d = load_fktable(loader.check_fktable(setname="BCDMSD", theoryID=208, cfac=()))

wp = fk_p.get_np_fktable()
wd = fk_d.get_np_fktable()
flavor_index = 2  # T3 = u^+ - d^+

wp_t3 = wp[:, flavor_index, :]
wd_t3 = wd[:, flavor_index, :]
idx_p = merged_df["idx_p"].to_numpy()
idx_d = merged_df["idx_d"].to_numpy()
W = wp_t3[idx_p] - wd_t3[idx_d]  # convolution matrix (N_data, N_grid)

# %% ---- Covariance Matrix ----
params = {
    "dataset_inputs": [inp_p["dataset_input"], inp_d["dataset_input"]],
    "use_cuts": "internal",
    "theoryid": 208,
}
cov_full = API.dataset_inputs_covmat_from_systematics(**params)
n_p, n_d = len(df_p), len(df_d)
C_pp = cov_full[:n_p, :n_p]
C_dd = cov_full[n_p:, n_p:]
C_pd = cov_full[:n_p, n_p:]
C_yy = C_pp[np.ix_(idx_p, idx_p)] + C_dd[np.ix_(idx_d, idx_d)] - 2 * C_pd[np.ix_(idx_p, idx_d)]

eps = 1e-6 * np.mean(np.diag(C_yy))
C_yy += np.eye(C_yy.shape[0]) * eps

# now invert safely
Cinv = np.linalg.inv(C_yy)

# %% ---- x-grid & Tensors ----
xgrid = fk_p.xgrid  # (N_grid,)
x_torch = torch.tensor(xgrid, dtype=torch.float32).unsqueeze(1)
W_torch = torch.tensor(W, dtype=torch.float32)
Cinv_torch = torch.tensor(Cinv, dtype=torch.float32)


# %%
# x-bin coverage
plt.figure(figsize=(5, 4))
coverage = np.count_nonzero(W, axis=0)
plt.semilogx(xgrid, coverage, "o-")
plt.xlabel(r"$x$")
plt.ylabel("Number of data contributing")
plt.title("Data coverage per x-bin")
plt.tight_layout()
plt.show()

# %% ---- Pseudo-data Generation ----
pdfset = lhapdf.getPDFSet("NNPDF40_nnlo_as_01180")
pdf0 = pdfset.mkPDF(0)
Qref = fk_p.Q0
T3_ref = []
for x in xgrid:
    u, ub = pdf0.xfxQ(2, x, Qref), pdf0.xfxQ(-2, x, Qref)
    d, db = pdf0.xfxQ(1, x, Qref), pdf0.xfxQ(-1, x, Qref)
    T3_ref.append((u + ub) - (d + db))
T3_ref = np.array(T3_ref)  # true x*T3

# mean and fluctuation
base_seed = 42
rng = np.random.default_rng(base_seed)
y_pseudo_mean = W @ T3_ref
y_pseudo = rng.multivariate_normal(y_pseudo_mean, C_yy)

y_raw = merged_df["y_val"].to_numpy()

y_torch = torch.tensor(y_pseudo, dtype=torch.float32)

# %%
# Immediately show pseudo-data before fitting
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
# kinematic coverage
sc = ax1.scatter(merged_df["x"], merged_df["Q2"], c=y_pseudo, cmap="viridis", s=25, alpha=0.8)
ax1.set(
    xscale="log",
    yscale="log",
    xlabel=r"$x$",
    ylabel=r"$Q^2\,\mathrm{[GeV^2]}$",
    title=r"Pseudo-data kinematics: $F_2^p - F_2^d$",
)
plt.colorbar(sc, ax=ax1, label=r"$y_\mathrm{pseudo}$")
# distribution of y
ax2.hist(y_pseudo, bins=30, alpha=0.7)
ax2.set(xlabel=r"$y = F_2^p-F_2^d$", ylabel="Counts", title="Pseudo-data distribution")
fig.tight_layout()
fig.savefig(images_dir / "pseudo_data_kinematics_and_distribution.png", dpi=300)
plt.show()


# %% ---- FK convolution with x-colouring + distinct legend markers ----

y_ref = W @ T3_ref

colour_vals = merged_df["W2"].to_numpy()

fig, ax = plt.subplots(figsize=(6, 6))

# 1) pseudo-data
sc_pseudo = ax.scatter(y_ref, y_pseudo, color="C0", marker="o", alpha=0.6, label="Pseudo-data")

# 2) real BCDMS data coloured by W2
sc_real = ax.scatter(y_ref, y_raw, c=colour_vals, cmap="plasma", marker="s", s=50, alpha=0.8)
cbar = fig.colorbar(sc_real, ax=ax, label=r"$W^2$ [GeV$^2$]")

# 3) diagonal guide
lims = [
    min(y_ref.min(), y_pseudo.min(), y_raw.min()),
    max(y_ref.max(), y_pseudo.max(), y_raw.max()),
]
ax.plot(lims, lims, "k--", label="Ideal")

# 4) manual legend entries

proxy_real = Line2D([], [], marker="s", color="gray", linestyle="None", label="Real BCDMS")
proxy_ideal = Line2D([], [], linestyle="--", color="k", label="Ideal")
ax.legend(handles=[sc_pseudo, proxy_real, proxy_ideal])


ax.set_xlabel(r"$y_{\rm ref}=W\cdot xT_3$")
ax.set_ylabel(r"$y$")
ax.set_title("FK convolution: pseudo vs real data (coloured by $W^2$)")
fig.tight_layout()
fig.savefig(images_dir / "fk_convolution_pseudo_vs_real.png", dpi=300)
plt.show()


# %% ---- Model Definition ----
class T3Net(nn.Module):
    """Neural network for non-singlet PDF, outputs x*T3(x) with extra flexibility."""

    def __init__(
        self,
        n_hidden: int,
        n_layers: int = 3,
        alpha: float = 1.0,
        beta: float = 3.0,
        dropout: float = 0.1,
    ) -> None:
        """Parameters.

        ----------
        n_hidden : int
            Number of units per hidden layer.
        n_layers : int
            Total number of hidden layers.
        alpha : float
            Initial small-x exponent.
        beta : float
            Initial large-x exponent.
        dropout : float
            Dropout probability between layers.
        """
        super().__init__()

        layers: list[nn.Module] = []
        # input layer
        layers.append(nn.Linear(1, n_hidden))
        layers.append(nn.Tanh())
        layers.append(nn.BatchNorm1d(n_hidden))

        # hidden blocks
        for _ in range(n_layers - 1):
            layers.append(nn.Linear(n_hidden, n_hidden))
            layers.append(nn.Tanh())
            layers.append(nn.BatchNorm1d(n_hidden))
            layers.append(nn.Dropout(dropout))

        # output block
        layers.append(nn.Linear(n_hidden, 1))

        self.net = nn.Sequential(*layers)

        # overall normalization
        self.A = nn.Parameter(torch.tensor(1.0))

        # make exponents trainable
        self.alpha = nn.Parameter(torch.tensor(alpha))
        self.beta = nn.Parameter(torch.tensor(beta))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward Pass.

        computes A * x^alpha (1+x)^beta * softplus( NN(x) ).
        """
        raw = self.net(x)
        pos = torch_func.softplus(raw)
        pre = x.pow(self.alpha) * (1.0 - x).pow(self.beta)
        return self.A * pre * pos


# %% ---- Replica Loop with Post-Fit Validation & Discarding ----

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

patience, wait = 250, 0
num_epochs = 2000
n_replicas = 100

lambda_sum, lambda_pos = 100.0, 10.0


fits = []  # accepted fits
loss_per_replica = []  # validation χ²_phys for accepted replicas
discarded = []  # indices of discarded replicas
train_histories = []  # train-loss histories
val_histories = []  # val-loss histories

for i in range(n_replicas):
    # 1) Generate pseudo-data for this replica
    rng_rep = np.random.default_rng(base_seed + i)
    y_pseudo = rng_rep.multivariate_normal(y_pseudo_mean, C_yy)
    y_torch_rep = torch.tensor(y_pseudo, dtype=torch.float32).to(device)

    # 2) Train/validation split
    idx = np.arange(W_torch.size(0))
    train_idx, val_idx = train_test_split(idx, test_size=0.2, random_state=base_seed + i)

    # 3) Build & invert sub-covariances
    C_tr = C_yy[np.ix_(train_idx, train_idx)]
    C_val = C_yy[np.ix_(val_idx, val_idx)]
    eps_tr = 1e-6 * np.mean(np.diag(C_tr))
    eps_val = 1e-6 * np.mean(np.diag(C_val))
    C_tr += np.eye(len(train_idx)) * eps_tr
    C_val += np.eye(len(val_idx)) * eps_val
    Cinv_tr = torch.tensor(np.linalg.inv(C_tr), dtype=torch.float32).to(device)
    Cinv_val = torch.tensor(np.linalg.inv(C_val), dtype=torch.float32).to(device)

    # thresholds
    chi2_tol = 1.5 * len(val_idx)
    sum_tol = 0.5
    neg_tol = 1e-5

    # 4) Initialize model & optimizer
    model_rep = T3Net(n_hidden=100, n_layers=15, alpha=0.5, beta=5.0, dropout=0.2).to(device)
    opt_rep = Adam(model_rep.parameters(), lr=1e-3, weight_decay=1e-4)

    # 5) Train/Val loop with early stopping on validation loss
    train_loss, val_loss = [], []
    best_val, wait = float("inf"), 0

    for epoch in range(1, num_epochs + 1):
        # a) Train step
        model_rep.train()
        opt_rep.zero_grad()
        f_pred = model_rep(x_torch).squeeze()
        resid_tr = (W_torch[train_idx] @ f_pred) - y_torch_rep[train_idx]
        loss_tr = resid_tr @ (Cinv_tr @ resid_tr)
        # physics penalties
        x = x_torch.squeeze()
        t3 = f_pred / x
        loss_tr += lambda_sum * (torch.trapz(t3, x) - 1.0) ** 2
        loss_tr += lambda_pos * torch.sum(torch.relu(-t3) ** 2)
        loss_tr.backward()
        opt_rep.step()
        train_loss.append(loss_tr.item())

        # b) Validation step
        model_rep.eval()
        with torch.no_grad():
            resid_val = (W_torch[val_idx] @ f_pred) - y_torch_rep[val_idx]
            loss_val = resid_val @ (Cinv_val @ resid_val)
            loss_val += lambda_sum * (torch.trapz(t3, x) - 1.0) ** 2
            loss_val += lambda_pos * torch.sum(torch.relu(-t3) ** 2)
        val_loss.append(loss_val.item())

        # c) Early stop on validation
        if loss_val.item() < best_val:
            best_val, wait = loss_val.item(), 0
            torch.save(model_rep.state_dict(), model_state_dir / f"t3_replica{i}.pt")
        else:
            wait += 1
            if wait >= patience:
                logger.info(f"Replica {i}: early stop at epoch {epoch}")
                break

    train_histories.append(train_loss)
    val_histories.append(val_loss)

    # 6) Post-fit validation
    model_rep.load_state_dict(torch.load(model_state_dir / f"t3_replica{i}.pt"))
    model_rep.eval()
    with torch.no_grad():
        f_best = model_rep(x_torch).squeeze()
        resid_val = (W_torch[val_idx] @ f_best) - y_torch_rep[val_idx]
        chi2_val = float(resid_val @ (Cinv_val @ resid_val))
        T3_vals = (f_best / x).cpu()
        sum_dev = abs(torch.trapz(T3_vals, x.cpu()).item() - 1.0)
        neg_max = float((-T3_vals.clamp_max(0.0)).max().item())

    if chi2_val <= chi2_tol and sum_dev <= sum_tol and neg_max <= neg_tol:
        fits.append(f_best.cpu().numpy())
        loss_per_replica.append(chi2_val)
        logger.success(
            f"Replica {i} kept: χ²={chi2_val:.1f}, sum_dev={sum_dev:.2e}, neg_max={neg_max:.2e}",
        )
    else:
        discarded.append(i)
        logger.warning(
            f"Replica {i} discard: χ²={chi2_val:.1f}, sum_dev={sum_dev:.2e}, neg_max={neg_max:.2e}",
        )

# Convert accepted fits to array
fits = np.array(fits)
logger.info(f"Kept {fits.shape[0]} / {n_replicas} replicas; discarded {len(discarded)}.")

# %% ---- Plot Accepted Replica Ensemble vs. Truth ----
mean_fit = fits.mean(axis=0)
std_fit = fits.std(axis=0)

plt.figure(figsize=(6, 4))
plt.fill_between(
    xgrid,
    mean_fit - std_fit,
    mean_fit + std_fit,
    color="C1",
    alpha=0.3,
    label=r"Accepted ensemble ±1$\sigma$",
)
plt.plot(xgrid, mean_fit, "-", label="Accepted mean")
plt.plot(xgrid, T3_ref, "--", label="Truth (NNPDF4.0)")
plt.xlabel(r"$x$")
plt.ylabel(r"$x\,T_3(x)$")
plt.title("Accepted Replica Ensemble vs. Truth")
plt.legend()
plt.tight_layout()
plt.savefig(images_dir / "accepted_replica_vs_truth.png", dpi=300)
plt.show()

# %% ---- Plot Train/Val Loss (Last Accepted Replica) ----
plt.figure(figsize=(6, 4))
plt.plot(train_histories[-1], label="Train loss")
plt.plot(val_histories[-1], label="Val   loss")
plt.xlabel("Epoch")
plt.ylabel(r"$\chi^2_{\rm phys}$")
plt.title("Train vs. Validation Loss (Last Accepted Replica)")
plt.legend()
plt.tight_layout()
plt.savefig(images_dir / "train_val_loss_last_accepted.png", dpi=300)
plt.show()


# %% ---- PDF Correlation Matrix Heatmap ----
cov_pdf = np.cov(fits, rowvar=False)
corr_pdf = cov_pdf / np.sqrt(np.outer(np.diag(cov_pdf), np.diag(cov_pdf)))
plt.figure(figsize=(6, 5))
plt.imshow(corr_pdf, origin="lower", aspect="auto", vmin=-1, vmax=1)
plt.colorbar(label="Correlation")
plt.xlabel("x-bin index")
plt.ylabel("x-bin index")
plt.title("PDF Correlation Matrix")
plt.tight_layout()
plt.show()
# %% ---- Principal Components of PDF Uncertainty ----
eigvals, eigvecs = np.linalg.eigh(cov_pdf)
idx = np.argsort(eigvals)[::-1]
eigvals, eigvecs = eigvals[idx], eigvecs[:, idx]

# plot first 3 eigen-modes
plt.figure(figsize=(6, 4))
for j in range(3):
    plt.plot(xgrid, eigvecs[:, j], label=f"PC{j + 1} ({eigvals[j]:.2e})")
plt.xlabel(r"$x$")
plt.ylabel("Eigenvector")
plt.title("Leading Principal Components of PDF Covariance")
plt.legend()
plt.tight_layout()
plt.show()


# %%
