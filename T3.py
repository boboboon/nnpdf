# %%
"""T3 Script (Closure/Pseudo-data version with physics-inspired NN, smoothness prior, and valence).

Fits a neural-network PDF to pseudo-data generated from
W @ T3_ref_norm (LHAPDF truth normalized to ∫T₃=1) plus correlated Gaussian noise,
enforcing
  1) small-x Regge behavior x^alpha,
  2) large-x counting (1-x)^beta,
  3) positivity (SoftPlus output),
  4) exact valence-sum ∫₀¹ T₃(x) dx = 1,
  5) a curvature penalty for smoothness in unconstrained regions,
and applies replica-selection to accept only physically reasonable fits.
At the end, it performs additional diagnostics:
  • Histograms of alpha and beta across replicas
  • Distribution of chisquared/pt for all vs kept replicas
  • Histogram of the first moment ∫ x T₃(x) dx
  • Correlation matrix heatmap for T₃(x) across x-grid
  • A summary table of key statistics
"""

from pathlib import Path

import lhapdf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
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

# Create directories for images and model states
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

merged_df = df_p.merge(df_d, on=["x", "Q2"], suffixes=("_p", "_d")).assign(
    y_val=lambda df: df["F2_p"] - df["F2_d"],
    W2=lambda df: df["Q2"] * (1 - df["x"]) / df["x"] + mp2,
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

# Build y = p - d combination covariance
C_yy = C_pp[np.ix_(idx_p, idx_p)] + C_dd[np.ix_(idx_d, idx_d)] - 2 * C_pd[np.ix_(idx_p, idx_d)]
# 1) Symmetrize
C_yy = 0.5 * (C_yy + C_yy.T)
# 2) Add jitter until positive definite
jitter = 1e-6 * np.mean(np.diag(C_yy))
for _ in range(10):
    try:
        np.linalg.cholesky(C_yy)
        break
    except np.linalg.LinAlgError:
        C_yy += np.eye(C_yy.shape[0]) * jitter
        jitter *= 10
else:
    cov_var_err = "Covariance matrix is not positive-definite even after jitter."
    raise RuntimeError(cov_var_err)
# Invert
Cinv = np.linalg.inv(C_yy)

# %% ---- x-grid & Tensors for PyTorch ----
xgrid = fk_p.xgrid  # (N_grid,)
x_torch = torch.tensor(xgrid, dtype=torch.float32).unsqueeze(1)  # (N_grid,1)
W_torch = torch.tensor(W, dtype=torch.float32)  # (N_data, N_grid)
Cinv_torch = torch.tensor(Cinv, dtype=torch.float32)  # (N_data, N_data)

# %%
# x-bin coverage plot
plt.figure(figsize=(5, 4))
coverage = np.count_nonzero(W, axis=0)
plt.plot(xgrid, coverage, "o-")
plt.xscale("log")
plt.xlabel(r"$x$")
plt.ylabel("Number of data contributing")
plt.title("Data coverage per x-bin")
plt.tight_layout()
plt.savefig(images_dir / "coverage_plot.png", dpi=300)
plt.show()

# %% ---- Pseudo-data Generation ----
# Load LHAPDF truth and compute x*T3_ref(x)
pdfset = lhapdf.getPDFSet("NNPDF40_nnlo_as_01180")
pdf0 = pdfset.mkPDF(0)
Qref = fk_p.Q0

T3_ref = []
for x in xgrid:
    u, ub = pdf0.xfxQ(2, x, Qref), pdf0.xfxQ(-2, x, Qref)
    d, db = pdf0.xfxQ(1, x, Qref), pdf0.xfxQ(-1, x, Qref)
    T3_ref.append((u + ub) - (d + db))
T3_ref = np.array(T3_ref)  # shape: (N_grid,) = x*T3_true(x)

# Normalize LHAPDF truth so that ∫ T3_true(x) dx = 1
x_np = xgrid
T3_val_ref = T3_ref / x_np  # T3_true(x)
I_truth = np.trapz(T3_val_ref, x_np)  # ∫ T3_true(x) dx over grid  # noqa: NPY201
T3_ref_norm = T3_ref * (1.0 / I_truth)  # now ∫[T3_ref_norm/x] dx = 1

# Generate pseudo-data from W @ T3_ref_norm → ensures closure
y_pseudo_mean = W @ T3_ref_norm
rng = np.random.default_rng(42)
y_pseudo = rng.multivariate_normal(y_pseudo_mean, C_yy)

y_raw = merged_df["y_val"].to_numpy()
y_torch = torch.tensor(y_pseudo, dtype=torch.float32)

# %%
# View pseudo-data before fitting
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
sc = ax1.scatter(merged_df["x"], merged_df["Q2"], c=y_pseudo, cmap="viridis", s=25, alpha=0.8)
ax1.set(
    xscale="log",
    yscale="log",
    xlabel=r"$x$",
    ylabel=r"$Q^2\,\mathrm{[GeV^2]}$",
    title=r"Pseudo-data kinematics: $F_2^p - F_2^d$",
)
plt.colorbar(sc, ax=ax1, label=r"$y_\mathrm{pseudo}$")
ax2.hist(y_pseudo, bins=30, alpha=0.7)
ax2.set(xlabel=r"$y = F_2^p-F_2^d$", ylabel="Counts", title="Pseudo-data distribution")
fig.tight_layout()
fig.savefig(images_dir / "pseudo_data_kinematics_and_distribution.png", dpi=300)
plt.show()

# %% ---- FK Convolution: pseudo vs real data ----
colour_vals = merged_df["W2"].to_numpy()

fig, ax = plt.subplots(figsize=(6, 6))
# 1) pseudo-data
sc_pseudo = ax.scatter(
    y_pseudo_mean,
    y_pseudo,
    color="C0",
    marker="o",
    alpha=0.6,
    label="Pseudo-data",
)
# 2) real BCDMS data colored by W²
sc_real = ax.scatter(
    y_pseudo_mean,
    y_raw,
    c=colour_vals,
    cmap="plasma",
    marker="s",
    s=50,
    alpha=0.8,
)
cbar = fig.colorbar(sc_real, ax=ax, label=r"$W^2$ [GeV$^2$]")
# 3) diagonal guide
lims = [
    min(y_pseudo_mean.min(), y_pseudo.min(), y_raw.min()),
    max(y_pseudo_mean.max(), y_pseudo.max(), y_raw.max()),
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


# %% ---- Model Definition: T3Net with physics preprocessing ----
class T3Net(nn.Module):
    """Neural network for non-singlet PDF with x^alpha(1-x)^beta preprocessing."""

    def __init__(
        self,
        n_hidden: int,
        n_layers: int = 3,
        init_alpha: float = 1.0,
        init_beta: float = 3.0,
        dropout: float = 0.2,
    ) -> None:
        """n_hidden : int    Number of units per hidden layer.

        n_layers : int    Number of hidden layers.
        init_alpha : float  Initial small-x exponent.
        init_beta  : float  Initial large-x exponent.
        dropout     : float  Dropout probability between layers.
        """
        super().__init__()

        # Trainable exponents (log-parametrized to ensure positivity)
        self.logalpha = nn.Parameter(torch.log(torch.tensor(init_alpha)))
        self.logbeta = nn.Parameter(torch.log(torch.tensor(init_beta)))

        # Build a small MLP
        layers: list[nn.Module] = []
        layers.append(nn.Linear(1, n_hidden))
        layers.append(nn.Tanh())
        layers.append(nn.BatchNorm1d(n_hidden))

        for _ in range(n_layers - 1):
            layers.append(nn.Linear(n_hidden, n_hidden))
            layers.append(nn.BatchNorm1d(n_hidden))
            layers.append(nn.Tanh())
            layers.append(nn.Dropout(dropout))

        layers.append(nn.Linear(n_hidden, 1))  # output a single raw score
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Computes f_raw(x) = x^alpha (1-x)^beta * SoftPlus( NN(x) ), which is unnorm x*T3(x)."""
        raw = self.net(x)  # shape: (N_grid, 1)
        pos = torch_func.softplus(raw).squeeze()  # (N_grid, ); ensures ≥ 0

        alpha = torch.exp(self.logalpha).clamp(min=1e-3)
        beta = torch.exp(self.logbeta).clamp(min=1e-3)

        # Prevent x=0 or x=1 exactly by clamping
        x_ = x.squeeze().clamp(min=1e-6, max=1.0 - 1e-6)  # (N_grid, )

        pre = x_.pow(alpha) * (1.0 - x_).pow(beta)  # (N_grid, )

        return pre * pos  # shape: (N_grid,), this is x*T3_unc(x)


# %% ---- Precompute Δx weights for valence-sum ----
# We use trapezoidal weights on the sorted xgrid
dx = (x_torch[1:] - x_torch[:-1]).squeeze()  # (N_grid-1,)
dx_low = torch.cat([dx, dx[-1:]], dim=0).squeeze()  # extend last bin  (N_grid,)
dx_high = torch.cat([dx[0:1], dx], dim=0).squeeze()  # extend first bin  (N_grid,)
weights = 0.5 * (dx_low + dx_high)  # (N_grid,)

# %% ---- Replica Loop with Physics-Inspired NN, Curvature Penalty, and Selection ----
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
x_torch = x_torch.to(device)  # (N_grid,1)
W_torch = W_torch.to(device)  # (N_data, N_grid)
Cinv_torch = Cinv_torch.to(device)  # (N_data, N_data)

patience = 1000
num_epochs = 5000
n_replicas = 100  # increased to 30 for smoother ensemble
lambda_smooth = 1e-4  # curvature penalty coefficient

# Prepare containers for all and kept chi2/pt
all_chi2_perpt = []
kept_chi2_perpt = []

data = {
    "replica": [],
    "chi2": [],
    "chi2_perpt": [],
    "alpha": [],
    "beta": [],
    "train_history": [],
    "val_history": [],
    "fit": [],
}

for i in range(n_replicas):
    logger.info(f"Starting replica {i}")

    # 1) Generate new pseudo-data for this replica (closure using T3_ref_norm)
    rng = np.random.default_rng(42 + i)
    y_pseudo = rng.multivariate_normal(y_pseudo_mean, C_yy)
    y_torch_rep = torch.tensor(y_pseudo, dtype=torch.float32, device=device)

    # 2) Train/validation split (10% validation)
    idx = np.arange(W_torch.size(0))
    train_idx, val_idx = train_test_split(idx, test_size=0.1, random_state=42 + i)

    # 3) Sub-covariances and invert
    C_tr = C_yy[np.ix_(train_idx, train_idx)]
    C_val = C_yy[np.ix_(val_idx, val_idx)]
    Cinv_tr = torch.tensor(np.linalg.inv(C_tr), dtype=torch.float32, device=device)
    Cinv_val = torch.tensor(np.linalg.inv(C_val), dtype=torch.float32, device=device)

    # 4) Initialize model & optimizer (smaller NN + weight decay)
    model = T3Net(n_hidden=30, n_layers=3, dropout=0.2).to(device)
    opt = Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

    train_hist, val_hist = [], []
    best_val, wait = float("inf"), 0

    for epoch in range(1, num_epochs + 1):
        model.train()
        opt.zero_grad()

        # --- Forward pass: compute unnormalized f_raw(x)
        f_raw = model(x_torch).squeeze()  # (N_grid,)
        t3_unnorm = f_raw / x_torch.squeeze()  # T3_unc(x)
        I_raw = torch.dot(weights, t3_unnorm)  # discrete ∫ T3_unc dx
        A = 1.0 / I_raw  # enforce ∫ T3 dx = 1
        f_pred = A * f_raw  # x*T3_pred(x) (unnormalized)

        # --- chisquared on training subset
        resid_tr = (W_torch[train_idx] @ f_pred) - y_torch_rep[train_idx]
        loss_chi2 = resid_tr @ (Cinv_tr @ resid_tr)

        # --- CURVATURE PENALTY on T3_unc(x)
        dx_vals = (x_torch[1:] - x_torch[:-1]).squeeze()  # (N_grid-1,)
        t3 = t3_unnorm
        d2 = (t3[:-2] - 2 * t3[1:-1] + t3[2:]) / (dx_vals[:-1] ** 2)  # (N_grid-2,)
        loss_smooth = torch.sum(d2.pow(2))

        # --- Total loss: chisquared + λ_smooth * curvature
        loss_total = loss_chi2 + lambda_smooth * loss_smooth
        loss_total.backward()
        opt.step()
        train_hist.append(loss_total.item())

        # --- Validation pass (no curvature penalty)
        model.eval()
        with torch.no_grad():
            f_raw_val = model(x_torch).squeeze()
            t3_unnorm_val = f_raw_val / x_torch.squeeze()
            I_raw_val = torch.dot(weights, t3_unnorm_val)
            A_val = 1.0 / I_raw_val
            f_val = A_val * f_raw_val

            resid_val = (W_torch[val_idx] @ f_val) - y_torch_rep[val_idx]
            loss_val = resid_val @ (Cinv_val @ resid_val)

        val_hist.append(loss_val.item())

        if epoch % 200 == 0:
            logger.info(f"Replica:{i}, Epoch:{epoch}, Val chisquared={loss_val:.3f}")

        # --- Early stopping on validation chisquared
        if loss_val.item() < best_val:
            best_val, wait = loss_val.item(), 0
            torch.save(model.state_dict(), model_state_dir / f"t3_replica_{i}.pt")
        else:
            wait += 1
            if wait >= patience:
                logger.info(f"Replica {i} early-stopped at epoch {epoch}")
                break

    # --- Post-fit evaluation on validation subset
    model.load_state_dict(torch.load(model_state_dir / f"t3_replica_{i}.pt"))
    model.eval()
    with torch.no_grad():
        f_raw_best = model(x_torch).squeeze()
        t3_unnorm_best = f_raw_best / x_torch.squeeze()
        I_raw_best = torch.dot(weights, t3_unnorm_best)
        A_best = 1.0 / I_raw_best
        f_best = A_best * f_raw_best  # (N_grid,)

        resid_v = (W_torch[val_idx] @ f_best) - y_torch_rep[val_idx]
        chi2_val = float(resid_v @ (Cinv_val @ resid_v))

    N_val = len(val_idx)
    chi2_perpt = chi2_val / N_val

    # Store all chisquared/pt
    all_chi2_perpt.append(chi2_perpt)

    # Replica selection: keep only if chisquared/pt ∈ [0.5, 1.5]
    if 0.5 <= chi2_perpt <= 1.5:
        kept_chi2_perpt.append(chi2_perpt)

        alpha_val = float(torch.exp(model.logalpha).item())
        beta_val = float(torch.exp(model.logbeta).item())

        data["replica"].append(i)
        data["chi2"].append(chi2_val)
        data["chi2_perpt"].append(chi2_perpt)
        data["alpha"].append(alpha_val)
        data["beta"].append(beta_val)
        data["train_history"].append(train_hist)
        data["val_history"].append(val_hist)
        data["fit"].append(f_best.cpu().numpy())

        logger.success(
            f"Replica {i} kept: chisquared={chi2_val}, chisquared/pt={chi2_perpt}, alpha={alpha_val}, beta={beta_val}",  # noqa: E501
        )
    else:
        logger.warning(f"Replica {i} discarded: chisquared/pt={chi2_perpt:.2f} outside [0.5,1.5]")

# Build DataFrame of accepted replicas
results_df = pd.DataFrame(
    {
        "replica": data["replica"],
        "chi2": data["chi2"],
        "chi2_perpt": data["chi2_perpt"],
        "alpha": data["alpha"],
        "beta": data["beta"],
    },
)
results_df["train_history"] = pd.Series(data["train_history"], dtype="object")
results_df["val_history"] = pd.Series(data["val_history"], dtype="object")
results_df["fit"] = pd.Series(data["fit"], dtype="object")

# %% ---- Plot Replica Ensemble vs. Truth ----
fits = np.stack(results_df["fit"].values)  # (n_kept, N_grid)
mean_f = fits.mean(axis=0)
std_f = fits.std(axis=0)

fig, ax = plt.subplots(figsize=(6, 5))
ax.fill_between(xgrid, mean_f - std_f, mean_f + std_f, alpha=0.3, label="Envelope")
ax.plot(xgrid, mean_f, "-", label="Mean")
ax.plot(xgrid, T3_ref_norm, "--", label="Truth (normalized)")

ax.set_xscale("log")
ax.set_xlabel(r"$x$")
ax.set_ylabel(r"$x\,T_{3}(x)$")
ax.set_title("Ensemble of Accepted Replicas vs. Truth")
ax.legend(fontsize="small")
fig.tight_layout()
fig.savefig(images_dir / "ensemble_vs_truth.png", dpi=300)
plt.show()


# %% ---- Plot Mean Train vs. Validation Loss per Point (Split into Two Halves) ----

N_data = W.shape[0]
N_val = int(0.10 * N_data)  # 10% for validation
N_train = N_data - N_val  # 90% for training

# Gather all the train/val histories:
all_train = results_df["train_history"].tolist()
all_val = results_df["val_history"].tolist()
max_epochs = max(len(h) for h in all_train)

# Build arrays, padded with NaN:
train_arr = np.full((len(all_train), max_epochs), np.nan)
val_arr = np.full((len(all_val), max_epochs), np.nan)
for j, h in enumerate(all_train):
    train_arr[j, : len(h)] = h
for j, h in enumerate(all_val):
    val_arr[j, : len(h)] = h

# Compute the epoch-by-epoch mean of each (ignoring NaNs):
mean_train_raw = np.nanmean(train_arr, axis=0)  # raw χ² + curvature
mean_val_raw = np.nanmean(val_arr, axis=0)  # raw χ²

# Convert to “per point”:
mean_train_perpt = mean_train_raw / N_train
mean_val_perpt = mean_val_raw / N_val

# Epoch index array:
epochs = np.arange(1, len(mean_train_perpt) + 1)

# Split into two halves:
mid = len(epochs) // 2
epochs_first = epochs[:mid]
train_first = mean_train_perpt[:mid]
val_first = mean_val_perpt[:mid]

epochs_second = epochs[mid:]
train_second = mean_train_perpt[mid:]
val_second = mean_val_perpt[mid:]

# Plot first half vs second half side by side
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), sharey=False)

# First half (early epochs)
ax1.plot(epochs_first, train_first, label=r"Train ($\chi^2/N_{\rm train}$)")
ax1.plot(epochs_first, val_first, label=r"Val   ($\chi^2/N_{\rm val}$)", linestyle="--")
ax1.set_title(f"Epochs 1 to {mid}")
ax1.set_xlabel("Epoch")
ax1.set_ylabel(r"$\chi^2/\mathrm{point}$")
ax1.legend(fontsize="small")

# Second half (later epochs)
ax2.plot(epochs_second, train_second, label=r"Train ($\chi^2/N_{\rm train}$)")
ax2.plot(epochs_second, val_second, label=r"Val   ($\chi^2/N_{\rm val}$)", linestyle="--")
ax2.set_title(f"Epochs {mid + 1} to {len(epochs)}")
ax2.set_xlabel("Epoch")
ax2.legend(fontsize="small")

fig.suptitle(r"Mean Train vs. Validation Loss per Point (Split by Epoch Halves)", y=1.02)
fig.tight_layout(rect=[0, 0, 1, 0.95])  # leave space for the suptitle
fig.savefig(images_dir / "train_vs_val_loss_per_pt_split.png", dpi=300)
plt.show()


# %% ---- Additional Diagnostics and Plots ----

# 1. Histogram of alpha and beta distributions across kept replicas
alphas = np.array(results_df["alpha"])
betas = np.array(results_df["beta"])

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
ax1.hist(alphas, bins="auto", edgecolor="black", alpha=0.7)
ax1.set_xlabel(r"$\alpha$")
ax1.set_ylabel("Number of Replicas")
ax1.set_title(r"Distribution of Small-$x$ Exponent $\alpha$")

ax2.hist(betas, bins="auto", edgecolor="black", alpha=0.7)
ax2.set_xlabel(r"$\beta$")
ax2.set_ylabel("Number of Replicas")
ax2.set_title(r"Distribution of Large-$x$ Exponent $\beta$")

plt.tight_layout()
plt.savefig(images_dir / "alpha_beta_distributions.png", dpi=300)
plt.show()
# %%
# 2. 2D scatter of alpha vs beta
fig, ax = plt.subplots(figsize=(5, 5))
ax.scatter(alphas, betas, alpha=0.7)
ax.set_xlabel(r"$\alpha$")
ax.set_ylabel(r"$\beta$")
ax.set_title(r"Correlation: $\alpha$ vs.\ $\beta$")
plt.tight_layout()
plt.savefig(images_dir / "alpha_vs_beta_scatter.png", dpi=300)
plt.show()
# %%
# 3. Histogram of chisquared/pt: all vs kept replicas
fig, ax = plt.subplots(figsize=(6, 4))
ax.hist(all_chi2_perpt, bins=20, alpha=0.5, label="All Replicas", edgecolor="black")
ax.hist(kept_chi2_perpt, bins=20, alpha=0.7, label="Kept Replicas", edgecolor="black")
ax.axvline(0.5, color="red", linestyle="--")
ax.axvline(1.5, color="red", linestyle="--")
ax.set_xlabel(r"$\chi^2/\mathrm{pt}$")
ax.set_ylabel("Number of Replicas")
ax.set_title(r"Distribution of $\chi^2/\mathrm{pt}$ Across Replicas")
ax.legend(fontsize="small")
plt.tight_layout()
plt.savefig(images_dir / "chi2_perpt_distribution.png", dpi=300)
plt.show()
# %%
# 4. First moment ∫ x T₃(x) dx distribution
first_moments = []
for fit in results_df["fit"]:
    # fit = x*T3_pred(x) on grid
    T3_pred = fit / x_np  # T3_pred(x) on grid
    m1 = np.trapz(x_np * T3_pred, x_np)  # ∫ x T3(x) dx  # noqa: NPY201
    first_moments.append(m1)

fig, ax = plt.subplots(figsize=(5, 4))
ax.hist(first_moments, bins="auto", alpha=0.7, edgecolor="black")
ax.set_xlabel(r"$\int_{0}^{1} x\,T_{3}(x)\,dx$")
ax.set_ylabel("Number of Replicas")
ax.set_title(r"Distribution of First Moment: $\langle x\rangle_{T_{3}}$")
plt.tight_layout()
plt.savefig(images_dir / "first_moment_distribution.png", dpi=300)
plt.show()
# %%
# 5. Correlation matrix for T₃(x) across x-grid
# Build matrix of shape (n_kept, N_grid) for T3 values
T3_reps = fits / x_np[np.newaxis, :]  # shape: (n_kept, N_grid)
cov_mat = np.cov(T3_reps.T)  # (N_grid, N_grid)
stds = np.sqrt(np.diag(cov_mat))
corr = cov_mat / np.outer(stds, stds)  # (N_grid, N_grid)

fig, ax = plt.subplots(figsize=(6, 5))
im = ax.imshow(
    corr,
    origin="lower",
    extent=[np.log10(xgrid[0]), np.log10(xgrid[-1]), np.log10(xgrid[0]), np.log10(xgrid[-1])],
    cmap="RdBu_r",
    vmin=-1,
    vmax=1,
)
ax.set_xlabel(r"$\log_{10}x$")
ax.set_ylabel(r"$\log_{10}x$")
ax.set_title(r"Correlation Matrix: $T_{3}(x_i)\,$ vs.\ $\,T_{3}(x_j)$")
cbar = fig.colorbar(im, ax=ax, label="Correlation $\rho_{ij}$")
plt.tight_layout()
plt.savefig(images_dir / "T3_correlation_matrix.png", dpi=300)
plt.show()
# %%
# 6. Summary Table of Key Statistics
n_total = n_replicas
n_kept = len(results_df)

alpha_vals = alphas
beta_vals = betas
chi2pt_vals = np.array(data["chi2_perpt"])

summary = {
    "Total replicas": [n_total],
    "Kept replicas": [n_kept],
    "mean alpha ± sigma": [f"{alpha_vals.mean():.3f} ± {alpha_vals.std():.3f}"],
    "mean beta ± sigma": [f"{beta_vals.mean():.3f} ± {beta_vals.std():.3f}"],
    "mean chisquared/pt ± sigma": [f"{chi2pt_vals.mean():.3f} ± {chi2pt_vals.std():.3f}"],
    "mean ⟨x⟩ ± sigma": [f"{np.mean(first_moments):.3f} ± {np.std(first_moments):.3f}"],
}
summary_df = pd.DataFrame(summary)
logger.info("\n=== Summary of T₃ Closure Fit ===")
logger.info(summary_df.to_string(index=False))
# %%
