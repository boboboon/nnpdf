# %%
"""t3_BSM_Comparison."""

# %%
# --- Imports & Setup ---
from __future__ import annotations

from pathlib import Path

import lhapdf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F  # noqa: N812
from loguru import logger
from sklearn.model_selection import train_test_split
from torch import nn
from torch.optim import Adam
from validphys.api import API
from validphys.fkparser import load_fktable
from validphys.loader import Loader

# Device for PyTorch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Create directories for outputs
model_state_dir = Path("model_states")
model_state_dir.mkdir(parents=True, exist_ok=True)
results_dir = Path("results")
results_dir.mkdir(parents=True, exist_ok=True)
plt.rc("text", usetex=True)
plt.rc("font", family="serif")

# %%
# 1. DATA LOADING & PREPROCESSING—PART 1: FETCH RAW TABLES
# ------------------------------------------------------------------------------
logger.info("Loading BCDMS F2 data from validphys API...")

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

df_p = (
    lcd_p.commondata_table.reset_index()
    .rename(
        columns={
            "kin1": "x",
            "kin2": "q2",
            "kin3": "y",
            "data": "F2_p",
            "stat": "error",
            "entry": "entry_p",
        },
    )
    .assign(idx_p=lambda df: df.index)
)
df_d = (
    lcd_d.commondata_table.reset_index()
    .rename(
        columns={
            "kin1": "x",
            "kin2": "q2",
            "kin3": "y",
            "data": "F2_d",
            "stat": "error",
            "entry": "entry_d",
        },
    )
    .assign(idx_d=lambda df: df.index)
)


# Merge on (x, q2) to form F2_p - F2_d
mp = 0.938
mp2 = mp**2
merged_df = df_p.merge(df_d, on=["x", "q2"], suffixes=("_p", "_d")).assign(
    y_val=lambda df: df["F2_p"] - df["F2_d"],
    w2=lambda df: df["q2"] * (1 - df["x"]) / df["x"] + mp2,
)

# Extract q2_vals and y_real for later use
q2_vals = merged_df["q2"].to_numpy()
y_real = merged_df["y_val"].to_numpy()

# %%
# 2. DATA LOADING & PREPROCESSING—PART 2: BUILD FK TABLES & W
# ------------------------------------------------------------------------------
logger.info("Building FK tables and computing convolution matrix W for t3 channel...")

t3_index = 2  # flavor index in FK table
loader = Loader()
fk_p = load_fktable(loader.check_fktable(setname="BCDMSP", theoryID=208, cfac=()))
fk_d = load_fktable(loader.check_fktable(setname="BCDMSD", theoryID=208, cfac=()))

wp = fk_p.get_np_fktable()  # shape (n_data_fk, n_flav, n_grid)
wd = fk_d.get_np_fktable()
wp_t3 = wp[:, t3_index, :]
wd_t3 = wd[:, t3_index, :]

entry_p_rel = merged_df["entry_p"].to_numpy() - 1
entry_d_rel = merged_df["entry_d"].to_numpy() - 1
W = wp_t3[entry_p_rel] - wd_t3[entry_d_rel]  # shape (n_data, n_grid)

# Save xgrid for later normalization
xgrid = fk_p.xgrid.copy()  # shape (n_grid,)

# %%
# 3. DATA LOADING & PREPROCESSING—PART 3: COMPUTE C_YY & ITS INVERSE
# ------------------------------------------------------------------------------
logger.info("Building covariance matrix c_yy for y = F2_p - F2_d...")

params_cov = {
    "dataset_inputs": [inp_p["dataset_input"], inp_d["dataset_input"]],
    "use_cuts": "internal",
    "theoryid": 208,
}
cov_full = API.dataset_inputs_covmat_from_systematics(**params_cov)

# Suppose merged_df has columns idx_p and idx_d (these were created earlier in your preprocessing)
idx_p_merge = merged_df["idx_p"].to_numpy()  # length = N (number of matched points)
idx_d_merge = merged_df["idx_d"].to_numpy()  # length = N (same N)

# cov_full is (Np + Nd) x (Np + Nd), so:
n_p = len(df_p)
# Extract the proton-proton, deuteron-deuteron, and proton-deuteron sub-blocks:
c_pp = cov_full[:n_p, :n_p]  # shape = (Np, Np)
c_dd = cov_full[n_p:, n_p:]  # shape = (Nd, Nd)
c_pd = cov_full[:n_p, n_p:]  # shape = (Np, Nd)

# Now restrict each block to only those rows/cols that appear in merged_df:
c_pp_sub = c_pp[np.ix_(idx_p_merge, idx_p_merge)]  # (N, N)
c_dd_sub = c_dd[np.ix_(idx_d_merge, idx_d_merge)]  # (N, N)
c_pd_sub = c_pd[np.ix_(idx_p_merge, idx_d_merge)]  # (N, N)


c_yy = c_pp_sub + c_dd_sub - 2 * c_pd_sub

# Make sure it's exactly symmetric:
c_yy = 0.5 * (c_yy + c_yy.T)


# Add jitter until positive-definite
jitter = 1e-6 * np.mean(np.diag(c_yy))
for _ in range(10):
    try:
        np.linalg.cholesky(c_yy)
        break
    except np.linalg.LinAlgError:
        c_yy += np.eye(c_yy.shape[0]) * jitter
        jitter *= 10
else:
    msg = "Covariance matrix not positive-definite"
    raise RuntimeError(msg)

# %%
# 4. DATA LOADING & PREPROCESSING—PART 4: TRAPEZOIDAL WEIGHTS ON XGRID
# ------------------------------------------------------------------------------
logger.info("Computing trapezoidal weights on xgrid for normalization...")

x_torch = torch.tensor(xgrid, dtype=torch.float32).unsqueeze(1).to(device)
dx = (x_torch[1:] - x_torch[:-1]).squeeze()
dx_low = torch.cat([dx, dx[-1:]], dim=0).squeeze()
dx_high = torch.cat([dx[0:1], dx], dim=0).squeeze()
weights = 0.5 * (dx_low + dx_high)  # shape (n_grid,)

# %%
# 5. DATA LOADING & PREPROCESSING—PART 5: COMPUTE t3_REF_NORM FOR CLOSURE
# ------------------------------------------------------------------------------
logger.info("Computing reference t3 (t3_ref_norm) for closure test...")

pdfset = lhapdf.getPDFSet("NNPDF40_nnlo_as_01180")
pdf0 = pdfset.mkPDF(0)
Q0 = fk_p.Q0
t3_true = np.zeros_like(xgrid)

for i, x in enumerate(xgrid):
    u = pdf0.xfxQ(2, x, Q0)
    ub = pdf0.xfxQ(-2, x, Q0)
    d = pdf0.xfxQ(1, x, Q0)
    db = pdf0.xfxQ(-1, x, Q0)
    t3_true[i] = 2 * ((u + ub) - (d + db))

# 2) Convolution ⇒ noiseless pseudo-data:
y_theory = W.dot(t3_true)  # shape (N,)

# 3) Add experimental noise drawn from Cyy:
rng = np.random.default_rng(seed=451)  # you can set seed if you want reproducible “data”
noise = rng.multivariate_normal(mean=np.zeros(len(y_theory)), cov=c_yy)

y_pseudo = y_theory + noise

plt.figure()

# 1) Real data vs. Theory (open blue circles)
plt.scatter(
    y_theory,
    y_real,
    s=24,
    alpha=0.7,
    facecolors="none",
    edgecolors="C0",
    label=r"Real Data: $y_{data} = F_{2}^{p} - F_{2}^{d}$",
)

# 2) Pseudo-data vs. Theory (filled orange dots)
plt.scatter(
    y_theory,
    y_pseudo,
    s=18,
    alpha=0.6,
    color="C1",
    label=r"Pseudo-Data: $y_{pseudo} = W\,t_{3}^{NNPDF} + \eta$",
)

# 3) Diagonal y = x (gray dashed line)
mn = min(y_theory.min(), y_real.min(), y_pseudo.min())
mx = max(y_theory.max(), y_real.max(), y_pseudo.max())
plt.plot(
    [mn, mx],
    [mn, mx],
    linestyle="--",
    color="gray",
    alpha=0.5,
    label=r"$y_{theory} = y_{observed}$",
)

# 4) Labels and Title (all math-text in "$...$")
plt.xlabel(
    r"$y_{theory} = [\,W \cdot x\,t_{3}(x)\,]_{NNPDF40}$",
    fontsize=14,
)
plt.ylabel(r"$y_{observed}$", fontsize=14)

plt.title(
    r"Comparison of Real vs. Pseudo-Data for $F_{2}^{p} - F_{2}^{d}$",
)

plt.legend(loc="lower right", frameon=True, edgecolor="k")
plt.grid(alpha=0.2)
plt.tight_layout()
plt.show()


# %%
# 7. NEURAL NETWORK MODEL DEFINITION (for real-data training only)
# ------------------------------------------------------------------------------


class T3Net(nn.Module):
    """Neural network for non-singlet PDF t₃(x) with preprocessing x^alpha (1-x)^beta."""

    def __init__(
        self,
        n_hidden: int,
        n_layers: int = 3,
        init_alpha: float = 1.0,
        init_beta: float = 3.0,
        dropout: float = 0.2,
    ) -> None:
        """Create T3 Net."""
        super().__init__()
        # Log-parametrization for alpha, beta
        self.logalpha = nn.Parameter(torch.log(torch.tensor(init_alpha)))
        self.logbeta = nn.Parameter(torch.log(torch.tensor(init_beta)))

        # Build MLP: [Linear → Tanh → BatchNorm] x (n_layers), ending in Linear
        layers: list[nn.Module] = [nn.Linear(1, n_hidden), nn.Tanh(), nn.BatchNorm1d(n_hidden)]
        for _ in range(n_layers - 1):
            layers += [
                nn.Linear(n_hidden, n_hidden),
                nn.BatchNorm1d(n_hidden),
                nn.Tanh(),
                nn.Dropout(dropout),
            ]
        layers.append(nn.Linear(n_hidden, 1))  # final raw output
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass → returns x · t₃_unc(x) ≥ 0.

        raw = self.net(x) is unconstrained; apply SoftPlus to ensure nonnegativity.
        Multiply by x^alpha (1-x)^beta to impose endpoints behavior.
        """
        raw = self.net(x).squeeze()  # shape (N_grid,)
        pos = F.softplus(raw)  # shape (N_grid,), enforces ≥ 0

        alpha = torch.exp(self.logalpha).clamp(min=1e-3)
        beta = torch.exp(self.logbeta).clamp(min=1e-3)
        x_ = x.squeeze().clamp(min=1e-6, max=1 - 1e-6)

        pre = x_.pow(alpha) * (1.0 - x_).pow(beta)  # shape (N_grid,)
        return pre * pos  # returns x · t₃_unc(x)


# %%
# 8. REAL-DATA TRAINING LOOP (no pseudo, no BSM, no extra functions)
# ------------------------------------------------------------------------------

# (Assumes W, c_yy, xgrid, weights, and y_real are already defined in the notebook.)

n_data = W.shape[0]
n_grid = xgrid.shape[0]
n_replicas = 2

# Hyperparameters

n_hidden = 30
n_layers = 3
dropout = 0.2
lr = 1e-3
weight_decay = 1e-4
patience = 1000
num_epochs = 5000


results_list = []

W_torch = torch.tensor(W, dtype=torch.float32, device=device)  # (n_data, n_grid)
x_torch = torch.tensor(xgrid, dtype=torch.float32).unsqueeze(1).to(device)  # (n_grid,1)
wgt_torch = torch.tensor(weights, dtype=torch.float32, device=device)  # (n_grid,)

for lambda_smooth in [0, 1e-1, 1e-2]:
    for replica in range(n_replicas):
        torch.manual_seed(replica * 1234)

        # 1) Split indices into 90% train / 10% validation
        idx_all = np.arange(n_data)
        train_idx, val_idx = train_test_split(idx_all, test_size=0.2, random_state=replica * 1000)

        # 2) Build cov-submatrix inverses for train and validation
        #    (Extract C_tr and C_val from c_yy and invert them)
        c_tr = c_yy[np.ix_(train_idx, train_idx)]
        c_val = c_yy[np.ix_(val_idx, val_idx)]
        Cinv_tr = torch.tensor(np.linalg.inv(c_tr), dtype=torch.float32, device=device)
        Cinv_val = torch.tensor(np.linalg.inv(c_val), dtype=torch.float32, device=device)

        replica_rng = np.random.default_rng(seed=replica * 451)

        noise = replica_rng.multivariate_normal(mean=np.zeros(len(y_theory)), cov=c_yy)
        y_pseudo_replica = y_theory + noise

        # We can pick one of these inputs
        y_real_replica = rng.multivariate_normal(y_real, c_yy)
        y_pseudo_replica = y_theory + noise
        y_real_real = y_real

        y_torch = torch.tensor(y_pseudo_replica, dtype=torch.float32, device=device)

        # 4) Initialize model and optimizer
        model = T3Net(
            n_hidden=n_hidden,
            n_layers=n_layers,
            init_alpha=1.0,
            init_beta=3.0,
            dropout=dropout,
        ).to(device)

        optimizer = Adam(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
        )

        best_val_loss = float("inf")
        wait_counter = 0

        # 5) Epoch loop
        for epoch in range(1, num_epochs + 1):
            model.train()
            optimizer.zero_grad()

            # 5a) Forward pass → compute raw = x·t₃_unc(x)
            f_raw = model(x_torch).squeeze()  # (n_grid,)
            t3_unnorm = f_raw / x_torch.squeeze()  # t₃_unc(x), (n_grid,)
            #! I think this is still just trapz, we should be certain though  # noqa: E501, EXE001, EXE003, EXE005
            integral = torch.dot(wgt_torch, t3_unnorm)  # ∫ t₃_unc(x) dx
            norm_factor = 1.0 / integral
            f_norm = norm_factor * f_raw  # x·t₃(x), normalized

            # 5b) Convolution → y_pred = W · [x·t₃(x)]
            y_pred = W_torch.matmul(f_norm)  # (n_data,)

            # 5c) χ² loss on train subset
            resid_tr = y_pred[train_idx] - y_torch[train_idx]  # (n_train,)
            loss_chi2 = resid_tr @ (Cinv_tr.matmul(resid_tr))  # scalar

            # 5d) Smoothness penalty: ∥d²t₃_unc/dx²∥² (second-derivative on grid)
            dx_vals = (x_torch[1:] - x_torch[:-1]).squeeze()  # (n_grid-1,)
            d2 = (t3_unnorm[:-2] - 2 * t3_unnorm[1:-1] + t3_unnorm[2:]) / (dx_vals[:-1] ** 2)
            loss_smooth = torch.sum(d2.pow(2))  # scalar

            # 5e) Total loss = χ² + lambda smooth x smoothness
            loss_total = loss_chi2 + lambda_smooth * loss_smooth
            loss_total.backward()
            optimizer.step()

            # 5f) Validation pass (compute χ² on val subset)
            model.eval()
            with torch.no_grad():
                f_raw_val = model(x_torch).squeeze()
                t3_unnorm_val = f_raw_val / xgrid
                integral_val = torch.dot(wgt_torch, t3_unnorm_val.float())
                norm_val = 1.0 / integral_val
                f_norm_val = norm_val * f_raw_val
                y_pred_val = W_torch[val_idx].matmul(f_norm_val)  # (n_val,)
                resid_val = y_pred_val - y_torch[val_idx]  # (n_val,)
                loss_val = resid_val @ (Cinv_val.matmul(resid_val))  # scalar

            # 5g) Early stopping & checkpointing in memory
            if loss_val.item() < best_val_loss:
                best_val_loss = loss_val.item()
                wait_counter = 0
                # Save best-state_dict in memory
                best_state_dict = {k: v.clone() for k, v in model.state_dict().items()}
            else:
                wait_counter += 1
                if wait_counter >= patience:
                    break

            if epoch % 200 == 0:
                n_val = val_idx.shape[0]
                logger.info(
                    f"Replica {replica} | Epoch {epoch:4d} | val χ²/n = {(loss_val / n_val).item():.4f}",  # noqa: E501
                )

        # 6) After training, reload best state and compute final metrics on val set
        model.load_state_dict(best_state_dict)
        model.eval()
        with torch.no_grad():
            f_raw_best = model(x_torch).squeeze()
            t3_unnorm_best = f_raw_best / xgrid
            integral_best = torch.dot(wgt_torch, t3_unnorm_best.float())
            norm_best = 1.0 / integral_best
            f_norm_best = norm_best * f_raw_best  # (n_grid,)
            y_pred_best = W_torch[val_idx].matmul(f_norm_best)  # (n_val,)
            resid_v = y_pred_best - y_torch[val_idx]
            chi2_val_final = float(resid_v @ (Cinv_val.matmul(resid_v)))
            chi2_per_pt = chi2_val_final / float(val_idx.shape[0])

        # 7) Extract learned alpha, beta
        alpha_val = float(torch.exp(model.logalpha).item())
        beta_val = float(torch.exp(model.logbeta).item())

        # 8) Store results for this replica
        results_list.append(
            {
                "replica": replica,
                "lambda": lambda_smooth,
                "chi2_val": chi2_val_final,
                "chi2_per_pt": chi2_per_pt,
                "alpha": alpha_val,
                "beta": beta_val,
                "f_norm_best": f_norm_best.cpu().numpy(),  # numpy array of length n_grid
            },
        )

# 9) Combine into a DataFrame and save

df_results = pd.DataFrame(results_list).set_index("replica")
df_results.to_csv("training_results.csv")


# %%
# 10. PLOT Mean ±1sigma envelope of x·t₃(x) across replicas
# ------------------------------------------------------------------------------

t3_unnorm = t3_true / xgrid
I_truth = np.trapz(t3_unnorm, xgrid)  # noqa: NPY201
t3_norm = t3_true * (1.0 / I_truth)

# Now build the 3x1 figure
lambdas = sorted(df_results["lambda"].unique())
fig, axes = plt.subplots(nrows=len(lambdas), ncols=1, figsize=(6, 4 * len(lambdas)), sharex=True)
if len(lambdas) == 1:
    axes = [axes]

for idx, lam in enumerate(lambdas):
    ax = axes[idx]
    subset = df_results[df_results["lambda"] == lam]
    all_f_norm = np.vstack(subset["f_norm_best"].values)  # (n_replicas, n_grid)
    mean_f = np.mean(all_f_norm, axis=0)
    std_f = np.std(all_f_norm, axis=0)
    avg_sigma = np.mean(std_f)

    # ±1sigma envelope (shaded)
    ax.fill_between(
        xgrid,
        mean_f - std_f,
        mean_f + std_f,
        color="C0",
        alpha=0.3,
        label=rf"±1$\sigma$ envelope ($\langle\sigma\rangle={avg_sigma:.3f}$)",
    )
    # Mean learned x·t3(x)
    ax.plot(xgrid, mean_f, color="C0", linewidth=2, label=r"Mean $x\,t_{3}(x)$")

    # ——> Here is the corrected “truth” overlay:
    ax.plot(xgrid, t3_norm, color="k", linestyle="--", linewidth=1.5, label=r"NNPDF4.0 (truth)")

    # Compute and annotate χ²/pt
    chi_vals = subset["chi2_per_pt"].to_numpy()
    mean_chi = np.mean(chi_vals)
    std_chi = np.std(chi_vals)
    ax.text(
        0.95,
        0.95,
        rf"$\chi^2/\mathrm{{pt}} = {mean_chi:.2f}\pm{std_chi:.2f}$",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=10,
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "alpha": 0.7},
    )

    ax.set_ylabel(r"$x\,t_{3}(x)$", fontsize=12)
    ax.set_title(rf"$\lambda = {lam}$", fontsize=14)
    ax.grid(alpha=0.2)
    ax.legend(fontsize=10)

axes[-1].set_xlabel(r"$x$", fontsize=12)
plt.tight_layout()
plt.show()
# %%
