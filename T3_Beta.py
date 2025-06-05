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
from matplotlib.lines import Line2D
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
    y_val=lambda df: (df["F2_p"] - df["F2_d"]),
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
# 5. DATA LOADING & PREPROCESSING—PART 5: COMPUTE t3_REF_NORM FOR CLOSURE
# ------------------------------------------------------------------------------
logger.info("Computing reference t3 (t3_ref_norm) for closure test...")

pdfset = lhapdf.getPDFSet("NNPDF40_nnlo_as_01180")
pdf0 = pdfset.mkPDF(0)
Q0 = fk_p.Q0
t3_true = np.zeros_like(xgrid)

for i, x in enumerate(xgrid):
    u = pdf0.xfxQ(2, x, Q0)  # x·u(x)
    ub = pdf0.xfxQ(-2, x, Q0)  # x·ū(x)
    d = pdf0.xfxQ(1, x, Q0)  # x·d(x)
    db = pdf0.xfxQ(-1, x, Q0)  # x·d̄(x)
    t3_true[i] = (u - ub) - (d - db)

# 2) Convolution ⇒ noiseless pseudo-data:
y_theory = W @ t3_true  # shape (N,)

# 3) Add experimental noise drawn from Cyy:
rng = np.random.default_rng(seed=451)  # you can set seed if you want reproducible “data”
noise = rng.multivariate_normal(mean=np.zeros(len(y_theory)), cov=c_yy)

y_pseudo = y_theory + noise

t3_ref_int = np.trapz(t3_true / xgrid, xgrid)  # noqa: NPY201


# %%
# ? PRELIM DATA PLOTS
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

plt.legend(loc="upper right", frameon=True, edgecolor="k")
plt.grid(alpha=0.2)
plt.tight_layout()
plt.show()

# %%
# ? Heatmap
# 1) (Exactly as before) add y_theory into merged_df
merged_df["y_theory"] = y_theory

# 2) Pivot on (q2, x), taking the mean of y_val (data) and y_theory
pivot_real = (
    merged_df.pivot_table(index="q2", columns="x", values="y_val", aggfunc="mean")
    .sort_index(axis=0)
    .sort_index(axis=1)
)
pivot_theory = (
    merged_df.pivot_table(index="q2", columns="x", values="y_theory", aggfunc="mean")
    .sort_index(axis=0)
    .sort_index(axis=1)
)

# 3) Compute the difference: ⟨y_data⟩ - ⟨y_theory⟩
pivot_diff = pivot_real - pivot_theory

# 4) Extract the (sorted) x and Q² grids
x_vals = pivot_real.columns.to_numpy()  # (N_x,)
q2_vals = pivot_real.index.to_numpy()  # (N_q2,)
X_grid, Y_grid = np.meshgrid(x_vals, q2_vals)

# 5) Plot a single heatmap of (⟨y_data⟩ - ⟨y_theory⟩)
fig, ax = plt.subplots(figsize=(7, 6))

pcm = ax.pcolormesh(
    X_grid,
    Y_grid,
    pivot_diff.values,
    shading="auto",
    cmap="RdBu_r",  # diverging colormap is often useful for “difference”
    vmin=-np.max(np.abs(pivot_diff.values)),  # center zero at white
    vmax=np.max(np.abs(pivot_diff.values)),
)

cbar = fig.colorbar(pcm, ax=ax, label=r"$\langle\,y_{\rm data} - y_{\rm theory}\rangle$")
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_title(r"Mean Difference: $\langle\,y_{\rm data} - y_{\rm theory}\rangle$")
ax.set_xlabel(r"$x$")
ax.set_ylabel(r"$Q^2\,[\mathrm{GeV}^2]$")

plt.tight_layout()
plt.show()

# %%
# ? Theory Comparison
# 1) Compute sigma_i = sqrt(diagonal(C_yy))_i  divided by y_real_i
sigma = np.abs(np.sqrt(np.diag(c_yy)) / y_real)

# 2) Make an index array to place points on the x-axis
x_idx = np.arange(len(y_theory))  # 0, 1, 2, … N-1
ref = np.ones_like(y_theory)  # reference = 1 for “data/theory = 1”

# 3) Plot
plt.figure(figsize=(20, 5))
plt.errorbar(
    x_idx,
    ref,
    sigma,
    fmt="none",
    ecolor="gray",
    alpha=0.5,
    label=r"Data uncertainty $( \frac{\sigma_i}{y_i})$",
)
plt.scatter(
    x_idx,
    y_theory / y_real,
    marker="*",
    c="red",
    label="Theory / Data",
)

plt.ylim([0.1, 2.5])
plt.xlabel("Data point index (i)")
plt.ylabel(r"$\frac{y_{theory}}{y_{data}}$")
plt.title(r"Comparison of $y_{theory}$ vs.\ $y_{data}$ (with relative errors)")
plt.legend(loc="upper right")
plt.grid(alpha=0.3)
plt.show()

# %%
# ? Kinematic Plot
fig, (ax_p, ax_d) = plt.subplots(1, 2, figsize=(12, 5), sharex=True, sharey=True)

# Proton subplot
ax_p.scatter(
    df_p["x"],
    df_p["q2"],
    marker="o",
    c="C0",
    label=r"$F_2^p$",
    alpha=0.7,
)
ax_p.set_xscale("log")
ax_p.set_yscale("log")

ax_p.set_xlabel(r"$x$")
ax_p.set_ylabel(r"$Q^2\ \mathrm{[GeV^2]}$")
ax_p.set_title("BCDMS $F_2^p$")
ax_p.grid(which="both", alpha=0.3)

# Deuteron subplot
ax_d.scatter(
    df_d["x"],
    df_d["q2"],
    marker="s",
    c="C1",
    label=r"$F_2^d$",
    alpha=0.7,
)
ax_d.set_xscale("log")
ax_d.set_yscale("log")

ax_d.set_xlabel(r"$x$")
# Only include ylabel on the left subplot to avoid redundancy
ax_d.set_title("BCDMS $F_2^d$")
ax_d.grid(which="both", alpha=0.3)

plt.suptitle("Kinematic Coverage of BCDMS $F_2^p$ and $F_2^d$", y=1.02)
plt.tight_layout()
plt.show()

# %%
# 7. NEURAL NETWORK MODEL DEFINITION
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


class T3NetWithC(nn.Module):
    """Neural network for x·t₃(x) plus a single BSM parameter C."""

    def __init__(
        self,
        n_hidden: int,
        n_layers: int = 3,
        init_alpha: float = 1.0,
        init_beta: float = 3.0,
        dropout: float = 0.2,
    ) -> None:
        """Init our T3Net with additional BSM C Param."""
        super().__init__()
        # 1) instantiate the original T3Net
        self.base = T3Net(
            n_hidden=n_hidden,
            n_layers=n_layers,
            init_alpha=init_alpha,
            init_beta=init_beta,
            dropout=dropout,
        )
        # 2) expose logalpha/logbeta so that `model.logalpha` still works
        self.logalpha = self.base.logalpha
        self.logbeta = self.base.logbeta

        # 3) add a single learnable scalar C (initialized at 0)
        self.C = nn.Parameter(torch.tensor(0.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns f_raw = x · t₃_unc(x) (shape = n_grid,).

        The BSM correction factor (1 + C·K) is applied downstream in the training loop.
        """
        # simply delegate to the base network
        return self.base(x).squeeze()  # shape = (n_grid,)


# %%
# ─── 1) DEFINE BASE ANSATZ FUNCTIONS (NORMALIZED TO UNIT AMPLITUDE) ───
q2_vals = merged_df["q2"].to_numpy()  # (n_data,)
x_vals_data = merged_df["x"].to_numpy()  # (n_data,)
Q2_min = q2_vals.min()

# Raw (unnormalized) shapes
K1_raw = (q2_vals - Q2_min) ** 2
K2_raw = x_vals_data * (1.0 - x_vals_data) * (q2_vals - Q2_min)

# Normalize so max(|K_raw|)=1
K1_unit = K1_raw / np.max(np.abs(K1_raw))
K2_unit = K2_raw / np.max(np.abs(K2_raw))

# Convert to torch tensors
K_dict = {
    "ansatz1": torch.tensor(K1_unit, dtype=torch.float32, device=device),
    "ansatz2": torch.tensor(K2_unit, dtype=torch.float32, device=device),
    "noansatz": torch.zeros(len(q2_vals), dtype=torch.float32, device=device),
}

# ─── 2) DEFINE GRID OF “TRUE” C VALUES FOR SENSITIVITY SCAN ───
C_trues = [0.001, 0.1, 1]

# ─── 3) BUILD CONFIG DICTIONARY INCLUDING ORIGINAL FITS + SENSITIVITY SCANS ───
config = {
    # Original fits (no BSM)
    "fit_real_real": {
        "name": "Real-Data Fit",
        "input_key": "real_real",
        "n_hidden": 30,
        "n_layers": 3,
        "dropout": 0.2,
        "lr": 1e-3,
        "weight_decay": 1e-4,
        "patience": 500,
        "num_epochs": 5000,
        "n_replicas": 100,
        "lambda_sr": 0.0,
        "bsm": False,
        "ansatz": None,
        "C_true": 0.0,
    },
    "fit_pseudo_replica": {
        "name": "Pseudo-Replica Fit",
        "input_key": "pseudo_replica",
        "n_hidden": 30,
        "n_layers": 3,
        "dropout": 0.2,
        "lr": 1e-3,
        "weight_decay": 1e-4,
        "patience": 500,
        "num_epochs": 5000,
        "n_replicas": 100,
        "lambda_sr": 10000.0,
        "bsm": False,
        "ansatz": None,
        "C_true": 0.0,
    },
    "sens_noansatz_C0": {
        "name": "Pseudo-Replica BSM (No Ansatz), $C_{true}=0e0$",
        "input_key": "pseudo_replica",
        "n_hidden": 30,
        "n_layers": 3,
        "dropout": 0.2,
        "lr": 1e-3,
        "weight_decay": 1e-4,
        "patience": 500,
        "num_epochs": 5000,
        "n_replicas": 100,
        "lambda_sr": 10000.0,
        "bsm": True,
        "ansatz": "noansatz",
        "C_true": 0.0,
    },
}

for ansatz_name in ["ansatz1", "ansatz2"]:
    for C_true in C_trues:
        cfg_key = f"sens_{ansatz_name}_C{C_true:.0e}"
        display = {
            "ansatz1": "Sensitivity Scan 1 (Q²² shape)",
            "ansatz2": "Sensitivity Scan 2 (x(1-x)Q² shape)",
        }[ansatz_name]
        cfg_name = f"{display}, $C_{{true}}$={C_true:.0e}"
        config[cfg_key] = {
            "name": cfg_name,
            "input_key": "pseudo_replica",  # always pseudo-data for sensitivity scans
            "n_hidden": 30,
            "n_layers": 3,
            "dropout": 0.2,
            "lr": 1e-3,
            "weight_decay": 1e-4,
            "patience": 500,
            "num_epochs": 5000,
            "n_replicas": 100,
            "lambda_sr": 10000.0,
            "bsm": True,
            "ansatz": ansatz_name,
            "C_true": C_true,
        }


# ─── 4) PREPARE FIXED TENSORS FOR TRAINING ───
n_data = W.shape[0]
n_grid = xgrid.shape[0]
W_torch = torch.tensor(W, dtype=torch.float32, device=device)
x_torch = torch.tensor(xgrid, dtype=torch.float32).unsqueeze(1).to(device)
# %%
# Collect all results here
all_results = []

# ─── 5) OUTER LOOP OVER CONFIG ENTRIES ───
for cfg_key, cfg in config.items():
    input_key = cfg["input_key"]
    n_hidden = cfg["n_hidden"]
    n_layers = cfg["n_layers"]
    dropout = cfg["dropout"]
    lr = cfg["lr"]
    weight_decay = cfg["weight_decay"]
    patience = cfg["patience"]
    num_epochs = cfg["num_epochs"]
    n_replicas = cfg["n_replicas"]
    lambda_sr = cfg["lambda_sr"]
    is_bsm = cfg["bsm"]
    ansatz_name = cfg["ansatz"]
    C_true = cfg["C_true"]
    display_name = cfg["name"]

    for replica in range(n_replicas):
        # ─── 5.a) Split train/validation indices ───
        torch.manual_seed(replica * 1234)
        idx_all = np.arange(n_data)
        train_idx, val_idx = train_test_split(idx_all, test_size=0.2, random_state=replica * 1000)

        # ─── 5.b) Prepare y-input (with or without BSM injection) ───
        rng = np.random.default_rng(seed=replica * 451)
        y_real_replica = rng.multivariate_normal(y_real, c_yy)
        y_pseudo_replica = rng.multivariate_normal(y_theory, c_yy)

        if is_bsm:
            K_torch = K_dict[ansatz_name]
            y_theory_bsm = (W @ t3_true) * (1.0 + C_true * K_torch.cpu().numpy())
            y_select = rng.multivariate_normal(y_theory_bsm, c_yy)
        else:
            y_select = {"real_real": y_real.copy(), "pseudo_replica": y_pseudo_replica}[input_key]

        y_torch = torch.tensor(y_select, dtype=torch.float32, device=device)

        # ─── 5.c) Build covariance inverses for train & val ───
        c_tr = c_yy[np.ix_(train_idx, train_idx)]
        c_val = c_yy[np.ix_(val_idx, val_idx)]
        Cinv_tr = torch.tensor(np.linalg.inv(c_tr), dtype=torch.float32, device=device)
        Cinv_val = torch.tensor(np.linalg.inv(c_val), dtype=torch.float32, device=device)

        # ─── 5.d) Initialize model & optimizer ───
        if is_bsm:
            model = T3NetWithC(
                n_hidden=n_hidden,
                n_layers=n_layers,
                init_alpha=1.0,
                init_beta=3.0,
                dropout=dropout,
            ).to(device)
        else:
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
        best_state = {}

        # ─── 5.e) TRAINING LOOP ───
        for epoch in range(1, num_epochs + 1):
            model.train()
            optimizer.zero_grad()

            f_raw = model(x_torch).squeeze()  # (n_grid,)
            y_pred_sm = W_torch.matmul(f_raw)  # (n_data,)

            if is_bsm:
                K_t = K_dict[ansatz_name]
                y_pred = y_pred_sm * (1.0 + model.C * K_t)
            else:
                y_pred = y_pred_sm

            resid_tr = y_pred[train_idx] - y_torch[train_idx]
            loss_chi2 = resid_tr @ (Cinv_tr.matmul(resid_tr))

            # Sum-rule penalty
            loss_sumrule = torch.tensor(0.0, device=device)
            if lambda_sr > 0.0:
                t3_unc = f_raw / x_torch.squeeze()
                I_mid = torch.trapz(t3_unc, x_torch.squeeze())
                loss_sumrule = lambda_sr * (I_mid - float(t3_ref_int)) ** 2

            loss_total = loss_chi2 + loss_sumrule
            loss_total.backward()
            optimizer.step()

            # ─── Validation χ² (no sum-rule penalty) ───
            model.eval()
            with torch.no_grad():
                f_raw_val = model(x_torch).squeeze()
                y_val_sm = W_torch[val_idx].matmul(f_raw_val)

                y_val_pred = y_val_sm * (1.0 + model.C * K_t[val_idx]) if is_bsm else y_val_sm

                resid_val = y_val_pred - y_torch[val_idx]
                loss_val = resid_val @ (Cinv_val.matmul(resid_val))
                val_chi2_pt = (loss_val / float(len(val_idx))).item()

            # ─── Early stopping ───
            if loss_val.item() < best_val_loss:
                best_val_loss = loss_val.item()
                wait_counter = 0
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
            else:
                wait_counter += 1
                if wait_counter >= patience:
                    break

            # ─── Logging every 200 epochs ───
            if epoch % 200 == 0:
                if is_bsm:
                    logger.info(
                        f"{cfg_key} | Replica {replica} | Epoch {epoch:4d} | "
                        f"val χ²/pt = {val_chi2_pt:.4f} | C = {model.C.item():.3e}",
                    )
                else:
                    logger.info(
                        f"{cfg_key} | Replica {replica} | Epoch {epoch:4d} | "
                        f"val χ²/pt = {val_chi2_pt:.4f}",
                    )

        # ─── 5.f) Reload best state, compute final metrics on validation ───
        model.load_state_dict(best_state)
        model.eval()
        with torch.no_grad():
            f_raw_best = model(x_torch).squeeze()
            y_val_sm_best = W_torch[val_idx].matmul(f_raw_best)

            if is_bsm:
                y_val_pred_best = y_val_sm_best * (1.0 + model.C * K_t[val_idx])
            else:
                y_val_pred_best = y_val_sm_best

            resid_v = y_val_pred_best - y_torch[val_idx]
            chi2_val_final = float(resid_v @ (Cinv_val.matmul(resid_v)))
            chi2_pt = chi2_val_final / float(len(val_idx))

        alpha_val = float(torch.exp(model.logalpha).item())
        beta_val = float(torch.exp(model.logbeta).item())
        C_fit = float(model.C.item()) if is_bsm else float("nan")

        all_results.append(
            {
                "config_key": cfg_key,
                "config_name": display_name,
                "replica": replica,
                "alpha": alpha_val,
                "beta": beta_val,
                "C_true": C_true,
                "C_fit": C_fit,
                "chi2_pt": chi2_pt,
                "f_raw_best": f_raw_best,
            },
        )

# ─── 6) COMBINE INTO DATAFRAME AND SAVE ───
df_results = pd.DataFrame(all_results)
df_results.to_pickle("training_results.pkl")

# %%
# ─── 7) LOAD FOR PLOTTING ───
df_plot = (
    pd.read_pickle("training_results.pkl")
    .reset_index()
    .loc[
        lambda df: ~df["config_key"].isin(
            ["sens_noansatz_C1e-03", "sens_noansatz_C1e-01", "sens_noansatz_C1e+00"],
        )
    ]
).loc[lambda df: (df["chi2_pt"] < 1.1) & (df["chi2_pt"] > 0.9)]

# ---------------------------------------------------------------------
# 8a) PLOTTING: 1x2 COMPARISON — Real Data Fit vs. Pseudo-Data Fit
# ---------------------------------------------------------------------
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 5), sharex=True, sharey=True)
comparison_keys = ["fit_real_real", "fit_pseudo_replica"]
comparison_map = {
    "fit_real_real": axes[0],
    "fit_pseudo_replica": axes[1],
}

for cfg_key, ax in comparison_map.items():
    display_name = config[cfg_key]["name"]
    subset = df_plot[df_plot["config_key"] == cfg_key]

    # Stack all f_raw_best arrays (shape = (n_replicas, n_grid))
    all_f_raw = np.vstack(subset["f_raw_best"].values)
    mean_f = np.mean(all_f_raw, axis=0)
    std_f = np.std(all_f_raw, axis=0)

    # Compute average sigma for annotation
    avg_sigma = np.mean(std_f)

    # ±1sigma band
    ax.fill_between(
        xgrid,
        mean_f - std_f,
        mean_f + std_f,
        color="C0",
        alpha=0.3,
        label=rf"$\pm\sigma$ (⟨$\sigma$⟩ = {avg_sigma:.3f})",
    )

    # Mean x·t₃(x)
    ax.plot(
        xgrid,
        mean_f,
        color="C0",
        linewidth=2,
        label=r"Mean $x\,t_{3}(x)$",
    )

    # Overlay NNPDF40 “truth”
    ax.plot(
        xgrid,
        t3_true,
        color="k",
        linestyle="--",
        linewidth=1.5,
        label=r"NNPDF40 (truth)",
    )

    # Annotate ⟨χ²/pt⟩ ± std(χ²/pt)
    chi_vals = subset["chi2_pt"].astype(float).to_numpy()
    mean_chi = np.mean(chi_vals)
    std_chi = np.std(chi_vals)
    ax.text(
        0.95,
        0.95,
        rf"$\chi^2/\mathrm{{pt}} = {mean_chi:.2f}\,\pm\,{std_chi:.2f}$",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=10,
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "alpha": 0.7},
    )

    ax.set_title(display_name, fontsize=14)
    ax.set_xlabel(r"$x$", fontsize=12)
    ax.set_ylabel(r"$x\,t_{3}(x)$", fontsize=12)
    ax.grid(alpha=0.2)
    ax.legend(fontsize=10)

plt.tight_layout()
plt.show()
# %%
# ---------------------------------------------------------------------
# 8b) PLOTTING: 2x2 COMPARISON — Pseudo, No-Ansatz (C=0), Ansatz1 (smallest non-zero C),
# Ansatz2 (smallest non-zero C)
# ---------------------------------------------------------------------
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 10), sharex=True, sharey=True)

pseudo_map = {
    "fit_pseudo_replica": axes[0, 0],  # Pure pseudo-data
    "sens_noansatz_C0": axes[0, 1],  # BSM closure with no ansatz, C=0
    # (Since we do not have ansatz1/ansatz2 at C=0 in the new config, use the smallest non-zero C
    # case: C=1e-03)
    "sens_ansatz1_C1e-03": axes[1, 0],  # Sensitivity Scan 1, C=1e-03
    "sens_ansatz2_C1e-03": axes[1, 1],  # Sensitivity Scan 2, C=1e-03
}

for cfg_key, ax in pseudo_map.items():
    display_name = config[cfg_key]["name"]
    subset = df_plot[df_plot["config_key"] == cfg_key]

    # Stack all f_raw_best arrays
    all_f_raw = np.vstack(subset["f_raw_best"].values)
    mean_f = np.mean(all_f_raw, axis=0)
    std_f = np.std(all_f_raw, axis=0)

    # Compute average sigma for annotation
    avg_sigma = np.mean(std_f)

    # ±1sigma band
    ax.fill_between(
        xgrid,
        mean_f - std_f,
        mean_f + std_f,
        color="C0",
        alpha=0.3,
        label=rf"$\pm\sigma$ (⟨$\sigma$⟩ = {avg_sigma:.3f})",
    )

    # Mean x·t₃(x)
    ax.plot(
        xgrid,
        mean_f,
        color="C0",
        linewidth=2,
        label=r"Mean $x\,t_{3}(x)$",
    )

    # Overlay NNPDF40 “truth”
    ax.plot(
        xgrid,
        t3_true,
        color="k",
        linestyle="--",
        linewidth=1.5,
        label=r"NNPDF40 (truth)",
    )

    # Annotate ⟨χ²/pt⟩ ± std(χ²/pt)
    chi_vals = subset["chi2_pt"].astype(float).to_numpy()
    mean_chi = np.mean(chi_vals)
    std_chi = np.std(chi_vals)
    ax.text(
        0.95,
        0.95,
        rf"$\chi^2/\mathrm{{pt}} = {mean_chi:.2f}\,\pm\,{std_chi:.2f}$",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=10,
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "alpha": 0.7},
    )

    ax.set_title(display_name, fontsize=14)
    ax.set_xlabel(r"$x$", fontsize=12)
    ax.set_ylabel(r"$x\,t_{3}(x)$", fontsize=12)
    ax.grid(alpha=0.2)
    ax.legend(fontsize=10)

plt.tight_layout()
plt.show()
# %%
# ---------------------------------------------------------------------
# 9) PLOTTING: alpha vs. beta SCATTER SPLIT INTO TWO SUBPLOTS
#               LEFT = real, pseudo, no-ansatz (all C_true = 0)
#               RIGHT = ansatz1 & ansatz2 (marker = ansatz, color = C_true, discrete legend)
#               Both subplots share x- and y-axes for direct comparison
# ---------------------------------------------------------------------


fig, (ax_left, ax_right) = plt.subplots(ncols=2, figsize=(14, 6), sharex=True, sharey=True)

# -----------------------
# LEFT SUBPLOT (C_true = 0)
# -----------------------
left_configs = {
    "fit_real_real": "Real-Data Fit",
    "fit_pseudo_replica": "Pseudo-Replica Fit",
    "sens_noansatz": "BSM Closure (No Ansatz)",
}
# assign one distinct color per config_key prefix (using C3, C4, C5 so they differ from right
# subplot)
color_map_left = {
    "fit_real_real": "C3",
    "fit_pseudo_replica": "C4",
    "sens_noansatz": "C5",
}

for prefix, label in left_configs.items():
    subset = df_plot[df_plot["config_key"].str.startswith(prefix)]
    if subset.empty:
        continue

    alphas = subset["alpha"].astype(float).to_numpy()
    betas = subset["beta"].astype(float).to_numpy()

    ax_left.scatter(
        alphas,
        betas,
        marker="o",
        color=color_map_left[prefix],
        edgecolor="k",
        alpha=0.8,
        label=label,
        linewidth=0.5,
        s=50,
    )

ax_left.set_xlabel(r"$\alpha$", fontsize=12)
ax_left.set_ylabel(r"$\beta$", fontsize=12)
ax_left.set_title("No BSM (all $C_{true}=0$)", fontsize=14)
ax_left.grid(alpha=0.2)
ax_left.legend(title="Configuration", loc="upper left")

# ------------------------
# RIGHT SUBPLOT (BSM Scan)
# ------------------------
C_trues = [0.001, 0.1, 1.0]
# Assign one discrete color per C_true (using C0, C1, C2)
color_map = {
    0.001: "C0",
    0.1: "C1",
    1.0: "C2",
}
# Marker by ansatz
marker_map_ansatz = {
    "ansatz1": "s",  # square
    "ansatz2": "o",  # circle
}

# Plot each (ansatz, C_true) combination, but only label C_true once (when ansatz1)
for ansatz_name, mkr in marker_map_ansatz.items():
    for C_true in C_trues:
        cfg_key = f"sens_{ansatz_name}_C{C_true:.0e}"
        subset = df_plot[df_plot["config_key"] == cfg_key]
        if subset.empty:
            continue

        alphas = subset["alpha"].astype(float).to_numpy()
        betas = subset["beta"].astype(float).to_numpy()

        # Only give a label for C_true on the ansatz1 pass so that each C_true appears once in the
        # legend
        label_ct = (f"$C_{{true}}$={C_true:.0e}") if ansatz_name == "ansatz1" else None

        ax_right.scatter(
            alphas,
            betas,
            marker=mkr,
            color=color_map[C_true],
            edgecolor="k",
            alpha=0.8,
            label=label_ct,
            linewidth=0.5,
            s=50,
        )

ax_right.set_xlabel(r"$\alpha$", fontsize=12)
ax_right.set_title("BSM Sensitivity Scans", fontsize=14)
ax_right.grid(alpha=0.2)

# Create a separate legend for the marker-shape ⇒ ansatz mapping
ansatz_handles = [
    Line2D([0], [0], marker="s", color="gray", linestyle="", label="Ansatz 1", markeredgecolor="k"),
    Line2D([0], [0], marker="o", color="gray", linestyle="", label="Ansatz 2", markeredgecolor="k"),
]
legend1 = ax_right.legend(handles=ansatz_handles, title="Ansatz", loc="upper left")
ax_right.add_artist(legend1)

# Create a second legend for C_true ⇒ color mapping
ax_right.legend(title="$C_{true}$", loc="lower left")

plt.tight_layout()
plt.show()


# %%
# --- 10) PLOTTING: Raw C_fit Histograms with Mean, Std, and C_true Lines ---
fig, axes = plt.subplots(
    nrows=1,
    ncols=len(C_trues),
    figsize=(4 * len(C_trues), 4),
    sharex=True,
    sharey=True,
)

# Define colors for each ansatz
ansatz_colors = {
    "ansatz1": "C0",
    "ansatz2": "C1",
}

for col_idx, C_true_val in enumerate(C_trues):
    ax = axes[col_idx]
    stats_texts = []
    for ansatz_name in ["ansatz1", "ansatz2"]:
        cfg_key = f"sens_{ansatz_name}_C{C_true_val:.0e}"
        subset = df_plot[df_plot["config_key"] == cfg_key]
        if subset.empty:
            continue

        C_vals = subset["C_fit"].to_numpy()
        mean_i = C_vals.mean()
        std_i = C_vals.std()

        # Plot histogram of raw C_fit
        ax.hist(
            C_vals,
            bins=30,
            histtype="stepfilled",
            alpha=0.5,
            density=True,
            color=ansatz_colors[ansatz_name],
        )

        # Draw vertical dashed line at the mean
        ax.axvline(
            mean_i,
            color=ansatz_colors[ansatz_name],
            linestyle="--",
            linewidth=1.0,
        )

        # Prepare annotation text for this ansatz using LaTeX for mu and sigma
        stats_texts.append(
            f"{ansatz_name.capitalize()}: $\\mu={mean_i:.3f}$, $\\sigma={std_i:.3f}$",
        )

    # Draw vertical solid line at the injected C_true
    ax.axvline(
        C_true_val,
        color="k",
        linestyle="-",
        linewidth=1.0,
    )

    ax.set_title(f"$C_{{true}} = {C_true_val:.0e}$", fontsize=12)
    ax.set_xlabel(r"$C_{\rm fit}$", fontsize=12)
    if col_idx == 0:
        ax.set_ylabel("Density", fontsize=12)
    ax.grid(alpha=0.2)

    # Place mean and std annotation in upper right corner
    ax.text(
        0.95,
        0.95,
        "\n".join(stats_texts),
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=9,
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "alpha": 0.7},
    )


legend_handles = [
    Line2D([0], [0], color=ansatz_colors["ansatz1"], lw=4, label="Ansatz 1"),
    Line2D([0], [0], color=ansatz_colors["ansatz2"], lw=4, label="Ansatz 2"),
]
fig.legend(
    handles=legend_handles,
    title="Ansatz",
    loc="upper center",
    ncol=2,
    bbox_to_anchor=(0.5, 1.05),
)

plt.suptitle("Raw $C_{\\rm fit}$ Distributions with Mean, Std, and True Value", y=1.10, fontsize=14)
plt.tight_layout()
plt.show()

# %%
