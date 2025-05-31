# %%
"""T3_BSM_Comparison.ipynb."""

# %%
# --- Imports & Setup ---
from pathlib import Path

import lhapdf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as torch_func
from loguru import logger
from sklearn.model_selection import train_test_split
from torch import nn
from torch.optim import Adam
from validphys.api import API
from validphys.fkparser import load_fktable
from validphys.loader import Loader

# Configure matplotlib for LaTeX-style text
plt.rc("text", usetex=True)
plt.rc("font", family="serif")

# Device for PyTorch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Create directories for outputs
images_dir = Path("images")
images_dir.mkdir(exist_ok=True)
model_state_dir = Path("model_states")
model_state_dir.mkdir(exist_ok=True)

# %%
# 1. Data Loading & Preprocessing
# ------------------------------------------------------------------------------
logger.info("Loading BCDMS F2 data and constructing T3 convolution tables...")

# 1.1 Load BCDMS proton and deuteron F2 via validphys API
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

# Add unique indices for merging
df_p["idx_p"] = np.arange(len(df_p))
df_d["idx_d"] = np.arange(len(df_d))

# Merge on (x, Q2) to form F2_p - F2_d
mp = 0.938  # proton mass [GeV]
mp2 = mp**2
merged_df = df_p.merge(df_d, on=["x", "Q2"], suffixes=("_p", "_d")).assign(
    y_val=lambda df: df["F2_p"] - df["F2_d"],
    W2=lambda df: df["Q2"] * (1 - df["x"]) / df["x"] + mp2,
)
# %%
# 1.2 Build FK tables for T3 = u+ubar - d - dbar
t3_index = 2  # flavor index for T3 in the FK table
loader = Loader()
fk_p = load_fktable(loader.check_fktable(setname="BCDMSP", theoryID=208, cfac=()))
fk_d = load_fktable(loader.check_fktable(setname="BCDMSD", theoryID=208, cfac=()))
wp = fk_p.get_np_fktable()  # shape: (N_data_fk, N_flav, N_grid)
wd = fk_d.get_np_fktable()
wp_t3 = wp[:, t3_index, :]
wd_t3 = wd[:, t3_index, :]


idx_p = merged_df["idx_p"].to_numpy()
idx_d = merged_df["idx_d"].to_numpy()
W = wp_t3[idx_p] - wd_t3[idx_d]


# Query the dataset-input covariance
params = {
    "dataset_inputs": [inp_p["dataset_input"], inp_d["dataset_input"]],
    "use_cuts": "internal",
    "theoryid": 208,
}


cov_full = API.dataset_inputs_covmat_from_systematics(**params)


n_p = len(df_p)
# Partition the full cov matrix
C_pp = cov_full[:n_p, :n_p]
C_dd = cov_full[n_p:, n_p:]
C_pd = cov_full[:n_p, n_p:]
# Build C_yy = C_pp[ii,ii] + C_dd[jj,jj] - 2 C_pd[ii,jj]
C_yy = C_pp[np.ix_(idx_p, idx_p)] + C_dd[np.ix_(idx_d, idx_d)] - 2 * C_pd[np.ix_(idx_p, idx_d)]
# Symmetrize
C_yy = 0.5 * (C_yy + C_yy.T)
# Add jitter until positive-definite
jitter = 1e-6 * np.mean(np.diag(C_yy))
for _ in range(10):
    try:
        np.linalg.cholesky(C_yy)
        break
    except np.linalg.LinAlgError:
        C_yy += np.eye(C_yy.shape[0]) * jitter
        jitter *= 10
else:
    cov_err = "Covariance matrix not positive-definite after jitter."
    raise RuntimeError(cov_err)
Cinv = np.linalg.inv(C_yy)

# %%
# 1.4 Prepare tensors for PyTorch: x_grid, W, Cinv, dx weights
xgrid = fk_p.xgrid  # (N_grid,)
x_torch = torch.tensor(xgrid, dtype=torch.float32).unsqueeze(1).to(device)  # (N_grid,1)
W_torch = torch.tensor(W, dtype=torch.float32).to(device)  # (N_data, N_grid)
Cinv_torch_full = torch.tensor(Cinv, dtype=torch.float32).to(device)  # (N_data, N_data)

# Compute trapezoidal weights on xgrid for valence-sum normalization
# dx[i] = x[i+1] - x[i]; then dx_low/dx_high to approximate ∫ f(x) dx

dx = (x_torch[1:] - x_torch[:-1]).squeeze()
dx_low = torch.cat([dx, dx[-1:]], dim=0).squeeze()
dx_high = torch.cat([dx[0:1], dx], dim=0).squeeze()
weights = 0.5 * (dx_low + dx_high)  # (N_grid,)

# 1.5 Compute LHAPDF T3_ref(x) for closure tests (normalized so ∫ T3 dx = 1)
pdfset = lhapdf.getPDFSet("NNPDF40_nnlo_as_01180")
pdf0 = pdfset.mkPDF(0)
Qref = fk_p.Q0
T3_ref_list = []
for x in xgrid:
    u, ub = pdf0.xfxQ(2, x, Qref), pdf0.xfxQ(-2, x, Qref)
    d, db = pdf0.xfxQ(1, x, Qref), pdf0.xfxQ(-1, x, Qref)
    T3_ref_list.append((u + ub) - (d + db))
T3_ref = np.array(T3_ref_list)  # x * T3_true(x)
# Normalize
T3_val_ref = T3_ref / xgrid
I_truth = np.trapz(T3_val_ref, xgrid)
T3_ref_norm = T3_ref * (1.0 / I_truth)  # now ∫ T3 dx = 1

y_pseudo_mean = W @ T3_ref_norm  # N_data-length vector of closure pseudo-data means

y_real = merged_df["y_val"].to_numpy()  # real BCDMS diff (unused in closure)


# %%
# 2. Define the Physics-Inspired Neural Network Model for T3
# ------------------------------------------------------------------------------
class T3Net(nn.Module):
    """Neural network for non-singlet PDF T3(x) with preprocessing x^alpha (1-x)^beta.

    Args:
        n_hidden: int, number of units per hidden layer.
        n_layers: int, number of hidden layers.
        init_alpha: float, initial small-x exponent.
        init_beta: float, initial large-x exponent.
        dropout: float, dropout probability between layers.
        use_bsm: bool, if True, add a trainable scalar `c` for BSM.
    """

    def __init__(
        self,
        n_hidden: int,
        n_layers: int = 3,
        init_alpha: float = 1.0,
        init_beta: float = 3.0,
        dropout: float = 0.2,
        use_bsm: bool = False,
    ):
        super().__init__()
        # Log-parametrized exponents ensure alpha,beta > 0
        self.logalpha = nn.Parameter(torch.log(torch.tensor(init_alpha)))
        self.logbeta = nn.Parameter(torch.log(torch.tensor(init_beta)))

        # Build MLP stack: [Linear→Tanh→BatchNorm] × n_layers, finishing with Linear→SoftPlus
        layers = [nn.Linear(1, n_hidden), nn.Tanh(), nn.BatchNorm1d(n_hidden)]
        for _ in range(n_layers - 1):
            layers += [
                nn.Linear(n_hidden, n_hidden),
                nn.BatchNorm1d(n_hidden),
                nn.Tanh(),
                nn.Dropout(dropout),
            ]
        layers.append(nn.Linear(n_hidden, 1))  # raw output (scalar per x)
        self.net = nn.Sequential(*layers)

        self.use_bsm = use_bsm
        if self.use_bsm:
            # Trainable scalar `c` for BSM deformation
            self.c = nn.Parameter(torch.tensor(0.0))  # start near 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass to compute x * T3_unc(x).

        Args:
            x: torch.Tensor of shape (N_grid, 1), input x values.

        Returns:
            f_raw: torch.Tensor of shape (N_grid,), representing x * T3_unc(x) ≥ 0.
        """
        raw = self.net(x)  # (N_grid, 1)
        pos = torch_func.softplus(raw).squeeze()  # ensure non-negative (N_grid,)

        # Compute alpha, beta from log parameters
        alpha = torch.exp(self.logalpha).clamp(min=1e-3)
        beta = torch.exp(self.logbeta).clamp(min=1e-3)
        x_ = x.squeeze().clamp(min=1e-6, max=1.0 - 1e-6)

        # Preprocessing x^alpha (1 - x)^beta
        pre = x_.pow(alpha) * (1.0 - x_).pow(beta)
        return pre * pos  # (N_grid,) = x * T3_unc(x)


# %%
# 3. Replica-Fitting Routine (SM-only or BSM)
# ------------------------------------------------------------------------------


def run_replicas(
    n_replicas: int,
    n_hidden: int,
    n_layers: int,
    dropout: float,
    lambda_smooth: float,
    patience: int,
    num_epochs: int,
    use_bsm: bool = False,  # noqa: FBT001
    K_vector: np.ndarray = None,
) -> tuple:
    """Runs N replicas of the T3-fitting procedure.

    If `use_bsm` is False, trains only the PDF network.  If True, adds a scalar `c` and
    deforms each theory prediction by (1 + c*K_vector).

    Args:
        n_replicas: int, number of Monte Carlo replicas to run.
        n_hidden: int, hidden-layer size for the neural network.
        n_layers: int, number of hidden layers.
        dropout: float, dropout probability between layers.
        lambda_smooth: float, curvature-penalty coefficient.
        patience: int, epochs of no improvement before early stopping.
        num_epochs: int, maximum epochs per replica.
        use_bsm: bool, whether to include a trainable BSM coefficient `c`.
        K_vector: np.ndarray of shape (N_data,), precomputed shape if use_bsm=True.

    Returns:
        results_df: pandas.DataFrame with columns [replica, chi2, chi2_perpt, alpha, beta,
        (c if BSM)].
        fits_array: np.ndarray of shape (n_kept, N_grid), kept x*T3(x) fits.
        c_list: list of fitted c values (None if not BSM).
    """
    N_data = W.shape[0]
    N_val = int(0.10 * N_data)

    data_records = {"replica": [], "chi2": [], "chi2_perpt": [], "alpha": [], "beta": []}
    if use_bsm:
        data_records["c"] = []
    data_records["train_history"] = []
    data_records["val_history"] = []
    data_records["fit"] = []

    all_chi2pt = []
    kept_chi2pt = []

    # Convert K_vector to torch once
    if use_bsm:
        K_torch_full = torch.tensor(K_vector, dtype=torch.float32, device=device)

    for i in range(n_replicas):
        logger.info(f"Starting replica {i} | BSM={'Yes' if use_bsm else 'No'}")

        # --- Generate pseudo-data for closure: y_pseudo_i = W @ T3_ref_norm + Gaussian noise
        rng = np.random.default_rng(42 + i)
        y_pseudo_i = rng.multivariate_normal(y_pseudo_mean, C_yy)
        y_torch_rep = torch.tensor(y_pseudo_i, dtype=torch.float32, device=device)

        # --- Split data indices 90% train / 10% val
        idx = np.arange(N_data)
        train_idx, val_idx = train_test_split(idx, test_size=0.1, random_state=42 + i)

        # --- Build sub-covariances and invert
        C_tr = C_yy[np.ix_(train_idx, train_idx)]
        C_val = C_yy[np.ix_(val_idx, val_idx)]
        Cinv_tr = torch.tensor(np.linalg.inv(C_tr), dtype=torch.float32, device=device)
        Cinv_val = torch.tensor(np.linalg.inv(C_val), dtype=torch.float32, device=device)

        # --- Initialize model & optimizer
        model = T3Net(n_hidden=n_hidden, n_layers=n_layers, dropout=dropout, use_bsm=use_bsm).to(
            device,
        )
        if use_bsm:
            # Separate parameter groups: faster lr for c
            optimizer = Adam(
                [
                    {"params": model.net.parameters()},
                    {"params": [model.logalpha, model.logbeta], "weight_decay": 1e-4},
                    {"params": [model.c], "lr": 1e-2},
                ],
                lr=1e-3,
            )
        else:
            optimizer = Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

        train_hist = []
        val_hist = []
        best_val = float("inf")
        wait = 0

        # --- Epoch loop
        for epoch in range(1, num_epochs + 1):
            model.train()
            optimizer.zero_grad()

            # 1) Forward: compute x*T3_unc(x)
            f_raw = model(x_torch)  # (N_grid,)
            t3_unnorm = f_raw / x_torch.squeeze()  # T3_unc(x)
            I_raw = torch.dot(weights.to(device), t3_unnorm)
            A = 1.0 / I_raw
            f_norm = A * f_raw  # (N_grid,) = x*T3_pred(x)

            # 2) Convolution: SM theory prediction
            T_SM_pred = torch.matmul(W_torch, f_norm)  # (N_data,)
            T_pred = T_SM_pred * (1.0 + model.c * K_torch_full) if use_bsm else T_SM_pred

            # 3) Chi^2 on train subset
            resid_tr = T_pred[train_idx] - y_torch_rep[train_idx]
            loss_chi2 = resid_tr @ (Cinv_tr @ resid_tr)

            # 4) Curvature penalty on T3_unc
            dx_vals = (x_torch[1:] - x_torch[:-1]).squeeze()
            t3 = t3_unnorm
            d2 = (t3[:-2] - 2 * t3[1:-1] + t3[2:]) / (dx_vals[:-1] ** 2)
            loss_smooth = torch.sum(d2.pow(2))

            # 5) Total loss
            loss_total = loss_chi2 + lambda_smooth * loss_smooth
            loss_total.backward()
            optimizer.step()
            train_hist.append(loss_total.item())

            # Validation pass (chi^2 only)
            model.eval()
            with torch.no_grad():
                f_raw_val = model(x_torch).squeeze()
                t3_unnorm_val = f_raw_val / x_torch.squeeze()
                I_raw_val = torch.dot(weights.to(device), t3_unnorm_val)
                A_val = 1.0 / I_raw_val
                f_val = A_val * f_raw_val
                T_SM_val = torch.matmul(W_torch[val_idx], f_val)
                if use_bsm:
                    T_pred_val = T_SM_val * (1.0 + model.c * K_torch_full[val_idx])
                else:
                    T_pred_val = T_SM_val
                resid_val = T_pred_val - y_torch_rep[val_idx]
                loss_val = resid_val @ (Cinv_val @ resid_val)
            val_hist.append(loss_val.item())

            if epoch % 200 == 0:
                logger.info(f"Replica {i}, Epoch {epoch}, Val chi2 = {loss_val:.3f}")

            # Early stopping on validation
            if loss_val.item() < best_val:
                best_val = loss_val.item()
                wait = 0
                # Save best model
                state_name = f"{'bsm_' if use_bsm else ''}t3_replica_{i}.pt"
                torch.save(model.state_dict(), model_state_dir / state_name)
            else:
                wait += 1
                if wait >= patience:
                    logger.info(f"Replica {i} early-stopped at epoch {epoch}")
                    break

        # --- Load best model & compute final chisquared on validation subset
        state_name = f"{'bsm_' if use_bsm else ''}t3_replica_{i}.pt"
        model.load_state_dict(torch.load(model_state_dir / state_name))
        model.eval()
        with torch.no_grad():
            f_raw_best = model(x_torch).squeeze()
            t3_unnorm_best = f_raw_best / x_torch.squeeze()
            I_raw_best = torch.dot(weights.to(device), t3_unnorm_best)
            A_best = 1.0 / I_raw_best
            f_best_norm = A_best * f_raw_best
            T_SM_best_val = torch.matmul(W_torch[val_idx], f_best_norm)
            if use_bsm:
                T_pred_val_best = T_SM_best_val * (1.0 + model.c * K_torch_full[val_idx])
            else:
                T_pred_val_best = T_SM_best_val
            resid_v = T_pred_val_best - y_torch_rep[val_idx]
            chi2_val = float(resid_v @ (Cinv_val @ resid_v))
        chi2_perpt = chi2_val / N_val
        all_chi2pt.append(chi2_perpt)

        # Replica selection: keep if chisquared/pt ∈ [0.5, 1.5]
        if 0.8 <= chi2_perpt <= 1.2:
            kept_chi2pt.append(chi2_perpt)
            alpha_val = float(torch.exp(model.logalpha).item())
            beta_val = float(torch.exp(model.logbeta).item())
            data_records["replica"].append(i)
            data_records["chi2"].append(chi2_val)
            data_records["chi2_perpt"].append(chi2_perpt)
            data_records["alpha"].append(alpha_val)
            data_records["beta"].append(beta_val)
            if use_bsm:
                data_records["c"].append(float(model.c.item()))
            data_records["train_history"].append(train_hist)
            data_records["val_history"].append(val_hist)
            data_records["fit"].append(f_best_norm.cpu().numpy())
            logger.success(f"Replica {i} kept: chi2 = {chi2_val:.2f}, chi2/pt = {chi2_perpt:.2f}")
            logger.success(f"alpha = {alpha_val:.3f}, beta = {beta_val:.3f}")
            logger.success({f"c = {model.c.item()!s}" if use_bsm else ""})
        else:
            logger.warning(f"Replica {i} discarded: chi2/pt = {chi2_perpt:.2f} outside [0.5,1.5]")

    # Assemble results DataFrame
    results = {
        "replica": data_records["replica"],
        "chi2": data_records["chi2"],
        "chi2_perpt": data_records["chi2_perpt"],
        "alpha": data_records["alpha"],
        "beta": data_records["beta"],
    }
    if use_bsm:
        results["c"] = data_records["c"]
    results_df = pd.DataFrame(results)
    results_df["train_history"] = pd.Series(data_records["train_history"], dtype="object")
    results_df["val_history"] = pd.Series(data_records["val_history"], dtype="object")
    results_df["fit"] = pd.Series(data_records["fit"], dtype="object")

    # Stack fits into array for ensemble stats
    fits_array = np.stack(results_df["fit"].values)  # shape: (n_kept, N_grid)
    c_list = data_records["c"] if use_bsm else None
    return results_df, fits_array, c_list, all_chi2pt, kept_chi2pt


# %%
# 4. Compute BSM Shape Vector K(Q^2)
# ------------------------------------------------------------------------------
Q2_vals = merged_df["Q2"].to_numpy()
Q2_min = Q2_vals.min()
scale = 1e4  # chosen so that raw K is O(1) before standardization
K_raw = ((Q2_vals - Q2_min) ** 2) / (scale**2)
# Standardize to zero mean, unit std
K_mean = K_raw.mean()
K_std = K_raw.std()
K_vector = (K_raw - K_mean) / K_std

# Plot raw distribution of K(Q2)
plt.figure(figsize=(5, 4))
plt.hist(K_raw, bins=30, alpha=0.7, edgecolor="black")
plt.xlabel(r"Raw $K_\mathrm{raw}(Q^2)$")
plt.ylabel("Counts")
plt.title(r"Distribution of Raw BSM Shape $K_\mathrm{raw}$")
plt.tight_layout()
plt.savefig(images_dir / "K_raw_distribution.png", dpi=300)
plt.show()

plt.figure(figsize=(5, 4))
plt.hist(K_vector, bins=30, alpha=0.7, edgecolor="black")
plt.xlabel(r"Standardized $K(Q^2)$")
plt.ylabel("Counts")
plt.title(r"Distribution of Standardized BSM Shape $K$ (mean=0,std=1)")
plt.tight_layout()
plt.savefig(images_dir / "K_std_distribution.png", dpi=300)
plt.show()

# %%
# 5. Run SM-only (Base) Replica Fits
# ------------------------------------------------------------------------------
n_replicas = 100
logger.info("\n=== Running Base (SM-only) Replica Fits ===")
results_base_df, fits_base, _, all_chi2pt_base, kept_chi2pt_base = run_replicas(
    n_replicas=n_replicas,
    n_hidden=30,
    n_layers=3,
    dropout=0.2,
    lambda_smooth=1e-4,
    patience=1000,
    num_epochs=5000,
    use_bsm=False,
)

logger.info(f"Base model: kept {len(results_base_df)} out of {n_replicas} replicas.")

# %%
# 6. Run BSM Replica Fits
# ------------------------------------------------------------------------------
logger.info("\n=== Running BSM Replica Fits ===")
results_bsm_df, fits_bsm, c_list, all_chi2pt_bsm, kept_chi2pt_bsm = run_replicas(
    n_replicas=n_replicas,
    n_hidden=30,
    n_layers=3,
    dropout=0.2,
    lambda_smooth=1e-4,
    patience=1000,
    num_epochs=5000,
    use_bsm=True,
    K_vector=K_vector,
)

logger.info(f"BSM model: kept {len(results_bsm_df)} out of {n_replicas} replicas.")


# %%
# 7. Ensemble Statistics: Mean & Std for x*T3(x)
# ------------------------------------------------------------------------------


mean_base, std_base = fits_base.mean(axis=0), fits_base.std(axis=0)
mean_bsm, std_bsm = fits_bsm.mean(axis=0), fits_bsm.std(axis=0)

# %%
# 8. Plot Ensemble vs. Truth: Base vs. BSM
# ------------------------------------------------------------------------------
plt.figure(figsize=(8, 6))
# SM-only
plt.fill_between(
    xgrid,
    mean_base - std_base,
    mean_base + std_base,
    alpha=0.3,
    label="Base Envelope",
)
plt.plot(xgrid, mean_base, "-", label="Base Mean")
# BSM
plt.fill_between(
    xgrid,
    mean_bsm - std_bsm,
    mean_bsm + std_bsm,
    alpha=0.3,
    color="C1",
    label="BSM Envelope",
)
plt.plot(xgrid, mean_bsm, "-", color="C1", label="BSM Mean")
# Truth
plt.plot(xgrid, T3_ref_norm, "--", color="k", label="Truth (SM)")

plt.xscale("log")
plt.xlabel(r"$x$")
plt.ylabel(r"$x \,T_3(x)$")
plt.title("PDF Ensemble Comparison: Base vs. BSM vs. Truth")
plt.legend(fontsize="small")
plt.tight_layout()
plt.savefig(images_dir / "ensemble_comparison.png", dpi=300)
plt.show()

# %%
# 9. Histograms: alpha & beta Distributions for Base vs. BSM
# ------------------------------------------------------------------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
# Small-x exponent alpha
ax1.hist(results_base_df["alpha"], bins="auto", alpha=0.6, edgecolor="black", label="Base")
ax1.hist(results_bsm_df["alpha"], bins="auto", alpha=0.6, edgecolor="black", label="BSM")
ax1.set_xlabel(r"$\alpha$")
ax1.set_ylabel("Replicas")
ax1.set_title(r"Distribution of Small-$x$ Exponent $\alpha$")
ax1.legend()
# Large-x exponent beta
ax2.hist(results_base_df["beta"], bins="auto", alpha=0.6, edgecolor="black", label="Base")
ax2.hist(results_bsm_df["beta"], bins="auto", alpha=0.6, edgecolor="black", label="BSM")
ax2.set_xlabel(r"$\beta$")
ax2.set_ylabel("Replicas")
ax2.set_title(r"Distribution of Large-$x$ Exponent $\beta$")
ax2.legend()
fig.tight_layout()
fig.savefig(images_dir / "alpha_beta_comparison.png", dpi=300)
plt.show()

# %%
# 10. Scatter alpha vs. beta: Base vs. BSM
# ------------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(5, 5))
ax.scatter(results_base_df["alpha"], results_base_df["beta"], alpha=0.7, label="Base")
ax.scatter(results_bsm_df["alpha"], results_bsm_df["beta"], alpha=0.7, label="BSM")
ax.set_xlabel(r"$\alpha$")
ax.set_ylabel(r"$\beta$")
ax.set_title(r"Scatter: $\alpha$ vs. $\beta$")
ax.legend()
plt.tight_layout()
plt.savefig(images_dir / "alpha_vs_beta_scatter.png", dpi=300)
plt.show()

# %%
# 11. Histogram of chisquared/pt: Base vs. BSM (All vs. Kept)
# ------------------------------------------------------------------------------
# (a) All replicas
fig, ax = plt.subplots(figsize=(6, 4))
ax.hist(all_chi2pt_base, bins=20, alpha=0.5, edgecolor="black", label="Base All")
ax.hist(all_chi2pt_bsm, bins=20, alpha=0.5, edgecolor="black", label="BSM All")
ax.axvline(0.5, color="red", linestyle="--")
ax.axvline(1.5, color="red", linestyle="--")
ax.set_xlabel(r"$\chi^2/\mathrm{pt}$")
ax.set_ylabel("Replicas")
ax.set_title(r"All Replicas: Distribution of $\chi^2/\mathrm{pt}$")
ax.legend()
fig.tight_layout()
fig.savefig(images_dir / "chi2pt_all_comparison.png", dpi=300)
plt.show()

# (b) Kept replicas only
fig, ax = plt.subplots(figsize=(6, 4))
ax.hist(kept_chi2pt_base, bins=20, alpha=0.5, edgecolor="black", label="Base Kept")
ax.hist(kept_chi2pt_bsm, bins=20, alpha=0.5, edgecolor="black", label="BSM Kept")
ax.axvline(0.5, color="red", linestyle="--")
ax.axvline(1.5, color="red", linestyle="--")
ax.set_xlabel(r"$\chi^2/\mathrm{pt}$")
ax.set_ylabel("Replicas")
ax.set_title(r"Kept Replicas: Distribution of $\chi^2/\mathrm{pt}$")
ax.legend()
fig.tight_layout()
fig.savefig(images_dir / "chi2pt_kept_comparison.png", dpi=300)
plt.show()

# %%
# 12. Histogram of BSM Coefficient c (Kept Replicas)
# ------------------------------------------------------------------------------
plt.figure(figsize=(6, 4))
plt.hist(c_list, bins="auto", alpha=0.7, edgecolor="black")
plt.xlabel(r"Fitted BSM Parameter $c$")
plt.ylabel("Replicas")
plt.title(r"Distribution of Fitted BSM Coefficient $c$ (Kept)")
plt.tight_layout()
plt.savefig(images_dir / "c_distribution_kept.png", dpi=300)
plt.show()


# %%
# 13. Cross-Checks & Stability Plots
# ------------------------------------------------------------------------------
# 13.1 First moment <x> = ∫ x T3(x) dx for Base vs. BSM (no separate function)

# Compute first moments for Base
first_mom_base = []
for f in fits_base:
    T3 = f / xgrid  # recover T3(x) from x*T3(x)
    m1 = np.trapz(xgrid * T3, xgrid)  # noqa: NPY201
    first_mom_base.append(m1)

# Compute first moments for BSM
first_mom_bsm = []
for f in fits_bsm:
    T3 = f / xgrid
    m1 = np.trapz(xgrid * T3, xgrid)  # noqa: NPY201
    first_mom_bsm.append(m1)

# Plot both histograms on the same axes
plt.figure(figsize=(6, 4))
plt.hist(first_mom_base, bins=10, alpha=0.6, edgecolor="black", label="Base")
plt.hist(first_mom_bsm, bins=10, alpha=0.6, edgecolor="black", label="BSM")
plt.xlabel(r"$\langle x \rangle_{T_3}$")
plt.ylabel("Replicas")
plt.title(r"Comparison of First Moment $\langle x \rangle_{T_3}$: Base vs. BSM")
plt.legend(fontsize="small")
plt.tight_layout()
plt.savefig(images_dir / "first_moment_combined.png", dpi=300)
plt.show()


# 13.3 Scatter: c vs. alpha and c vs. beta (BSM only) as subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5), sharey=True)

# c vs alpha
ax1.scatter(results_bsm_df["alpha"], c_list, alpha=0.7)
ax1.set_xlabel(r"$\alpha$")
ax1.set_ylabel(r"$c$")
ax1.set_title(r"BSM $c$ vs. $\alpha$")

# c vs beta
ax2.scatter(results_bsm_df["beta"], c_list, alpha=0.7, color="C1")
ax2.set_xlabel(r"$\beta$")
ax2.set_title(r"BSM $c$ vs. $\beta$")

fig.tight_layout()
fig.savefig(images_dir / "c_vs_alpha_beta.png", dpi=300)
plt.show()

# 13.4 Stability: mean ± std of c as function of number of replicas kept
# Compute cumulative means/stds by sorting |chi2pt - 1| (proxy for stability)
sorted_indices = np.argsort(np.abs(np.array(results_bsm_df["chi2_perpt"]) - 1.0))
cum_means = []
cum_stds = []
c_vals_sorted = np.array(c_list)[sorted_indices]
for k in range(1, len(c_vals_sorted) + 1):
    subset = c_vals_sorted[:k]
    cum_means.append(subset.mean())
    cum_stds.append(subset.std())

plt.figure(figsize=(6, 4))
plt.plot(np.arange(1, len(c_vals_sorted) + 1), cum_means, label=r"Mean $c$ (cumulative)")
plt.fill_between(
    np.arange(1, len(c_vals_sorted) + 1),
    np.array(cum_means) - np.array(cum_stds),
    np.array(cum_means) + np.array(cum_stds),
    alpha=0.2,
    label=r"$\pm 1\sigma$ band",
)
plt.xlabel("Number of Top-K Replicas Included")
plt.ylabel(r"Mean BSM $c$")
plt.title(r"Stability of $\overline{c}$ vs. Number of Replicas")
plt.legend(fontsize="small")
plt.tight_layout()
plt.savefig(images_dir / "c_stability_vs_n.png", dpi=300)
plt.show()

# 13.5 C Value Histogram
plt.figure(figsize=(6, 4))
plt.hist(c_list, bins="auto", alpha=0.7, edgecolor="black")
# Draw a vertical line at c=0
plt.axvline(0.0, color="red", linestyle="--", label="c = 0")

# Compute mean and 1sigma
mean_c = np.mean(c_list)
std_c = np.std(c_list)
plt.axvline(mean_c, color="C1", linestyle="-", label=f"mean = {mean_c:.3f}")
plt.fill_betweenx(
    [0, plt.gca().get_ylim()[1]],
    mean_c - std_c,
    mean_c + std_c,
    color="C1",
    alpha=0.2,
    label=rf"±1$\sigma$ = {std_c:.3f}",
)

plt.xlabel(r"BSM Parameter $c$")
plt.ylabel("Number of Replicas")
plt.title(r"Histogram of Fitted BSM $c$ (Kept Replicas)")
plt.legend(fontsize="small")
plt.tight_layout()
plt.savefig(images_dir / "c_histogram_with_stats.png", dpi=300)
plt.show()


# %%
# 14. Summary Table of Key Statistics
# ------------------------------------------------------------------------------
# Build summary dictionary

summary = {
    "Model": ["Base", "BSM"],
    "Kept Replicas": [len(results_base_df), len(results_bsm_df)],
    "Mean alpha ± sigma": [
        f"{results_base_df['alpha'].mean():.3f} ± {results_base_df['alpha'].std():.3f}",
        f"{results_bsm_df['alpha'].mean():.3f} ± {results_bsm_df['alpha'].std():.3f}",
    ],
    "Mean beta ± sigma": [
        f"{results_base_df['beta'].mean():.3f} ± {results_base_df['beta'].std():.3f}",
        f"{results_bsm_df['beta'].mean():.3f} ± {results_bsm_df['beta'].std():.3f}",
    ],
    r"Mean chisquared/pt ± sigma": [
        f"{np.mean(all_chi2pt_base):.3f} ± {np.std(all_chi2pt_base):.3f}",
        f"{np.mean(all_chi2pt_bsm):.3f} ± {np.std(all_chi2pt_bsm):.3f}",
    ],
    r"Mean <x> ± sigma": [
        f"{np.mean(first_mom_base):.3f} ± {np.std(first_mom_base):.3f}",
        f"{np.mean(first_mom_bsm):.3f} ± {np.std(first_mom_bsm):.3f}",
    ],
    r"Mean c ± sigma": ["N/A", f"{np.mean(c_list):.3f} ± {np.std(c_list):.3f}"],
}


summary_df = pd.DataFrame(summary)
logger.info(summary_df.to_string(index=False))


# %%
