# %%
"""T3 Script: toy NN fit of the non-singlet PDF T3."""

from pathlib import Path

import lhapdf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as torch_f
import yaml
from loguru import logger
from torch import nn
from validphys.fkparser import load_fktable
from validphys.loader import Loader

# ──────────────────────────────────────────────────────────────────────────────
# Reproducibility
rng = np.random.default_rng(451)
torch.manual_seed(1)


BASE_DIR = Path("nnpdf_data") / "nnpdf_data" / "commondata"


# %%
# ? Functions
def load_commondata(dataset_folder: str, label: str) -> pd.DataFrame:
    """Load the 'rqcd' data variant for proton or deuteron from YAML files.

    Parameters
    ----------
    dataset_folder : str
        Name of the folder under commondata (e.g. "BCDMS_NC_NOTFIXED_P_EM-F2").
    label : str
        Column suffix for proton/deuteron, e.g. "p" or "d".

    Returns:
    -------
    pd.DataFrame
        Columns:
          - x, Q2 : kinematic points
          - F2_{label} : central F2 values
          - stat_{label}, sys_{label}, norm_{label} : absolute uncertainties
    """
    base = BASE_DIR / dataset_folder

    # 1) central values
    with (base / "data_rqcd.yaml").open() as f:
        central = yaml.safe_load(f)["data_central"]

    # 2) kinematics
    with (base / "kinematics_EM-F2-HEPDATA.yaml").open() as f:
        kin = yaml.safe_load(f)["bins"]
    xs = [b["x"]["mid"] for b in kin]
    q2s = [b["Q2"]["mid"] for b in kin]

    # 3) uncertainties
    with (base / "uncertainties_rqcd.yaml").open() as f:
        unc = yaml.safe_load(f)["bins"]

    return pd.DataFrame(
        {
            "x": xs,
            "Q2": q2s,
            f"F2_{label}": central,
            f"stat_{label}": [u["stat"] for u in unc],
            f"sys_{label}": [u["sys"] * v for u, v in zip(unc, central)],
            f"norm_{label}": [u["norm"] * v for u, v in zip(unc, central)],
        },
    )


def compute_covariance(exp_df: pd.DataFrame) -> np.ndarray:
    """Compute the combined covariance C_y = C_p + C_d for y = F2_p - F2_d.

    Parameters
    ----------
    exp_df : pd.DataFrame
        Must contain columns:
        stat_p, sys_p, norm_p, stat_d, sys_d, norm_d

    Returns:
    -------
    C_y : np.ndarray
        The (NxN) covariance matrix where N = len(exp_df).
    """
    # proton part
    sp = exp_df["stat_p"].to_numpy()
    xp = exp_df["sys_p"].to_numpy()
    np_ = exp_df["norm_p"].to_numpy()
    c_p = np.diag(sp**2) + np.outer(xp, xp) + np.outer(np_, np_)

    # deuteron part
    sd = exp_df["stat_d"].to_numpy()
    xd = exp_df["sys_d"].to_numpy()
    nd = exp_df["norm_d"].to_numpy()
    c_p = np.diag(sd**2) + np.outer(xd, xd) + np.outer(nd, nd)

    return c_p + c_p


class T3Net(nn.Module):
    """Neural network parametrization of T3(x)=A x^{1-alpha}(1-x)^beta N(x).

    Attributes:
    ----------
    net : nn.Sequential
        The hidden layers (Linear + Tanh).
    A : nn.Parameter
        Overall normalization factor.
    alpha : float
        Small-x preprocessing exponent.
    beta : float
        Large-x preprocessing exponent.
    """

    def __init__(self, n_hidden: int, alpha: float, beta: float) -> None:
        """Torch model."""
        """Parameters
        ----------
        n_hidden : int
            Number of neurons in each hidden layer.
        alpha : float
            Preprocessing exponent at small x.
        beta : float
            Preprocessing exponent at large x.
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, n_hidden),
            nn.Tanh(),
            nn.Linear(n_hidden, n_hidden),
            nn.Tanh(),
            nn.Linear(n_hidden, 1),
        )
        self.A = nn.Parameter(torch.tensor(1.0))
        self.alpha = alpha
        self.beta = beta

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute T3(x) on the grid.

        Parameters
        ----------
        x : torch.Tensor
            Shape (N_xgrid, 1)

        Returns:
        -------
        torch.Tensor
            Shape (N_xgrid, 1), the PDF values.
        """
        raw = self.net(x)  # shape (N,1)
        positive = torch_f.softplus(raw)  # ≥0
        pre = x.pow(1 - self.alpha) * (1 - x).pow(self.beta)
        return self.A * pre * positive


def chi2(
    model: T3Net,
    w: torch.Tensor,
    y: torch.Tensor,
    c_inv: torch.Tensor,
) -> torch.Tensor:
    """Compute χ² = (W f - y)^T Cinv (W f - y).

    Parameters
    ----------
    model : T3Net
        The neural network instance.
    W : torch.Tensor
        FK kernel subset, shape (N_points, N_xgrid).
    y : torch.Tensor
        Data vector, shape (N_points,).
    Cinv : torch.Tensor
        Inverse covariance for those points, shape (N_points, N_points).

    Returns:
    -------
    torch.Tensor
        Scalar χ² value.
    """
    f = model(x_torch).squeeze()  # (N_xgrid,)
    d = w @ f - y  # (N_points,)
    return d @ (c_inv @ d)  # scalar


# %%
# ──────────────────────────────────────────────────────────────────────────────
# 1) Load & merge data
loader = Loader()
df_p = (
    load_commondata("BCDMS_NC_NOTFIXED_P", "p")
    .reset_index(drop=True)
    .reset_index()
    .rename(columns={"index": "idx_p"})
)
df_d = (
    load_commondata("BCDMS_NC_NOTFIXED_D", "d")
    .reset_index(drop=True)
    .reset_index()
    .rename(columns={"index": "idx_d"})
)
exp_df = df_p.merge(df_d, on=["x", "Q2"], how="inner").assign(y=lambda d: d["F2_p"] - d["F2_d"])
# %%
# 2) Build full covariance & its inverse
C_y = compute_covariance(exp_df)
Cinv_np = np.linalg.inv(C_y)
Cinv_t = torch.tensor(Cinv_np, dtype=torch.float32)
# %%
# 3) Extract non-singlet FK kernel & x-grid

flavor_index = 2

fk_p = load_fktable(loader.check_fktable(setname="BCDMSP", cfac=(), theoryID=200))
fk_d = load_fktable(loader.check_fktable(setname="BCDMSD", cfac=(), theoryID=200))

wp = fk_p.get_np_fktable()  # (N_p, 5, N_xgrid)
wd = fk_d.get_np_fktable()  # (N_d, 5, N_xgrid)

# ensure both share same x-grid
x_p = fk_p.xgrid
xgrid = x_p

# select only the matched data rows
idx_p = exp_df["idx_p"].to_numpy()
idx_d = exp_df["idx_d"].to_numpy()
wp_sel = wp[idx_p, flavor_index, :]
wd_sel = wd[idx_d, flavor_index, :]
w_ns = wp_sel - wd_sel

W_t = torch.tensor(w_ns, dtype=torch.float32)
# %%
# 4) Pack data into torch
x_torch = torch.tensor(xgrid, dtype=torch.float32).unsqueeze(1)  # (N_xgrid,1)
y_torch = torch.tensor(exp_df["y"].to_numpy(), dtype=torch.float32)
# %%
# 5) Train/validation split
N = len(exp_df)
idx_all = np.arange(N)
rng.shuffle(idx_all)
n_val = int(0.2 * N)
val_idx, train_idx = idx_all[:n_val], idx_all[n_val:]
W_train, W_val = W_t[train_idx], W_t[val_idx]
y_train, y_val = y_torch[train_idx], y_torch[val_idx]
Cinv_train = Cinv_t[train_idx][:, train_idx]
Cinv_val = Cinv_t[val_idx][:, val_idx]
# %%
# 6) Build model & optimizer
model = T3Net(n_hidden=30, alpha=1.0, beta=3.0)
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
best_val, wait, patience = float("inf"), 0, 10
# %%
# 7) Training loop with early stopping
for epoch in range(1, 501):
    model.train()
    opt.zero_grad()
    train_loss = chi2(model, W_train, y_train, Cinv_train)
    train_loss.backward()
    opt.step()

    val_loss = chi2(model, W_val, y_val, Cinv_val).item()
    if val_loss < best_val:
        best_val, wait = val_loss, 0
        torch.save(model.state_dict(), "best.pt")
    else:
        wait += 1
        if wait >= patience:
            logger.info(f"Early stopping at epoch {epoch}")
            break

    if epoch % 50 == 0:
        logger.info(f"Epoch {epoch}: train χ²={train_loss:.2f}, val χ²={val_loss:.2f}")
# %%
# 8) Load best model & plot fit
model.load_state_dict(torch.load("best.pt"))
model.eval()
with torch.no_grad():
    T3_fit = model(x_torch).squeeze().numpy()

plt.loglog(xgrid, T3_fit, label="NN fit")
plt.xlabel("x")
plt.ylabel(r"$T_3(x)$")
plt.legend()
plt.show()
# %%
# 9) Monte Carlo replicas for uncertainty
n_rep = 100
T3_reps = np.zeros((n_rep, len(xgrid)))
for r in range(n_rep):
    y_rep = rng.multivariate_normal(exp_df["y"].to_numpy(), cov=C_y)
    y_t = torch.tensor(y_rep, dtype=torch.float32)
    model_r = T3Net(n_hidden=30, alpha=1.0, beta=3.0)
    opt_r = torch.optim.Adam(model_r.parameters(), lr=1e-3)
    # shorter training per replica
    for _ in range(100):
        opt_r.zero_grad()
        loss_r = chi2(model_r, W_t, y_t, Cinv_t)
        loss_r.backward()
        opt_r.step()
    with torch.no_grad():
        T3_reps[r] = model_r(x_torch).squeeze().numpy()

mean_t3 = T3_reps.mean(axis=0)
std_t3 = T3_reps.std(axis=0)

plt.fill_between(xgrid, mean_t3 - std_t3, mean_t3 + std_t3, alpha=0.3)
plt.loglog(xgrid, mean_t3, label="mean ±1sigma")
plt.xlabel("x")
plt.ylabel(r"$T_3(x)$")
plt.legend()
plt.show()
# %%
# 10) Compare to reference PDF from LHAPDF
pdf = lhapdf.getPDFSet("NNPDF40_nnlo_as_01180").mkPDF(0)
Qref = 10.0
T3_ref = []
for x in xgrid:
    u = pdf.xfxQ(2, x, Qref) / x
    ub = pdf.xfxQ(-2, x, Qref) / x
    d = pdf.xfxQ(1, x, Qref) / x
    db = pdf.xfxQ(-1, x, Qref) / x
    T3_ref.append((u + ub) - (d + db))
T3_ref = np.array(T3_ref)

plt.loglog(xgrid, T3_ref, label="Reference PDF")
plt.loglog(xgrid, T3_fit, label="NN fit")
plt.loglog(xgrid, mean_t3, label="Replica Mean")
plt.fill_between(xgrid, mean_t3 - std_t3, mean_t3 + std_t3, alpha=0.3)
plt.xlabel("x")
plt.ylabel(r"$T_3(x)$")
plt.legend()
plt.show()

# %%
