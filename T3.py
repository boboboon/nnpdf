# %%
"""T3 Script (Closure/Pseudo-data version, clean and documented).

Fits a neural-network PDF to pseudo-data generated from
W @ T3_ref (NNPDF4.0 central) plus correlated Gaussian noise.

"""

import lhapdf
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from loguru import logger
from torch import nn
from torch.optim import Adam
from validphys.api import API
from validphys.fkparser import load_fktable
from validphys.loader import Loader

# --- Set matplotlib to use LaTeX for all text
plt.rc("text", usetex=True)
plt.rc("font", family="serif")

# %% ---- Data Preparation ----

# 1. Load BCDMS F2_p, F2_d (legacy variant)
inp_p = {
    "dataset_input": {"dataset": "BCDMS_NC_NOTFIXED_P_EM-F2", "variant": "legacy"},
    "use_cuts": "internal",
    "theoryid": 200,
}
inp_d = {
    "dataset_input": {"dataset": "BCDMS_NC_NOTFIXED_D_EM-F2", "variant": "legacy"},
    "use_cuts": "internal",
    "theoryid": 200,
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

merged_df = df_p.merge(df_d, on=["x", "Q2"], suffixes=("_p", "_d"))

# %% ---- FK Table Construction ----
loader = Loader()
fk_p = load_fktable(loader.check_fktable(setname="BCDMSP", theoryID=200, cfac=()))
fk_d = load_fktable(loader.check_fktable(setname="BCDMSD", theoryID=200, cfac=()))
wp = fk_p.get_np_fktable()
wd = fk_d.get_np_fktable()
flavor_index = 2  # T3 = u^+ - d^+
wp_t3 = wp[:, flavor_index, :]
wd_t3 = wd[:, flavor_index, :]
idx_p = merged_df["idx_p"].to_numpy()
idx_d = merged_df["idx_d"].to_numpy()
W = wp_t3[idx_p] - wd_t3[idx_d]  # (N_data, 50)

# %% ---- Covariance ----
params = {
    "dataset_inputs": [
        {"dataset": "BCDMS_NC_NOTFIXED_P_EM-F2", "variant": "legacy"},
        {"dataset": "BCDMS_NC_NOTFIXED_D_EM-F2", "variant": "legacy"},
    ],
    "use_cuts": "internal",
    "theoryid": 200,
}
cov_full = API.dataset_inputs_covmat_from_systematics(**params)
n_p, n_d = len(df_p), len(df_d)
C_pp = cov_full[:n_p, :n_p]
C_dd = cov_full[n_p:, n_p:]
C_pd = cov_full[:n_p, n_p:]
C_yy = C_pp[np.ix_(idx_p, idx_p)] + C_dd[np.ix_(idx_d, idx_d)] - 2 * C_pd[np.ix_(idx_p, idx_d)]
eps = 1e-6 * np.mean(np.diag(C_yy))
C_yy_j = C_yy + np.eye(C_yy.shape[0]) * eps
Cinv = np.linalg.inv(C_yy_j)

# %% ---- x-grid ----
xgrid = fk_p.xgrid  # (50,)
x_torch = torch.tensor(xgrid, dtype=torch.float32).unsqueeze(1)
W_torch = torch.tensor(W, dtype=torch.float32)
Cinv_torch = torch.tensor(Cinv, dtype=torch.float32)

# %% ---- Pseudo-data Generation ----
pdfset = lhapdf.getPDFSet("NNPDF40_nnlo_as_01180")
pdf0 = pdfset.mkPDF(0)
Qref = fk_p.Q0
T3_ref = []
for x in xgrid:
    u = pdf0.xfxQ(2, x, Qref)
    ub = pdf0.xfxQ(-2, x, Qref)
    d = pdf0.xfxQ(1, x, Qref)
    db = pdf0.xfxQ(-1, x, Qref)
    T3_ref.append((u + ub) - (d + db))
T3_ref = np.array(T3_ref)  # (50,)

# Use numpy's Generator for robust RNG
rng = np.random.default_rng(42)
y_pseudo_mean = W @ T3_ref
y_pseudo = rng.multivariate_normal(y_pseudo_mean, C_yy_j)
y_torch = torch.tensor(y_pseudo, dtype=torch.float32)


# %% ---- Model Definition ----
class T3Net(nn.Module):
    """Neural network for the non-singlet PDF T3(x).
    - Input: x values (float tensor, shape (N, 1))
    - Output: x*f(x) values (float tensor, shape (N, 1))
    """

    def __init__(self, n_hidden: int, alpha: float, beta: float) -> None:
        """Args:
        n_hidden: Number of hidden units per layer.
        alpha: PDF small-x power.
        beta: PDF large-x power.
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
        """Forward pass.

        Args:
            x: Input tensor of shape (N, 1), values in (0, 1)

        Returns:
            PDF values x*f(x) of shape (N, 1)
        """
        raw = self.net(x)
        positive = F.softplus(raw)
        pre = x.pow(1 - self.alpha) * (1 - x).pow(self.beta)
        return self.A * pre * positive


def chi2(model: nn.Module) -> torch.Tensor:
    """Computes chi^2 for pseudo-data fit.

    Args:
        model: PDF neural network
    Returns:
        chi^2 value (scalar tensor)
    """
    f = model(x_torch).squeeze()  # (N_x,)
    y_pred = W_torch @ f  # (N_data,)
    resid = y_pred - y_torch
    return resid @ (Cinv_torch @ resid)


# %% ---- Training ----
model = T3Net(n_hidden=30, alpha=1.0, beta=3.0)
opt = Adam(model.parameters(), lr=1e-3)
best = float("inf")
patience, wait = 10, 0
loss_hist = []
chi2_hist = []

for epoch in range(1, 501):
    model.train()
    opt.zero_grad()
    loss = chi2(model)
    loss.backward()
    opt.step()
    val = loss.item()
    loss_hist.append(val)
    chi2_hist.append(val / len(y_pseudo))
    if val < best:
        best, wait = val, 0
        torch.save(model.state_dict(), "t3_best.pt")
    else:
        wait += 1
        if wait >= patience:
            logger.info(f"Early stopping at epoch {epoch}")
            break
    if epoch % 50 == 0:
        logger.info(f"Epoch {epoch}: $\\chi^2$ = {val:.2f}")

# %% ---- Plotting & Diagnostics ----
model.load_state_dict(torch.load("t3_best.pt"))
model.eval()
with torch.no_grad():
    T3_fit = model(x_torch).squeeze().numpy()

# PDF fit vs ground truth
plt.figure()
plt.plot(xgrid, T3_ref, label=r"NNPDF4.0 (truth, $x\,T_3$)")
plt.plot(xgrid, T3_fit, "--", label=r"NN fit (pseudo-data)")
plt.xlabel(r"$x$")
plt.ylabel(r"$x\,T_3(x)$")
plt.title(r"$x\,T_3(x)$ fit to pseudo-data")
plt.legend()
plt.tight_layout()
plt.show()

# Pseudo-data kinematics & distribution
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
sc = ax1.scatter(merged_df["x"], merged_df["Q2"], c=y_pseudo, cmap="coolwarm", s=20, alpha=0.8)
ax1.set(
    xscale="log",
    yscale="log",
    xlabel=r"$x$",
    ylabel=r"$Q^2$ [GeV$^2$]",
    title=r"Pseudo-data kinematics ($F_2^p-F_2^d$)",
)
plt.colorbar(sc, ax=ax1, label=r"$y = F_2^p-F_2^d$ (pseudo)")
ax2.hist(y_pseudo, bins=30, alpha=0.75)
ax2.set(xlabel=r"$y = F_2^p-F_2^d$ (pseudo)", ylabel="count", title="Pseudo-data $y$ distribution")
plt.tight_layout()
plt.show()

# FK convolution check: pseudo-data vs theory mean
y_ref = W @ T3_ref
plt.figure(figsize=(5, 5))
plt.scatter(y_ref, y_pseudo, alpha=0.65, s=18, label="pseudo-data vs truth")
lims = [min(y_ref.min(), y_pseudo.min()), max(y_ref.max(), y_pseudo.max())]
plt.plot(lims, lims, "k--")
plt.xlabel(r"$y_\mathrm{ref}$ (theory, $W\cdot xT_3$)")
plt.ylabel(r"$y_\mathrm{pseudo}$")
plt.title("FK convolution: pseudo-data vs theory")
plt.legend()
plt.tight_layout()
plt.show()

# Loss / chi2 per epoch
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
ax1.plot(loss_hist, label=r"Total $\chi^2$")
ax1.set_xlabel("Epoch")
ax1.set_ylabel(r"$\chi^2$")
ax1.set_title(r"Training Loss ($\chi^2$)")
ax1.legend()
ax2.plot(chi2_hist, label=r"$\chi^2$/N")
ax2.set_xlabel("Epoch")
ax2.set_ylabel(r"$\chi^2$/N")
ax2.set_title(r"Reduced $\chi^2$ per epoch")
ax2.legend()
plt.tight_layout()
plt.show()

# Check coverage of x grid
plt.figure()
coverage = np.count_nonzero(W, axis=0)
plt.semilogx(xgrid, coverage, "o-", label="Coverage per $x$ bin")
plt.xlabel(r"$x$")
plt.ylabel("Coverage")
plt.title(r"Pseudo-data: $x$-bin coverage")
plt.legend()
plt.tight_layout()
plt.show()

# Check residuals (pull distribution)
resid = (y_pseudo - y_ref) / np.sqrt(np.diag(C_yy_j))
plt.figure()
plt.hist(resid, bins=30, alpha=0.7)
plt.xlabel(r"Pull $(y_\mathrm{pseudo}-y_\mathrm{ref})/\sigma$")
plt.ylabel("Count")
plt.title("Pull distribution (should be standard normal)")
plt.tight_layout()
plt.show()

# Print fit result summary
logger.info(
    f"Fit complete: best total chi^2 = {min(loss_hist):.2f}, reduced chi^2 = {min(chi2_hist):.3f}",
)
logger.info(f"xgrid: min {xgrid.min():.3e}, max {xgrid.max():.3e}")
logger.info(f"T3_fit: min {T3_fit.min():.3e}, max {T3_fit.max():.3e}")
logger.info(f"W shape: {W.shape}, Cov shape: {C_yy_j.shape}, xgrid shape: {xgrid.shape}")
# %%
