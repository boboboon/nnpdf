# %%
"""T3 Script (Closure/Pseudo-data version, clean and documented).

Fits a neural-network PDF to pseudo-data generated from
W @ T3_ref (NNPDF4.0 central) plus correlated Gaussian noise.

"""

import lhapdf
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as torch_func
from loguru import logger
from matplotlib.lines import Line2D
from torch import nn
from torch.optim import Adam
from validphys.api import API
from validphys.fkparser import load_fktable
from validphys.loader import Loader

# --- Set matplotlib to use LaTeX for all text
plt.rc("text", usetex=True)
plt.rc("font", family="serif")

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
rng = np.random.default_rng(42)
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
plt.tight_layout()
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
ax.legend(handles=[sc_pseudo, proxy_real, Line2D([], [], linestyle="--", color="k")])

proxy_ideal = Line2D([], [], linestyle="--", color="k", label="Ideal")
ax.legend(handles=[sc_pseudo, proxy_real, proxy_ideal])


ax.set_xlabel(r"$y_{\rm ref}=W\cdot xT_3$")
ax.set_ylabel(r"$y$")
ax.set_title("FK convolution: pseudo vs real data (coloured by $W^2$)")
plt.tight_layout()
plt.show()


# %% ---- Model Definition ----
class T3Net(nn.Module):
    """Neural network for non-singlet PDF, outputs x*T3(x)."""

    def __init__(self, n_hidden: int, alpha: float, beta: float) -> None:
        """Init."""
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
        """Forward Pass."""
        raw = self.net(x)
        pos = torch_func.softplus(raw)
        pre = x.pow(1 - self.alpha) * (1 - x).pow(self.beta)
        return self.A * pre * pos


def chi2(model: nn.Module) -> torch.Tensor:
    """Loss."""
    f_pred = model(x_torch).squeeze()
    y_pred = W_torch @ f_pred
    resid = y_pred - y_torch
    return (resid @ (Cinv_torch @ resid)) / resid.size(0)


# %% ---- Training Loop ----
model = T3Net(n_hidden=30, alpha=1.0, beta=3.0)
optimizer = Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
best_loss = float("inf")
patience, wait = 10, 0
loss_hist, chi2_hist = [], []

for epoch in range(1, 501):
    model.train()
    optimizer.zero_grad()
    loss = chi2(model)
    loss.backward()
    optimizer.step()
    val = loss.item()
    loss_hist.append(val)
    chi2_hist.append(val / len(y_torch))

    # early stopping
    if val < best_loss:
        best_loss, wait = val, 0
        torch.save(model.state_dict(), "model_states/t3_best.pt")
    else:
        wait += 1
        if wait >= patience:
            logger.info(f"Early stopping at epoch {epoch}")
            break

    if epoch % 50 == 0:
        logger.info(rf"Epoch {epoch}: \chi^2 = {val:.2f}")

# %% ---- Loss Curves ----
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
axes[0].plot(loss_hist, label=r"Total $\chi^2$")
axes[0].set(xlabel="Epoch", ylabel=r"$\chi^2$", title="Training loss")
axes[0].legend()

axes[1].plot(chi2_hist, label=r"$\chi^2/N$")
axes[1].set(xlabel="Epoch", ylabel=r"$\chi^2/N$", title="Reduced chi2")
axes[1].legend()

plt.tight_layout()
plt.show()

# %% ---- Final PDF Comparison ----
model.load_state_dict(torch.load("model_states/t3_best.pt"))
model.eval()
with torch.no_grad():
    T3_fit = model(x_torch).squeeze().numpy()

plt.figure(figsize=(6, 4))
plt.plot(xgrid, T3_ref, label=r"NNPDF4.0 (truth, $xT_3$)")
plt.plot(xgrid, T3_fit, "--", label=r"NN fit")
plt.xlabel(r"$x$")
plt.ylabel(r"$x\,T_3(x)$")
plt.title(r"Closure test: fitted vs true $xT_3$")
plt.legend()
plt.tight_layout()
plt.show()


# %% replica loop
n_replicas = 20
fits = []  # will hold T3_fit for each replica
loss_per_replica = []  # optional: track final χ2 per replica

for i in range(n_replicas):
    # generate a new pseudo-dataset
    y_pseudo = rng.multivariate_normal(y_pseudo_mean, C_yy)
    y_torch = torch.tensor(y_pseudo, dtype=torch.float32)

    # re-init model & optimizer
    model = T3Net(n_hidden=30, alpha=1.0, beta=3.0)
    optimizer = Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    best_loss = float("inf")
    wait = 0

    # train on this replica
    for _ in range(1, 501):
        model.train()
        optimizer.zero_grad()
        loss = chi2(model)
        loss.backward()
        optimizer.step()

        val = loss.item()
        if val < best_loss:
            best_loss, wait = val, 0
            torch.save(model.state_dict(), f"model_states/t3_replica{i}.pt")
        else:
            wait += 1
            if wait >= patience:
                break

    loss_per_replica.append(best_loss)

    # load best and eval
    model.load_state_dict(torch.load(f"model_states/t3_replica{i}.pt"))
    model.eval()
    with torch.no_grad():
        T3_fit = model(x_torch).squeeze().cpu().numpy()
    fits.append(T3_fit)

fits = np.array(fits)  # shape (n_replicas, N_grid)
# %%
# compute mean & std, and plot
mean_fit = np.mean(fits, axis=0)
std_fit = np.std(fits, axis=0)

plt.figure(figsize=(6, 4))
# uncertainty band
plt.fill_between(
    xgrid,
    mean_fit - std_fit,
    mean_fit + std_fit,
    color="C1",
    alpha=0.3,
    label=r"NN fit ±1$\sigma$ (toy)",
)
# central fit
plt.plot(xgrid, mean_fit, "-", label="NN fit mean")
# truth
plt.plot(xgrid, T3_ref, "--", label="NNPDF4.0 (truth)")
plt.xlabel(r"$x$")
plt.ylabel(r"$x\,T_3(x)$")
plt.title(r"Closure-test ensemble: mean ± $\sigma$")
plt.legend()
plt.tight_layout()
plt.show()


# %%
