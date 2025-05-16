# %%
"""T3 Script."""

# 5) Quick plot
import lhapdf
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as torch_func
from loguru import logger
from torch import nn
from torch.optim import Adam
from validphys.api import API
from validphys.fkparser import load_fktable
from validphys.loader import Loader

# %%
# ? Load Commondata
inp_p = {
    "dataset_input": {
        "dataset": "BCDMS_NC_NOTFIXED_P_EM-F2",
        "variant": "legacy",
    },
    "use_cuts": "internal",
    "theoryid": 200,
}

inp_d = {
    "dataset_input": {
        "dataset": "BCDMS_NC_NOTFIXED_D_EM-F2",
        "variant": "legacy",
    },
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

df_p["idx_p"] = np.arange(len(df_p))  # 0 to 350
df_d["idx_d"] = np.arange(len(df_d))


# Match on x and Q2
merged_df = df_p.merge(df_d, on=["x", "Q2"], suffixes=("_p", "_d"))
merged_df["y"] = merged_df["F2_p"] - merged_df["F2_d"]


# %%
# ? Load FK Tables
loader = Loader()
fk_p = load_fktable(loader.check_fktable(setname="BCDMSP", theoryID=200, cfac=()))
fk_d = load_fktable(loader.check_fktable(setname="BCDMSD", theoryID=200, cfac=()))

# %%
# ? Extract the Non-Singlet FK Kernel
# Load the FK convolution kernels
wp = fk_p.get_np_fktable()  # shape (351, 5, 50)
wd = fk_d.get_np_fktable()  # shape (254, 5, 50)

# Index of T₃ in the evolution basis
flavor_index = 2  # T3 = u⁺ - d⁺

# Extract the T3 kernels for each dataset
wp_t3 = wp[:, flavor_index, :]  # shape (351, 50)
wd_t3 = wd[:, flavor_index, :]  # shape (254, 50)

# Get the matching FK row indices from the merged dataframe
idx_p = merged_df["idx_p"].to_numpy()
idx_d = merged_df["idx_d"].to_numpy()

# Subtract the proton and deuteron kernels: W = FK_p - FK_d
W = wp_t3[idx_p] - wd_t3[idx_d]  # shape (N_matched, 50)
# %%
# ? Build the full covariance for p and d combined
params = {
    "dataset_inputs": [
        {"dataset": "BCDMS_NC_NOTFIXED_P_EM-F2", "variant": "legacy"},
        {"dataset": "BCDMS_NC_NOTFIXED_D_EM-F2", "variant": "legacy"},
    ],
    "use_cuts": "internal",
    "theoryid": 200,
}
cov_full = API.dataset_inputs_covmat_from_systematics(**params)


# Now extract only the rows+cols matching our merged_df.
# The first block of cov_full corresponds to the proton points (0:len(df_p))
# and the second to deuteron (len(df_p):len(df_p)+len(df_d)).
n_p, n_d = len(df_p), len(df_d)
# build index list: [ idx_p[i] ] and [ n_p + idx_d[i] ] for each merged row
rows = np.concatenate([idx_p, n_p + idx_d])
# but we want only the difference Fp-Fd, so we need the covariance of y = Fp - Fd:
# Cov(y) = Cov(Fp,Fp)[idx_p,idx_p] + Cov(Fd,Fd)[idx_d,idx_d]
#           -2 Cov(Fp,Fd)[idx_p, n_p+idx_d]
C_pp = cov_full[:n_p, :n_p]
C_dd = cov_full[n_p:, n_p:]
C_pd = cov_full[:n_p, n_p:]
C_yy = C_pp[np.ix_(idx_p, idx_p)] + C_dd[np.ix_(idx_d, idx_d)] - 2 * C_pd[np.ix_(idx_p, idx_d)]

eps = 1e-6 * np.mean(np.diag(C_yy))
C_yy_j = C_yy + np.eye(C_yy.shape[0]) * eps
Cinv = np.linalg.inv(C_yy_j)

# %%
# ? Prepare Tensors
# your x-grid
xgrid = fk_p.xgrid  # length 50
x_torch = torch.tensor(xgrid, dtype=torch.float32).unsqueeze(1)

# data convolution matrix and target
W_torch = torch.tensor(W, dtype=torch.float32)  # (N_data, 50)
y_torch = torch.tensor(merged_df["y"].to_numpy(), dtype=torch.float32)  # (N_data,)

# inverse covariance
Cinv_torch = torch.tensor(Cinv, dtype=torch.float32)  # (N_data,N_data)

# %%
# ? Model Definition


# 1) Define the T3 network (same as in your earlier script)
class T3Net(nn.Module):
    """Our T3 Network."""

    def __init__(self, n_hidden: int, alpha: float, beta: float) -> None:
        """Init for our network."""
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

    def forward(self, x: float) -> tuple:
        """Forward Pass."""
        raw = self.net(x)  # (N_x,1)
        positive = torch_func.softplus(raw)  # ensure ≥0
        pre = x.pow(1 - self.alpha) * (1 - x).pow(self.beta)
        return self.A * pre * positive  # (N_x,1)


# 2) χ² loss function
def chi2(model) -> float:  # noqa: ANN001
    """Calculate our Chi Squared."""
    f = model(x_torch).squeeze()  # (N_x,)
    y_pred = W_torch @ f  # (N_data,)
    resid = y_pred - y_torch  # (N_data,)
    return resid @ (Cinv_torch @ resid)  # scalar


# %%
# ? Training
# 3) Instantiate & train
model = T3Net(n_hidden=30, alpha=1.0, beta=3.0)
opt = Adam(model.parameters(), lr=1e-3)

best = float("inf")
patience, wait = 10, 0

for epoch in range(1, 501):
    model.train()
    opt.zero_grad()
    loss = chi2(model)
    loss.backward()
    opt.step()

    val = loss.item()
    if val < best:
        best, wait = val, 0
        torch.save(model.state_dict(), "t3_best.pt")
    else:
        wait += 1
        if wait >= patience:
            logger.info(f"Early stopping at epoch {epoch}")
            break

    if epoch % 50 == 0:
        logger.info(f"Epoch {epoch}: χ² = {val:.2f}")

# %%
# ? Compare to reference PDF from LHAPDF

model.load_state_dict(torch.load("t3_best.pt"))
model.eval()
with torch.no_grad():
    T3_fit = model(x_torch).squeeze().numpy()

pdfset = lhapdf.getPDFSet("NNPDF40_nnlo_as_01180")
pdf0 = pdfset.mkPDF(0)  # central replica

# 2) Pick a reference scale (must match your FK theoryID setup)
Qref = fk_p.Q0  # GeV

# 3) Compute T3_ref on your xgrid
T3_ref = []
for x in xgrid:
    # xfxQ(flavor_id, x, Q) returns x*f(x)
    u = pdf0.xfxQ(2, x, Qref) / x  # up + anti-up
    ub = pdf0.xfxQ(-2, x, Qref) / x
    d = pdf0.xfxQ(1, x, Qref) / x  # down + anti-down
    db = pdf0.xfxQ(-1, x, Qref) / x
    T3_ref.append((u + ub) - (d + db))
T3_ref = np.array(T3_ref)


# %%
# ? Fit Plots
plt.plot(xgrid, T3_ref, label="NNPDF4.0")
plt.plot(xgrid, T3_fit, label="NN fit")
plt.xlabel("x")
plt.ylabel("T₃(x)")
plt.legend()
plt.show()


# %%
# 1) Data sanity: kinematic coverage & y‐distribution
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
# (a) x vs Q2
sc = ax1.scatter(
    merged_df["x"],
    merged_df["Q2"],
    c=merged_df["y"],
    cmap="coolwarm",
    s=20,
    alpha=0.8,
)
ax1.set(xscale="log", yscale="log", xlabel="x", ylabel="Q²", title="Data kinematics (F₂ₚ−F₂𝚍)")
plt.colorbar(sc, ax=ax1, label="y = F₂ₚ−F₂𝚍")

# (b) Histogram of y
ax2.hist(merged_df["y"], bins=30, alpha=0.7)
ax2.set(xlabel="y = F₂ₚ−F₂𝚍", ylabel="count", title="y distribution")
plt.tight_layout()
plt.show()

# %%
# 2) FK tables: heatmaps + coverage


xgrid = fk_p.xgrid

# (a) heatmaps of wp_t3, wd_t3, W
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for ax, arr, title in zip(
    axes,
    [wp_t3, wd_t3, W],
    ["proton T₃ kernels", "deuteron T₃ kernels", "difference W"],
):
    im = ax.imshow(
        arr,
        aspect="auto",
        origin="lower",
        extent=[xgrid[0], xgrid[-1], 0, arr.shape[0]],
    )
    ax.set(xlabel="x", title=title)
    fig.colorbar(im, ax=ax)
plt.tight_layout()
plt.show()

# (b) total coverage per x‐bin
coverage = np.count_nonzero(W, axis=0)
plt.figure(figsize=(6, 3))
plt.semilogx(xgrid, coverage, "-o")
plt.xlabel("x")
plt.ylabel("# data rows with W≠0")
plt.title("Sampling coverage of each x‐bin")
plt.show()
# %%
# 3) PDF sanity: direct LHAPDF x·f and bare f


pdfset = lhapdf.getPDFSet("NNPDF40_nnlo_as_01180")
pdf0 = pdfset.mkPDF(0)
Q0 = fk_p.Q0

x = xgrid
xf_up = np.array([pdf0.xfxQ(2, xi, Q0) for xi in x])
xf_dn = np.array([pdf0.xfxQ(1, xi, Q0) for xi in x])
# bare T3
T3_bare = (xf_up - xf_dn) / x
# x·T3
T3_xf = xf_up - xf_dn

plt.figure(figsize=(6, 4))
plt.loglog(x, T3_xf, label="x·T₃ from LHAPDF")
plt.loglog(x, T3_bare, "--", label="T₃ bare = (x·T₃)/x")
plt.xlabel("x")
plt.ylabel("T₃")
plt.legend()
plt.title("LHAPDF reference sanity")
plt.show()
# %%
# 4) FK‐PDF convolution check: does W @ [x·T₃_ref] roughly reproduce y?
y_ref = W @ T3_xf
y_data = merged_df["y"].to_numpy()

plt.figure(figsize=(5, 5))
plt.scatter(y_ref, y_data, alpha=0.6, s=20)
lims = [min(y_ref.min(), y_data.min()), max(y_ref.max(), y_data.max())]
plt.plot(lims, lims, "k--")
plt.xlabel("W·[x·T₃_ref]")
plt.ylabel("y = data F₂ₚ−F₂𝚍")
plt.title("FK convolution of LHAPDF vs data")
plt.tight_layout()
plt.show()
# %%
