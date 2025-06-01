# %%
"""T3 Data Comparison Script (+ Pseudo-data).

Compares author-provided prepared data,
our own real-data-processed version,
AND a closure-test "pseudo-data" version where
we generate y_pseudo = W_our @ T3_ref_our + Gaussian noise.

Usage:
    python compare_t3_data.py
"""

import lhapdf
import matplotlib.pyplot as plt
import numpy as np
from validphys.api import API
from validphys.fkparser import load_fktable
from validphys.loader import Loader

# ------------- #
# 1. File Paths #
# ------------- #
auth_data_path = "data/prepared_data/"

# -------------- #
# 2. Load Both   #
# -------------- #
# %%
# -- Author's processed data --
y_auth = np.load(auth_data_path + "data.npy")
Cy_auth = np.load(auth_data_path + "Cy.npy")
kin_auth = np.load(auth_data_path + "kin.npy")
FK_auth = np.load(auth_data_path + "FK.npy")
xgrid_auth = np.load(auth_data_path + "fk_grid.npy")
NNPDF40_auth = np.load(auth_data_path + "NNPDF40.npy")

T3_ref_auth = NNPDF40_auth[6 * 50 : 7 * 50]
# %%
# -- Our data (from validphys + LHAPDF pipeline) --
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
merged_df["y"] = merged_df["F2_p"] - merged_df["F2_d"]


# %%
# FK tables
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
W_our = wp_t3[idx_p] - wd_t3[idx_d]
# %%
# Covariance
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

xgrid_our = fk_p.xgrid
y_our = merged_df["y"].to_numpy()
# %%
# -------------- #
# 3. Pseudo-data #
# -------------- #
# Closure test: generate pseudo-y as FK @ T3_ref (with noise)
pdfset = lhapdf.getPDFSet("NNPDF40_nnlo_as_01180")
pdf0 = pdfset.mkPDF(0)
Qref = fk_p.Q0

# Our T3_ref (must NOT divide by x! NNPDF prepared_data is x*f(x))
T3_ref_our = []
for x in xgrid_our:
    u = pdf0.xfxQ(2, x, Qref)
    ub = pdf0.xfxQ(-2, x, Qref)
    d = pdf0.xfxQ(1, x, Qref)
    db = pdf0.xfxQ(-1, x, Qref)
    T3_ref_our.append((u + ub) - (d + db))
T3_ref_our = np.array(T3_ref_our)

# Generate closure-test pseudo-data (same shape as y_our)
np.random.seed(42)  # For reproducibility
y_pseudo_mean = W_our @ T3_ref_our  # "theory" prediction
y_pseudo = np.random.multivariate_normal(y_pseudo_mean, C_yy_j)
# %%
# ------------- #
# 4. Comparison #
# ------------- #

# == (1) Kinematic coverage ==
fig, axs = plt.subplots(1, 3, figsize=(14, 4), sharex=True)
axs[0].scatter(kin_auth[:, 0], kin_auth[:, 1], c=y_auth, cmap="coolwarm", s=15)
axs[0].set(xscale="log", yscale="log", xlabel="x", ylabel="Q²", title="Author: kinematics")
axs[1].scatter(merged_df["x"], merged_df["Q2"], c=y_our, cmap="coolwarm", s=15)
axs[1].set(xscale="log", yscale="log", xlabel="x", ylabel="Q²", title="Ours: kinematics")
sc = axs[2].scatter(merged_df["x"], merged_df["Q2"], c=y_pseudo, cmap="coolwarm", s=15)
axs[2].set(xscale="log", yscale="log", xlabel="x", ylabel="Q²", title="Pseudo: kinematics")
plt.colorbar(sc, ax=axs[2], label="F2p-F2d or pseudo-y")
plt.tight_layout()
plt.show()

# == (2) y vector ==
plt.figure(figsize=(8, 4))
plt.plot(y_auth, ".", label="Author", alpha=0.8)
plt.plot(y_our, ".", label="Ours (real data)", alpha=0.8)
plt.plot(y_pseudo, ".", label="Pseudo-data", alpha=0.7)
plt.xlabel("Matched Data Index")
plt.ylabel("F2p-F2d or pseudo-y")
plt.title("Fp-Fd: Author vs Ours vs Pseudo")
plt.legend()
plt.show()
# %%
# == (3) Covariance: diagonal and full matrix ==
plt.figure(figsize=(8, 4))
plt.plot(np.diag(Cy_auth), label="Author diag(C)")
plt.plot(np.diag(C_yy_j), label="Ours diag(C)")
plt.xlabel("Index")
plt.ylabel("Variance")
plt.title("Covariance Diagonal: Author vs Ours")
plt.legend()
plt.show()

plt.figure(figsize=(14, 5))
plt.subplot(1, 2, 1)
plt.imshow(Cy_auth, aspect="auto", origin="lower")
plt.title("Author: Covariance")
plt.colorbar()
plt.subplot(1, 2, 2)
plt.imshow(C_yy_j, aspect="auto", origin="lower")
plt.title("Ours: Covariance")
plt.colorbar()
plt.suptitle("Covariance Matrices (full)")
plt.tight_layout()
plt.show()
# %%
# == (4) FK Table comparison ==
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.imshow(FK_auth, aspect="auto", origin="lower")
plt.title("Author FK (248 x 50)")
plt.colorbar()
plt.subplot(1, 2, 2)
plt.imshow(W_our, aspect="auto", origin="lower")
plt.title("Ours W = FKT3p - FKT3d (N x 50)")
plt.colorbar()
plt.suptitle("FK Table Comparison")
plt.tight_layout()
plt.show()
# %%
# == Per-row correlation ==
min_rows = min(FK_auth.shape[0], W_our.shape[0])
corrs = [np.corrcoef(FK_auth[i], W_our[i])[0, 1] for i in range(min_rows)]
plt.figure()
plt.plot(corrs)
plt.xlabel("Matched Data Index")
plt.ylabel("Correlation (Author FK vs Our W)")
plt.title("Per-row correlation (should be ~1)")
plt.show()

# == (5) x-grid ==
plt.figure()
plt.plot(xgrid_auth, label="Author xgrid")
plt.plot(xgrid_our, "--", label="Our xgrid")
plt.xlabel("x-grid index")
plt.ylabel("x")
plt.title("x-grid: Author vs Ours")
plt.legend()
plt.show()

# == (6) NNPDF T3 reference ==
plt.figure()
plt.plot(xgrid_auth, T3_ref_auth, label="Author T3_ref (NNPDF4.0)")
plt.plot(xgrid_our, T3_ref_our, "--", label="Ours T3_ref (LHAPDF)")
plt.xlabel("x")
plt.ylabel("T₃(x)")
plt.title("NNPDF T₃: Author vs Ours")
plt.legend()
plt.show()
# %%
# == (7) FK @ T3_ref convolution ==

# (a) Author: y_pred = FK @ T3_ref_auth
y_pred_auth = FK_auth @ T3_ref_auth
plt.figure()
plt.scatter(y_pred_auth, y_auth, s=18, alpha=0.7, label="Author: y_pred vs y")
plt.plot([y_auth.min(), y_auth.max()], [y_auth.min(), y_auth.max()], "k--", alpha=0.5)
plt.xlabel("y_pred (FK·T3_ref)")
plt.ylabel("y (data)")
plt.title("Author: FK convolution")
plt.legend()
plt.show()
# (b) Ours: y_pred = W_our @ T3_ref_our
y_pred_our = W_our @ T3_ref_our
plt.figure()
plt.scatter(y_pred_our, y_our, s=18, alpha=0.7, label="Ours: y_pred vs y")
plt.plot([y_our.min(), y_our.max()], [y_our.min(), y_our.max()], "k--", alpha=0.5)
plt.xlabel("y_pred (W·T3_ref)")
plt.ylabel("y (data)")
plt.title("Ours: FK convolution")
plt.legend()
plt.show()
# (c) Pseudo: y_pred = W_our @ T3_ref_our vs y_pseudo
plt.figure()
plt.scatter(y_pred_our, y_pseudo, s=18, alpha=0.7, label="Pseudo: y_pred vs y_pseudo")
plt.plot([y_pseudo.min(), y_pseudo.max()], [y_pseudo.min(), y_pseudo.max()], "k--", alpha=0.5)
plt.xlabel("y_pred (W·T3_ref)")
plt.ylabel("y_pseudo (pseudo-data)")
plt.title("Pseudo: FK convolution")
plt.legend()
plt.show()

# %%
