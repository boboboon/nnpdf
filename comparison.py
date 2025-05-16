# %%
"""T3 Data Comparison Script.

Compares author-provided prepared data (from Gaussian process script)
to data generated via our validphys + LHAPDF pipeline.

- Loads both sets of data and kernels
- Plots: kinematics, covariance, FK tables, T3 values, reference PDFs, convolution output
- Makes it easy to spot any processing problems or mismatches

Usage:
    python compare_t3_data.py

"""

import lhapdf
import matplotlib.pyplot as plt
import numpy as np
from loguru import logger
from validphys.api import API
from validphys.fkparser import load_fktable
from validphys.loader import Loader

# ------------- #
# 1. File Paths #
# ------------- #

# Author's data
auth_data_path = "data/prepared_data/"

# -------------- #
# 2. Load Both   #
# -------------- #

# -- Author's processed data --
y_auth = np.load(auth_data_path + "data.npy")  # (248,)
Cy_auth = np.load(auth_data_path + "Cy.npy")  # (248, 248)
kin_auth = np.load(auth_data_path + "kin.npy")  # (248, 2)
FK_auth = np.load(auth_data_path + "FK.npy")  # (248, 50)
xgrid_auth = np.load(auth_data_path + "fk_grid.npy")  # (50,)
NNPDF40_auth = np.load(auth_data_path + "NNPDF40.npy")  # (nreplicas*50,)

# For reference: slice out T3_ref from central replica in NNPDF40
T3_ref_auth = NNPDF40_auth[6 * 50 : 7 * 50]

# -- Our data (regenerate as in your pipeline for exact match) --
# 1) Load BCDMS F2_p, F2_d and their covariance, and process as before:

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
df_p["idx_p"] = np.arange(len(df_p))
df_d["idx_d"] = np.arange(len(df_d))

# Match on x, Q2
merged_df = df_p.merge(df_d, on=["x", "Q2"], suffixes=("_p", "_d"))
merged_df["y"] = merged_df["F2_p"] - merged_df["F2_d"]

# -- Our FK tables --
loader = Loader()
fk_p = load_fktable(loader.check_fktable(setname="BCDMSP", theoryID=200, cfac=()))
fk_d = load_fktable(loader.check_fktable(setname="BCDMSD", theoryID=200, cfac=()))
wp = fk_p.get_np_fktable()  # (351, 5, 50)
wd = fk_d.get_np_fktable()  # (254, 5, 50)
flavor_index = 2  # T3 = u^+ - d^+
wp_t3 = wp[:, flavor_index, :]  # (351, 50)
wd_t3 = wd[:, flavor_index, :]  # (254, 50)
idx_p = merged_df["idx_p"].to_numpy()
idx_d = merged_df["idx_d"].to_numpy()
W_our = wp_t3[idx_p] - wd_t3[idx_d]  # (N, 50)

# -- Our Covariance (only for matched rows) --
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

# -- Our x-grid --
xgrid_our = fk_p.xgrid  # (50,)

# -- Our y (Fp-Fd at matched rows) --
y_our = merged_df["y"].to_numpy()

# ------------- #
# 3. Comparison #
# ------------- #

# == (1) Kinematic coverage ==
fig, axs = plt.subplots(1, 2, figsize=(11, 4), sharex=True)
axs[0].scatter(kin_auth[:, 0], kin_auth[:, 1], c=y_auth, cmap="coolwarm", s=15)
axs[0].set(xscale="log", yscale="log", xlabel="x", ylabel="Q²", title="Author: kinematics")
sc = axs[1].scatter(merged_df["x"], merged_df["Q2"], c=y_our, cmap="coolwarm", s=15)
axs[1].set(xscale="log", yscale="log", xlabel="x", ylabel="Q²", title="Our: kinematics")
plt.colorbar(sc, ax=axs[1], label="F₂p−F₂d")
plt.tight_layout()
plt.show()

# == (2) y vector (Fp-Fd) ==
plt.figure(figsize=(7, 4))
plt.plot(y_auth, ".", label="Author", alpha=0.8)
plt.plot(y_our, ".", label="Ours", alpha=0.8)
plt.xlabel("Matched Data Index")
plt.ylabel("F₂ₚ−F₂𝚍")
plt.title("Fp-Fd: Author vs Ours")
plt.legend()
plt.show()

# == (3) Covariance: diagonal and full matrix ==
plt.figure(figsize=(7, 4))
plt.plot(np.diag(Cy_auth), label="Author diag(C)")
plt.plot(np.diag(C_yy_j), label="Ours diag(C)")
plt.xlabel("Index")
plt.ylabel("Variance")
plt.title("Covariance Diagonal: Author vs Ours")
plt.legend()
plt.show()

plt.figure(figsize=(12, 5))
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

# == (4) FK Table comparison (heatmap and per-row correlation) ==
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

# == Per-row correlation (dot product, for sanity) ==
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
T3_ref_our = []
pdfset = lhapdf.getPDFSet("NNPDF40_nnlo_as_01180")
pdf0 = pdfset.mkPDF(0)
Qref = fk_p.Q0
for x in xgrid_our:
    u = pdf0.xfxQ(2, x, Qref)
    ub = pdf0.xfxQ(-2, x, Qref)
    d = pdf0.xfxQ(1, x, Qref)
    db = pdf0.xfxQ(-1, x, Qref)
    T3_ref_our.append((u + ub) - (d + db))
T3_ref_our = np.array(T3_ref_our)

plt.figure()
plt.plot(xgrid_auth, T3_ref_auth, label="Author T3_ref (NNPDF4.0)")
plt.plot(xgrid_our, T3_ref_our, "--", label="Ours T3_ref (LHAPDF)")
plt.xlabel("x")
plt.ylabel("T₃(x)")
plt.title("NNPDF T₃: Author vs Ours")
plt.legend()
plt.show()

# == (7) FK @ T3_ref convolution reproduces y? ==
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

# == (8) Print some summary stats for sanity ==
logger.info(f"Author: y shape {y_auth.shape}, Ours: {y_our.shape}")
logger.info(
    f"Author: Cov diag mean {np.mean(np.diag(Cy_auth)):.2e}, Ours: {np.mean(np.diag(C_yy_j)):.2e}",
)
logger.info(f"Author: FK shape {FK_auth.shape}, Ours: {W_our.shape}")
logger.info(
    f"Author: xgrid min/max: {xgrid_auth.min():.2e}/{xgrid_auth.max():.2e}, Ours: {xgrid_our.min():.2e}/{xgrid_our.max():.2e}",
)

# %%
