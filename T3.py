# %%
"""T3 Script."""

from collections import Counter

import lhapdf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from validphys.api import API
from validphys.fkparser import load_fktable
from validphys.loader import Loader

# 1) Initialize loader and load commondata for deuteron and proton
loader = Loader()
common_d = API.commondata(
    dataset_input={"dataset": "BCDMS_NC_NOTFIXED_D_EM-F2", "cfac": (), "variant": "legacy"},
    theory_id=200,
).load()
common_p = API.commondata(
    dataset_input={"dataset": "BCDMS_NC_NOTFIXED_P_EM-F2", "cfac": (), "variant": "legacy"},
    theory_id=200,
).load()


# 2) Rename kinematics + central & stat errors, disambiguate ADD/MULT
def prep_df(common, label):
    df = (
        common.commondata_table.rename(
            columns={f"kin{i + 1}": name for i, name in enumerate(common.kin_variables)},
        )
        .rename(columns={"data": f"F2_{label}", "stat": f"stat_{label}"})
        .reset_index(drop=True)
    )
    cnt = Counter()
    cols = []
    for c in df.columns:
        if c == "ADD":
            cnt["ADD"] += 1
            cols.append(f"ADD_{cnt['ADD']}_{label}")
        elif c == "MULT":
            cnt["MULT"] += 1
            cols.append(f"MULT_{cnt['MULT']}_{label}")
        else:
            cols.append(c)
    df.columns = cols
    return df


df_d = prep_df(common_d, "d")
df_p = prep_df(common_p, "p")

# 3) Merge experiment on (x,Q2)
exp_df = pd.merge(
    df_p.drop(columns=["process", "y"], errors="ignore"),
    df_d.drop(columns=["process", "y"], errors="ignore"),
    on=["x", "Q2"],
    how="inner",
)

# 4) Build experimental observable y and covariance Cy
F2_p = exp_df["F2_p"].to_numpy()
F2_d = exp_df["F2_d"].to_numpy()
stat_p = exp_df["stat_p"].to_numpy()
stat_d = exp_df["stat_d"].to_numpy()
N = len(exp_df)

add_p = sorted(c for c in exp_df if c.startswith("ADD_") and c.endswith("_p"))
mult_p = sorted(c for c in exp_df if c.startswith("MULT_") and c.endswith("_p"))
add_d = sorted(c for c in exp_df if c.startswith("ADD_") and c.endswith("_d"))
mult_d = sorted(c for c in exp_df if c.startswith("MULT_") and c.endswith("_d"))

# proton block
C_pp = np.diag(stat_p**2)
for c in add_p:
    C_pp += np.outer(exp_df[c], exp_df[c])
for c in mult_p:
    v = exp_df[c].to_numpy() * F2_p
    C_pp += np.outer(v, v)

# deuteron block
C_dd = np.diag(stat_d**2)
for c in add_d:
    C_dd += np.outer(exp_df[c], exp_df[c])
for c in mult_d:
    v = exp_df[c].to_numpy() * F2_d
    C_dd += np.outer(v, v)

# cross‐covariance
C_pd = np.zeros((N, N))
for cp, cd in zip(add_p, add_d):
    C_pd += np.outer(exp_df[cp], exp_df[cd])
for cp, cd in zip(mult_p, mult_d):
    C_pd += np.outer(exp_df[cp].to_numpy() * F2_p, exp_df[cd].to_numpy() * F2_d)

y = F2_p - F2_d
Cy = C_pp + C_dd - 2 * C_pd


# %%
# 5) Load raw FK tables (no cuts)
fk_p = load_fktable(loader.check_fktable(setname="BCDMSP", theoryID=200, cfac=()))
fk_d = load_fktable(loader.check_fktable(setname="BCDMSD", theoryID=200, cfac=()))

# 6) Build pseudo‐data y0 from a known PDF f0 (NNPDF4.0 central)
pdf0 = lhapdf.mkPDF("NNPDF40_nlo_as_01180")
xgrid_p = fk_p.xgrid  # length Np
basis_p = fk_p.sigma.columns  # n_basis_p
xgrid_d = fk_d.xgrid  # length Nd
basis_d = fk_d.sigma.columns  # n_basis_d

# evaluate f₀ on each grid
f0_p = np.array(
    [[pdf0.xfxQ2(pid, float(x), float(fk_p.Q0)) / x for pid in basis_p] for x in xgrid_p],
)
f0_d = np.array(
    [[pdf0.xfxQ2(pid, float(x), float(fk_d.Q0)) / x for pid in basis_d] for x in xgrid_d],
)

# dense FK-tensors
ft_p = fk_p.get_np_fktable()  # (Np, n_basis_p, nx)
ft_d = fk_d.get_np_fktable()  # (Nd, n_basis_d, nx)

# convolution
F2_p0_all = np.tensordot(ft_p, f0_p, axes=([1, 2], [1, 0]))  # (Np,)
F2_d0_all = np.tensordot(ft_d, f0_d, axes=([1, 2], [1, 0]))  # (Nd,)

# 7) Match on (x,Q2) exactly as for the data
pred_p = pd.DataFrame({"x": df_p["x"], "Q2": df_p["Q2"], "F2_p0": F2_p0_all})
pred_d = pd.DataFrame({"x": df_d["x"], "Q2": df_d["Q2"], "F2_d0": F2_d0_all})

matched = pred_p.merge(pred_d, on=["x", "Q2"], how="inner").merge(
    exp_df[["x", "Q2"]].drop_duplicates(),
    on=["x", "Q2"],
    how="inner",
)

y0 = (matched["F2_p0"] - matched["F2_d0"]).to_numpy()
# %%
# ─── Cell 1: Kinematic coverage of the experimental data ───


plt.figure()
plt.scatter(exp_df["x"], exp_df["Q2"], marker="o")
plt.xscale("log")
plt.yscale("log")
plt.xlabel("x")
plt.ylabel("Q² [GeV²]")
plt.title("Kinematic Coverage of BCDMS $F_2^p - F_2^d$")
plt.grid(visible=True)
# %%
# ─── Cell 2: Experimental y = F₂ᵖ–F₂ᵈ with total uncertainties ───


err = np.sqrt(np.diag(Cy))

plt.figure()
plt.errorbar(exp_df["x"], y, yerr=err, fmt="o")
plt.xscale("log")
plt.xlabel("x")
plt.ylabel("y = $F_2^p - F_2^d$")
plt.title("Experimental Observable with Uncertainties")
plt.grid(visible=True)

# ─── Cell 3: Pseudo-data y₀ vs x ───
plt.figure()
plt.scatter(matched["x"], y0, marker="x")
plt.xscale("log")
plt.xlabel("x")
plt.ylabel("y₀ = Theory($F_2^p-F_2^d$)")
plt.title("Pseudo-data from NNPDF4.0 Central")
plt.grid(visible=True)

# ─── Cell 4: Overlay experiment and pseudo-data ───
plt.figure()
plt.errorbar(exp_df["x"], y, yerr=err, fmt="o", label="Experimental")
plt.scatter(matched["x"], y0, marker="x", label="Pseudo-data")
plt.xscale("log")
plt.xlabel("x")
plt.ylabel("y")
plt.title("Comparison: Experimental vs Pseudo-Data")
plt.legend()
plt.grid(visible=True)
# %%
# ─── Cell 5: Underlying PDF f₀ at Q₀ ───
# Plot e.g. gluon, up and down at the FK scale Q₀
xvals = np.logspace(-3, 0, 100)
Q0 = fk_p.Q0

g = [pdf0.xfxQ2(21, x, Q0) / x for x in xvals]
u = [pdf0.xfxQ2(2, x, Q0) / x for x in xvals]
d = [pdf0.xfxQ2(1, x, Q0) / x for x in xvals]

plt.figure()
plt.plot(xvals, g)
plt.xscale("log")
plt.yscale("log")
plt.xlabel("x")
plt.ylabel("x·g(x, Q₀)")
plt.title("Gluon PDF at $Q_0$")
plt.grid(visible=True)

plt.figure()
plt.plot(xvals, u, label="u")
plt.plot(xvals, d, label="d")
plt.xscale("log")
plt.yscale("log")
plt.xlabel("x")
plt.ylabel("x·q(x, Q₀)")
plt.title("Up & Down PDFs at $Q_0$")
plt.legend()
plt.grid(visible=True)
# %%
