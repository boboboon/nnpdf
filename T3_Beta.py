# %%
"""T3_BSM_Comparison."""

# %%
# --- Imports & Setup ---
from __future__ import annotations

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

# Device for PyTorch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Create directories for outputs
model_state_dir = Path("model_states")
model_state_dir.mkdir(parents=True, exist_ok=True)
results_dir = Path("results")
results_dir.mkdir(parents=True, exist_ok=True)

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

df_p = lcd_p.commondata_table.rename(
    columns={"kin1": "x", "kin2": "q2", "kin3": "y", "data": "F2_p", "stat": "error"},
)
df_d = lcd_d.commondata_table.rename(
    columns={"kin1": "x", "kin2": "q2", "kin3": "y", "data": "F2_d", "stat": "error"},
)

# Add unique indices for merging
df_p["idx_p"] = np.arange(len(df_p))
df_d["idx_d"] = np.arange(len(df_d))

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
logger.info("Building FK tables and computing convolution matrix W for T3 channel...")

t3_index = 2  # flavor index in FK table
loader = Loader()
fk_p = load_fktable(loader.check_fktable(setname="BCDMSP", theoryID=208, cfac=()))
fk_d = load_fktable(loader.check_fktable(setname="BCDMSD", theoryID=208, cfac=()))

wp = fk_p.get_np_fktable()  # shape (n_data_fk, n_flav, n_grid)
wd = fk_d.get_np_fktable()
wp_t3 = wp[:, t3_index, :]
wd_t3 = wd[:, t3_index, :]

idx_p = merged_df["idx_p"].to_numpy()
idx_d = merged_df["idx_d"].to_numpy()
W = wp_t3[idx_p] - wd_t3[idx_d]  # shape (n_data, n_grid)

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
n_p = len(df_p)
c_pp = cov_full[:n_p, :n_p]
c_dd = cov_full[n_p:, n_p:]
c_pd = cov_full[:n_p, n_p:]
c_yy = c_pp[np.ix_(idx_p, idx_p)] + c_dd[np.ix_(idx_d, idx_d)] - 2 * c_pd[np.ix_(idx_p, idx_d)]
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
# 5. DATA LOADING & PREPROCESSING—PART 5: COMPUTE T3_REF_NORM FOR CLOSURE
# ------------------------------------------------------------------------------
logger.info("Computing reference T3 (t3_ref_norm) for closure test...")

pdfset = lhapdf.getPDFSet("NNPDF40_nnlo_as_01180")
pdf0 = pdfset.mkPDF(0)
q_ref = fk_p.Q0

t3_ref_list = []
for x in xgrid:
    u, ub = pdf0.xfxQ(2, x, q_ref), pdf0.xfxQ(-2, x, q_ref)
    d, db = pdf0.xfxQ(1, x, q_ref), pdf0.xfxQ(-1, x, q_ref)
    t3_ref_list.append((u + ub) - (d + db))
t3_ref = np.array(t3_ref_list)  # this is x * t3_true(x)
t3_val_ref = t3_ref / xgrid
integral_truth = np.trapz(t3_val_ref, xgrid)  # noqa: NPY201
t3_ref_norm = t3_ref * (1.0 / integral_truth)  # ∫ t3 dx = 1

# Compute y_pseudo_mean (closure) = W @ t3_ref_norm
y_pseudo_mean = W.dot(t3_ref_norm)  # shape (n_data,)


# %%
# 6. BSM SHAPE (K) COMPUTATION
# ------------------------------------------------------------------------------
def compute_k_quadratic(q2_vals: np.ndarray, scale: float) -> np.ndarray:
    """Compute raw quadratic K(q2) = ((q2 - q2_min)^2) / scale^2."""
    q2_min = q2_vals.min()
    return ((q2_vals - q2_min) ** 2) / (scale**2)


def compute_k_linear(q2_vals: np.ndarray) -> np.ndarray:
    """Compute raw linear K(q2) = (q2 - q2_min) / (q2_max - q2_min)."""
    q2_min = q2_vals.min()
    q2_max = q2_vals.max()
    eps = 1e-8
    return (q2_vals - q2_min) / (q2_max - q2_min + eps)


def compute_k_contact(q2_vals: np.ndarray, M: float) -> np.ndarray:
    """Compute raw contact-interaction K(q2) = q2 / M^2."""
    return q2_vals / (M**2)


# Map keys to functions for scalability
_k_func_map = {
    "quadratic": compute_k_quadratic,
    "linear": compute_k_linear,
    "contact": compute_k_contact,
}


def compute_k_vector(q2_vals: np.ndarray, key: str, **kwargs) -> np.ndarray:  # noqa: ANN003
    """Generate a standardized BSM-shape vector K(q2) based on key.

    Supported keys: "quadratic", "linear", "contact". Returns zero-mean, unit-std array.
    """
    try:
        func = _k_func_map[key]
    except KeyError:
        msg = f"Unsupported key: {key}"
        raise ValueError(msg)  # noqa: B904

    raw = func(q2_vals, **kwargs)
    mean_raw = raw.mean()
    std_raw = raw.std()
    if std_raw < 1e-12:
        msg = "Standard deviation of raw K is too small"
        raise RuntimeError(msg)

    return (raw - mean_raw) / std_raw


# %%
# 7. NEURAL NETWORK MODEL DEFINITION
# ------------------------------------------------------------------------------
class T3Net(nn.Module):
    """Neural network for non-singlet PDF t3(x) with preprocessing x^alpha (1-x)^beta.

    If use_bsm=True, includes a trainable scalar `c`.
    """

    def __init__(  # noqa: PLR0913
        self,
        n_hidden: int,
        n_layers: int = 3,
        init_alpha: float = 1.0,
        init_beta: float = 3.0,
        dropout: float = 0.2,
        use_bsm: bool = False,  # noqa: FBT001, FBT002
    ) -> None:
        """Create our T3 Model."""
        super().__init__()
        # Log-parameterization for alpha, beta (ensure > 0)
        self.logalpha = nn.Parameter(torch.log(torch.tensor(init_alpha)))
        self.logbeta = nn.Parameter(torch.log(torch.tensor(init_beta)))

        # MLP: [Linear -> Tanh -> BatchNorm] x n_layers, ending in Linear
        layers = [nn.Linear(1, n_hidden), nn.Tanh(), nn.BatchNorm1d(n_hidden)]
        for _ in range(n_layers - 1):
            layers += [
                nn.Linear(n_hidden, n_hidden),
                nn.BatchNorm1d(n_hidden),
                nn.Tanh(),
                nn.Dropout(dropout),
            ]
        layers.append(nn.Linear(n_hidden, 1))  # raw scalar output
        self.net = nn.Sequential(*layers)

        self.use_bsm = use_bsm
        if self.use_bsm:
            self.c = nn.Parameter(torch.tensor(0.0))  # BSM coefficient

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: returns x * t3_unc(x) ≥ 0."""
        raw = self.net(x)  # (n_grid, 1)
        pos = torch_func.softplus(raw).squeeze()  # ensure non-negative

        alpha = torch.exp(self.logalpha).clamp(min=1e-3)
        beta = torch.exp(self.logbeta).clamp(min=1e-3)
        x_ = x.squeeze().clamp(min=1e-6, max=1.0 - 1e-6)

        pre = x_.pow(alpha) * (1.0 - x_).pow(beta)  # (n_grid,)
        return pre * pos  # x * t3_unc(x)


# %%
# 8. HELPERS FOR REPLICA-FITTING
# ------------------------------------------------------------------------------
def generate_pseudo_data(
    y_base_mean: np.ndarray,
    c_yy: np.ndarray,
    k_vector: np.ndarray,
    injection_c: float,
    i: int,
) -> np.ndarray:
    """Generate one pseudo-data vector for replica i.

    If injection_c != 0, apply y_base_mean * (1 + injection_c * k_vector).
    Otherwise use y_base_mean directly.
    """
    if injection_c != 0.0:
        y_mean_i = y_base_mean * (1.0 + injection_c * k_vector)
    else:
        y_mean_i = y_base_mean.copy()

    rng = np.random.default_rng(1000 + i)
    return rng.multivariate_normal(y_mean_i, c_yy)


def split_train_val(n_data: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Split data indices into train (90%) and val (10%) using given seed.

    Returns (train_idx, val_idx).
    """
    idx_all = np.arange(n_data)
    return train_test_split(idx_all, test_size=0.10, random_state=seed)


def build_subcov_inverse(
    c_yy: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Given full covariance c_yy and train/val indices, compute submatrices and their inverses.
    Returns (cinv_tr, cinv_val), both torch.Tensor.
    """  # noqa: D205
    c_tr = c_yy[np.ix_(train_idx, train_idx)]
    c_val = c_yy[np.ix_(val_idx, val_idx)]
    cinv_tr = torch.tensor(np.linalg.inv(c_tr), dtype=torch.float32, device=device)
    cinv_val = torch.tensor(np.linalg.inv(c_val), dtype=torch.float32, device=device)
    return cinv_tr, cinv_val


def train_one_replica(  # noqa: D417, PLR0913, PLR0915
    i: int,
    scenario_name: str,
    W: np.ndarray,
    c_yy: np.ndarray,
    xgrid: np.ndarray,
    weights: torch.Tensor,
    y_base_mean: np.ndarray,
    y_real: np.ndarray,
    fit_params: dict,
    use_bsm: bool,  # noqa: FBT001
    k_vector: np.ndarray,
    injection_c: float,
    use_real_data: bool,  # noqa: FBT001
) -> tuple[bool, dict, np.ndarray]:
    """Train a single replica i for the given scenario.

    Args:
        use_real_data: if True, draw pseudo-data from y_real instead of y_base_mean.

    Returns:
      - keep_flag: bool, True if 0.8 ≤ chi2/pt ≤ 1.2
      - record: dict of {replica, chi2, chi2_perpt, alpha, beta, c}
      - fitted_f_norm: np.ndarray of shape (n_grid,) if kept, else None
    """
    n_data = W.shape[0]
    n_val = int(0.10 * n_data)

    # 1) Generate pseudo-data
    base_mean = y_real if use_real_data else y_base_mean
    y_pseudo_i = generate_pseudo_data(base_mean, c_yy, k_vector, injection_c, i)
    y_torch = torch.tensor(y_pseudo_i, dtype=torch.float32, device=device)

    # 2) Split train/val
    train_idx, val_idx = split_train_val(n_data, seed=1000 + i)

    # 3) Build sub-cov inverses
    cinv_tr, cinv_val = build_subcov_inverse(c_yy, train_idx, val_idx)

    # 4) Initialize model & optimizer
    model = T3Net(
        n_hidden=fit_params["n_hidden"],
        n_layers=fit_params["n_layers"],
        dropout=fit_params["dropout"],
        use_bsm=use_bsm,
    ).to(device)

    if use_bsm:
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

    best_val = float("inf")
    wait = 0

    # Precompute W on GPU once
    w_torch = torch.tensor(W, dtype=torch.float32, device=device)
    x_torch = torch.tensor(xgrid, dtype=torch.float32).unsqueeze(1).to(device)

    # 5) Epoch loop
    for epoch in range(1, fit_params["num_epochs"] + 1):
        model.train()
        optimizer.zero_grad()

        # Forward: compute x * t3_unc(x)
        f_raw = model(x_torch).squeeze()  # (n_grid,)
        t3_unnorm = f_raw / x_torch.squeeze()  # t3_unc(x)
        integral_raw = torch.dot(weights, t3_unnorm)  # ∫ t3_unc dx
        norm_factor = 1.0 / integral_raw
        f_norm = norm_factor * f_raw  # normalized x*t3(x)

        # Convolution → t_sm_pred
        t_sm_pred = w_torch.matmul(f_norm)  # (n_data,)
        if use_bsm:
            k_torch_full = torch.tensor(k_vector, dtype=torch.float32, device=device)
            t_pred = t_sm_pred * (1.0 + model.c * k_torch_full)
        else:
            t_pred = t_sm_pred

        # Chi^2 on train subset
        resid_tr = t_pred[train_idx] - y_torch[train_idx]
        loss_chi2 = resid_tr @ (cinv_tr.matmul(resid_tr))

        # Curvature penalty on t3_unc
        dx_vals = (x_torch[1:] - x_torch[:-1]).squeeze()
        d2 = (t3_unnorm[:-2] - 2 * t3_unnorm[1:-1] + t3_unnorm[2:]) / (dx_vals[:-1] ** 2)
        loss_smooth = torch.sum(d2.pow(2))

        # Total loss
        loss_total = loss_chi2 + fit_params["lambda_smooth"] * loss_smooth
        loss_total.backward()
        optimizer.step()

        # Validation pass (chi^2 only)
        model.eval()
        with torch.no_grad():
            f_raw_val = model(x_torch).squeeze()
            t3_unnorm_val = f_raw_val / xgrid
            integral_val = torch.dot(weights, t3_unnorm_val.float())
            norm_val = 1.0 / integral_val
            f_val = norm_val * f_raw_val
            t_sm_val = w_torch[val_idx].matmul(f_val)
            t_pred_val = t_sm_val * (1.0 + model.c * k_torch_full[val_idx]) if use_bsm else t_sm_val
            resid_val = t_pred_val - y_torch[val_idx]
            loss_val = resid_val @ (cinv_val.matmul(resid_val))

        if epoch % 200 == 0:
            logger.info(f"Epoch:{epoch},chi2/n:{loss_val / n_val}")

        # Early-stopping
        if loss_val.item() < best_val:
            best_val = loss_val.item()
            wait = 0
            state_name = model_state_dir / f"{scenario_name}_replica_{i}.pt"
            torch.save(model.state_dict(), state_name)
        else:
            wait += 1
            if wait >= fit_params["patience"]:
                break

    # 6) Load best model & compute final chi2 on validation
    state_name = model_state_dir / f"{scenario_name}_replica_{i}.pt"
    model.load_state_dict(torch.load(state_name))
    model.eval()
    with torch.no_grad():
        f_raw_best = model(x_torch).squeeze()
        t3_unnorm_best = f_raw_best / xgrid
        integral_best = torch.dot(weights, t3_unnorm_best.float())

        norm_best = 1.0 / integral_best
        f_best_norm = norm_best * f_raw_best
        t_sm_best_val = w_torch[val_idx].matmul(f_best_norm)
        if use_bsm:
            t_pred_val_best = t_sm_best_val * (1.0 + model.c * k_torch_full[val_idx])
        else:
            t_pred_val_best = t_sm_best_val
        resid_v = t_pred_val_best - y_torch[val_idx]
        chi2_val_final = float(resid_v @ (cinv_val.matmul(resid_v)))

    chi2_perpt = chi2_val_final / float(n_val)

    # Prepare record
    alpha_val = float(torch.exp(model.logalpha).item())
    beta_val = float(torch.exp(model.logbeta).item())
    c_val = float(model.c.item()) if use_bsm else float("nan")

    record = {
        "replica": i,
        "chi2": chi2_val_final,
        "chi2_perpt": chi2_perpt,
        "alpha": alpha_val,
        "beta": beta_val,
        "c": c_val,
    }

    return record, f_best_norm.cpu().numpy()


# %%
# 9. EXECUTION CELLS
# ------------------------------------------------------------------------------
# 9.1 Precompute everything needed in memory
# Already stored: merged_df, W, c_yy, xgrid, weights, t3_ref_norm, y_pseudo_mean, y_real, q2_vals

# 9.2 Compute k_vectors: quadratic, linear, contact (with M=1e4)
k_vector_quadratic = compute_k_vector(q2_vals, key="quadratic", scale=1e4)
k_vector_linear = compute_k_vector(q2_vals, key="linear")
k_vector_contact = compute_k_vector(q2_vals, key="contact", M=1e4)

# 9.3 Define fit-parameter template
fit_params = {
    "n_replicas": 2,
    "n_hidden": 30,
    "n_layers": 3,
    "dropout": 0.2,
    "lambda_smooth": 1e-4,
    "patience": 500,
    "num_epochs": 5000,
}

# 9.4 Define injection values
injection_values = [0.01, 0.02, 0.05]

# 9.5 Build scenarios dynamically

data_modes = [
    {"name_suffix": "closure", "use_real": False, "injections": [*injection_values, 0.0]},
    {"name_suffix": "realdata", "use_real": True, "injections": [0.0]},
]

k_types = {
    None: None,
    "quadratic": k_vector_quadratic,
    "linear": k_vector_linear,
    "contact": k_vector_contact,
}

scenarios = []

# Build scenarios for both data modes
for mode in data_modes:
    suffix = mode["name_suffix"]
    use_real = mode["use_real"]
    injections = mode["injections"]

    # Base (no BSM)
    scenarios.append(
        {
            "name": f"base_{suffix}",
            "use_bsm": False,
            "k_key": None,
            "k_vector": None,
            "injection_c": 0.0,
            "use_real_data": use_real,
        },
    )

    # BSM scenarios
    for k_key, kv in k_types.items():
        if k_key is None:
            continue  # skip None here, base already added
        for inj in injections:
            # only include nonzero injections for closure
            if use_real and inj != 0.0:
                continue
            scenarios.append(
                {
                    "name": f"bsm_{k_key}_{suffix}_inj_{inj:.2f}".replace(".", "p"),
                    "use_bsm": True,
                    "k_key": k_key,
                    "k_vector": kv,
                    "injection_c": inj,
                    "use_real_data": use_real,
                },
            )
# %%
# 9.6 Run each scenario, collect results (including f_best arrays)
all_results = []

# We'll also keep a dictionary of lists of f_best arrays, keyed by scenario,
# so it's easy to extract stacks of fits for plotting.
scenario_fits: dict[str, list[np.ndarray]] = {sc["name"]: [] for sc in scenarios}

for sc in scenarios:
    name = sc["name"]
    use_bsm = sc["use_bsm"]
    kv = sc["k_vector"]
    inj_c = sc["injection_c"]
    use_real = sc["use_real_data"]
    k_key = sc["k_key"]

    logger.info(f"\n=== Running scenario: {name} ===")
    scenario_records = []

    for i in range(fit_params["n_replicas"]):
        logger.info(f"  Replica {i + 1}/{fit_params['n_replicas']} for scenario {name}")

        record, f_norm_array = train_one_replica(
            i,
            name,
            W,
            c_yy,
            xgrid,
            weights,
            y_pseudo_mean,
            y_real,
            fit_params,
            use_bsm,
            kv,
            inj_c,
            use_real,
        )

        # Build the final record dictionary, adding f_norm_array under "f_best"
        record_final = {
            "scenario": name,
            **record,
            "use_bsm": use_bsm,
            "injection_c": inj_c,
            "k_type": k_key if k_key is not None else "none",
            "use_real_data": use_real,
            "f_best": f_norm_array,  # array of length n_grid
        }
        scenario_records.append(record_final)
        scenario_fits[name].append(f_norm_array)

    logger.success(
        f"Finished scenario “{name}”: kept {len(scenario_records)}",
    )

    results_df = pd.DataFrame(scenario_records)
    all_results.append(results_df)

combined_df = pd.concat(all_results, ignore_index=True, sort=False)


# %%
"""PLOTTING RESULTS for T3_BSM_Comparison"""

# Now combined_df has columns including:
#   scenario, replica, chi2, chi2_perpt, alpha, beta, c, use_bsm, injection_c, k_type,
#   use_real_data, f_best
# And scenario_fits maps each scenario name to a list of f_best arrays (kept replicas).

# -------------------------------
# 1) Ensemble PDF Comparison
# -------------------------------
# Choose two scenarios to compare (e.g., base_closure vs bsm_quadratic_closure_inj_0p01)
sc1 = "base_closure"
sc2 = "bsm_quadratic_closure_inj_0p01"

fits1 = np.stack(scenario_fits[sc1])  # shape (n_kept1, n_grid)
fits2 = np.stack(scenario_fits[sc2])  # shape (n_kept2, n_grid)

mean1, std1 = fits1.mean(axis=0), fits1.std(axis=0)
mean2, std2 = fits2.mean(axis=0), fits2.std(axis=0)

plt.figure(figsize=(7, 5))
plt.fill_between(
    xgrid,
    mean1 - std1,
    mean1 + std1,
    alpha=0.3,
    label=f"{sc1} envelope",
)
plt.plot(xgrid, mean1, "-", label=f"{sc1} mean")

plt.fill_between(
    xgrid,
    mean2 - std2,
    mean2 + std2,
    alpha=0.3,
    color="C1",
    label=f"{sc2} envelope",
)
plt.plot(xgrid, mean2, "-", color="C1", label=f"{sc2} mean")

plt.xscale("log")
plt.xlabel(r"$x$")
plt.ylabel(r"$x\,T_3(x)$")
plt.title("Ensemble Comparison: " + sc1 + " vs. " + sc2)
plt.legend(fontsize="small")
plt.tight_layout()
plt.show()

# -------------------------------
# 2) alpha and beta Histograms
# -------------------------------
df1 = combined_df.query("scenario == @sc1")
df2 = combined_df.query("scenario == @sc2")

plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.hist(df1["alpha"], bins="auto", alpha=0.6, edgecolor="black", label=sc1)
plt.hist(df2["alpha"], bins="auto", alpha=0.6, edgecolor="black", label=sc2)
plt.xlabel(r"$\alpha$")
plt.ylabel("Replicas")
plt.title(r"Distribution of $\alpha$")
plt.legend()

plt.subplot(1, 2, 2)
plt.hist(df1["beta"], bins="auto", alpha=0.6, edgecolor="black", label=sc1)
plt.hist(df2["beta"], bins="auto", alpha=0.6, edgecolor="black", label=sc2)
plt.xlabel(r"$\beta$")
plt.ylabel("Replicas")
plt.title(r"Distribution of $\beta$")
plt.legend()

plt.tight_layout()
plt.show()

# -------------------------------
# 3) alpha vs beta Scatter
# -------------------------------
plt.figure(figsize=(5, 5))
plt.scatter(df1["alpha"], df1["beta"], alpha=0.7, label=sc1)
plt.scatter(df2["alpha"], df2["beta"], alpha=0.7, label=sc2)
plt.xlabel(r"$\alpha$")
plt.ylabel(r"$\beta$")
plt.title(r"Scatter: $\alpha$ vs $\beta$")
plt.legend()
plt.tight_layout()
plt.show()

# -------------------------------
# 4) χ²/pt Histograms (Kept Replicas)
# -------------------------------
chi2pt1_kept = df1["chi2_perpt"].to_numpy()
chi2pt2_kept = df2["chi2_perpt"].to_numpy()

plt.figure(figsize=(6, 4))
plt.hist(chi2pt1_kept, bins=20, alpha=0.5, edgecolor="black", label=f"{sc1} kept")
plt.hist(chi2pt2_kept, bins=20, alpha=0.5, edgecolor="black", label=f"{sc2} kept")
plt.axvline(0.8, color="red", linestyle="--")
plt.axvline(1.2, color="red", linestyle="--")
plt.xlabel(r"$\chi^2/\mathrm{pt}$")
plt.ylabel("Replicas")
plt.title(r"Kept Replicas: $\frac{\chi^2}{pt}$ Distribution")
plt.legend()
plt.tight_layout()
plt.show()

# -------------------------------
# 5) BSM Coefficient c Histogram
# -------------------------------
if "bsm" in sc2:
    plt.figure(figsize=(6, 4))
    plt.hist(df2["c"], bins="auto", alpha=0.7, edgecolor="black")
    plt.axvline(0.0, color="red", linestyle="--", label="c=0")
    mean_c = df2["c"].mean()
    std_c = df2["c"].std()
    plt.axvline(mean_c, color="C1", linestyle="-", label=f"mean={mean_c:.3f}")
    plt.fill_betweenx(
        [0, plt.gca().get_ylim()[1]],
        mean_c - std_c,
        mean_c + std_c,
        color="C1",
        alpha=0.2,
        label=rf"±1$\sigma$={std_c:.3f}",
    )
    plt.xlabel(r"BSM parameter $c$")
    plt.ylabel("Replicas")
    plt.title(f"Histogram of $c$ for {sc2}")
    plt.legend()
    plt.tight_layout()
    plt.show()

# -------------------------------
# 6) First Moment <x> Comparison
# -------------------------------
mom1 = np.array([np.trapz(xgrid * (f / xgrid), xgrid) for f in fits1])  # noqa: NPY201
mom2 = np.array([np.trapz(xgrid * (f / xgrid), xgrid) for f in fits2])  # noqa: NPY201

plt.figure(figsize=(6, 4))
plt.hist(mom1, bins=10, alpha=0.6, edgecolor="black", label=sc1)
plt.hist(mom2, bins=10, alpha=0.6, edgecolor="black", label=sc2)
plt.xlabel(r"$\langle x \rangle_{T_3}$")
plt.ylabel("Replicas")
plt.title("First Moment Comparison")
plt.legend()
plt.tight_layout()
plt.show()
# %%
# Include Plots from Previous
# Make plots nicely from big data frame, also fix the names ...
# Actually, saving models may not be the move, we seem to be drifting more than I remembered.
