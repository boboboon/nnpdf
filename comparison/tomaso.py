"""Bayesian Paper Code."""

# %%
import numpy as np
import pymc as pm
import pytensor.tensor as pt

print(f"Running on PyMC v{pm.__version__}")

# %%
RANDOM_SEED = 8927
rng = np.random.default_rng(RANDOM_SEED)
# az.style.use("arviz-darkgrid")

# %%
# load data covariance matrix
Cy = np.load("Cy.npy")

# load data
y = np.load("data.npy")

# load FK table
FK = np.load("FK.npy")

# load x-grid of the FK table
fk_grid = np.load("fk_grid.npy")

# load T3 from NNPDF4.0
f_true = np.load("f_bcdms.npy")

L1_noise = np.load("L1_noise_BCDMS.npy")


# %%
n = len(fk_grid)  # number if points of the FK table xgrid
X = fk_grid[:, None]  # The inputs to the GP must be arranged as a column vector

# fit the real data
# y_obs = y

# fit pseudo-data
y_true = FK @ f_true
y_obs = y_true + L1_noise

# %%
# run for kinlim only!!!
from scipy.linalg import block_diag


def kinlim(y, Cy, FK):
    ngrid = FK.shape[1]
    A = np.zeros(ngrid)
    A[ngrid - 1] = 1.0
    FK_kinlim = np.block([[FK], [A]])
    y_kinlim = np.concatenate([y, np.zeros(1)])
    Cy_kinlim = block_diag(Cy, 1e-6 * np.identity(1))
    return y_kinlim, Cy_kinlim, FK_kinlim


y_obs, Cy, FK = kinlim(y_obs, Cy, FK)


# %%
N = Cy.shape[0]  # The number of data points


# function f(x)=x^alpha to rescale given kernel as f(x)k(x,y)f(y)
def scaling_function(X, alpha):
    return pm.math.exp(alpha * pm.math.log(X))


# correlation length entering Gibbs kernel definition
eps = 1e-6


def l(x, l0, eps):
    return l0 * (x + eps)


# fix alpha to given value which ensures integrability properties
# alpha = -0.5

with pm.Model() as gp_fit:
    # take zero mean function
    mu = np.zeros(N)

    # prior on hyperparameters
    # l0 = pm.HalfCauchy("l0", 5)
    # sigma = pm.HalfCauchy("sigma", 5)
    l0 = pm.Uniform("l0", lower=0, upper=10)
    sigma = pm.Uniform("sigma", lower=0, upper=10)
    alpha = pm.Uniform("alpha", lower=-0.9, upper=0)

    # l0 = pm.Normal("l0", mu=10, sigma=2)
    # sigma = pm.Normal("sigma", mu=10, sigma=2)

    # build the kernel
    kernel_ = sigma**2 * pm.gp.cov.Gibbs(1, l, args=(l0, eps))

    # rescale the kernel for small-x behaviour
    kernel = pm.gp.cov.ScaledCov(1, scaling_func=scaling_function, args=(alpha), cov_func=kernel_)

    # build the likelihood p(y|theta)
    Sigma = pt.dot(pt.dot(FK, kernel(X)), FK.T) + Cy
    y = pm.MvNormal("y", mu=mu, cov=Sigma, observed=y_obs)

# %%
# define grid for f*
grids_smallx = np.geomspace(1e-6, 0.1, 100)
grids_largex = np.linspace(0.1, 1.0, 100)
grids = np.concatenate([grids_smallx, grids_largex])
Xs = grids[:, None]

gp_fit.add_coords({"Xs": grids, "y": y, "X": fk_grid})


# now define the deterministic variable mu_post and sigma_post
with gp_fit as gp:
    # build Kx*x*
    sigma_pred = kernel(Xs)

    # build Kx*x
    kernel_off_diag = kernel(Xs, X)

    # build Kx*x FK.T
    sigma_off_diag = pt.dot(kernel_off_diag, FK.T)

    # Posterior mean.
    # Deterministic random variable: its value is completely determined by its parents’ values.
    # By wrapping the variable in Deterministic and giving it a name, you are saving this value in the trace

    mu_post = pm.Deterministic(
        "mu_post",
        pt.dot(pt.dot(sigma_off_diag, pm.math.matrix_inverse(Sigma)), y_obs),
        dims="Xs",
    )

    # Posterior covariance
    sigma_post = pm.Deterministic(
        "sigma_post",
        sigma_pred
        - pt.dot(pt.dot(sigma_off_diag, pm.math.matrix_inverse(Sigma)), sigma_off_diag.T),
        dims=("Xs", "Xs"),
    )


# %%
# run MCMC to sample from the posterior p(theta|y)
with gp_fit:
    trace = pm.sample(target_accept=0.9, nuts_sampler="numpyro")

# %%
trace.to_netcdf("BCDMS_L1.nc")
