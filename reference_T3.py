# %%
"""ggi_1.ipynb."""

import lhapdf
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras
from validphys.api import API
from validphys.fkparser import load_fktable
from validphys.loader import Loader

# %%
# ? Author Loadin
# auth_data_path = "data/prepared_data/"

# load data central values
# data = np.load(auth_data_path + "data.npy")

# load data covariance matrix
# Cy = np.load(auth_data_path + "Cy.npy")


# FK = np.load(auth_data_path + "FK.npy")

# load x-grid of the FK table
# fk_grid = np.load(auth_data_path + "fk_grid.npy")

# load results from NNPDF4.0. This is a vector containing the values of T_3 from
# NNPDF4.0 on the x-grid loaded in the previous lines
# f_ = np.load(auth_data_path + "NNPDF40.npy")

# f = f_[6 * 50 : 7 * 50]

# %%
# ? Our copy

# Load BCDMS F2_p, F2_d and form difference y = F2_p - F2_d
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


mp = 0.938
mp2 = mp**2

merged_df = (
    df_p.merge(df_d, on=["x", "Q2"], suffixes=("_p", "_d")).assign(
        y=lambda df: df["F2_p"] - df["F2_d"],
        W2=lambda df: df["Q2"] * (1 - df["x"]) / df["x"] + mp2,
    )  # difference
)

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
W = wp_t3[idx_p] - wd_t3[idx_d]  # convolution matrix (N_data, N_grid)

# %% ---- Covariance Matrix ----
params = {
    "dataset_inputs": [inp_p["dataset_input"], inp_d["dataset_input"]],
    "use_cuts": "internal",
    "theoryid": 200,
}
cov_full = API.dataset_inputs_covmat_from_systematics(**params)
n_p, n_d = len(df_p), len(df_d)
C_pp = cov_full[:n_p, :n_p]
C_dd = cov_full[n_p:, n_p:]
C_pd = cov_full[:n_p, n_p:]
C_yy = C_pp[np.ix_(idx_p, idx_p)] + C_dd[np.ix_(idx_d, idx_d)] - 2 * C_pd[np.ix_(idx_p, idx_d)]


# %% ---- x-grid & Tensors ----
xgrid = fk_p.xgrid  # (N_grid,)


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

# %%

# load data central values
data = merged_df["y"].to_numpy()

# load data covariance matrix
Cy = C_yy

FK = W

# load x-grid of the FK table
fk_grid = xgrid


f = T3_ref


# %%
# compute theory prediction using NNPDF4.0 and compare them with the experimental data

yth = FK @ f

sigma = np.sqrt(np.diagonal(Cy)) / data
x = np.arange(yth.size)
ref = np.ones(yth.size)
plt.figure(figsize=(20, 5))
plt.errorbar(x, ref, np.abs(sigma), alpha=0.5, label="data")
plt.scatter(x, yth, marker="*", c="red", label="theory predictions")
plt.ylim([0.01, 2.5])
plt.legend()
# %%
# Neural Network


class Linear(keras.layers.Layer):
    def __init__(self, units=32):
        super().__init__()
        self.units = units

    def build(self, input_shape):
        self.w = self.add_weight(
            shape=(input_shape[-1], self.units),
            initializer="random_normal",
            trainable=True,
        )
        self.b = self.add_weight(
            shape=(self.units,),
            initializer="random_normal",
            trainable=True,
        )

    def call(self, inputs):
        return tf.matmul(inputs, self.w) + self.b


# pdf layer
class pdf_NN(keras.layers.Layer):
    def __init__(self):
        super().__init__()
        self.linear_1 = Linear(64)
        self.linear_2 = Linear(64)
        self.linear_3 = Linear(32)
        self.linear_4 = Linear(1)

    def call(self, inputs):
        x = self.linear_1(inputs)
        x = tf.nn.relu(x)
        x = self.linear_2(x)
        x = tf.nn.relu(x)
        x = self.linear_3(x)
        x = tf.nn.relu(x)
        return self.linear_4(x)


# convolution layer
class ComputeConv(keras.layers.Layer):
    def __init__(self, FK):
        super().__init__()
        self.fk = tf.Variable(
            initial_value=tf.convert_to_tensor(FK, dtype="float32"),
            trainable=False,
        )

    def call(self, inputs):
        res = tf.tensordot(self.fk, inputs, axes=1)
        return res


# build the model
class Observable(keras.Model):
    """Combines the PDF and convolution into a model for training."""

    def __init__(
        self,
        FK,
        name="dis",
        **kwargs,
    ):
        super().__init__(name=name, **kwargs)
        self.pdf = pdf_NN()
        self.conv = ComputeConv(FK)

    def get_pdf(self, inputs):
        return self.pdf(inputs)

    def call(self, inputs):
        pdf_values = self.pdf(inputs)
        obs = self.conv(pdf_values)
        return obs


# %%
# define an observable, giving as input the corresponding FK table
thpred = Observable(FK)

# the input x-values correspond to the x-points enetring the FK table.
# For the following you need to reshape the input vector from (n,) to (n,1)
x_input = fk_grid[:, np.newaxis]

# the value of the observable as a function of the free parameters of the net
# can be obatined as thpred(x_input). Try. What yoiu get is a set of values
# corresponding to a random initialization of the net
# thpred(x_input)

# %%
# Loss function


def invert(Cy):
    l, u = tf.linalg.eigh(Cy)
    invCy = tf.cast(u @ tf.linalg.diag(1.0 / l) @ tf.transpose(u), "float32")
    invCy = u @ tf.linalg.diag(1.0 / l) @ tf.transpose(u)
    return tf.cast(invCy, "float32")


def chi2(y, yth, invCy):
    """Given a set of data with corrersponding th predictions and covariance matrix
    returns the value of the chi2
    """
    d = y - yth
    ndata = tf.cast(tf.size(y), "float32")
    res = tf.tensordot(d, tf.tensordot(invCy, d, axes=1), axes=1) / ndata
    return res


# %%

invCy = invert(Cy)
chi2(data, thpred(x_input)[:, 0], invCy)

x_input = fk_grid[:, np.newaxis]

# output grid for the final pdf results
grid_smallx = np.geomspace(1e-6, 0.1, 30)
grid_largex = np.linspace(0.1, 1.0, 30)
x_output = np.concatenate([grid_smallx, grid_largex])[:, np.newaxis]

# define the model
thpred = Observable(FK)

# define the optimizer
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)

# fix the numebr of epochs you want to use during the training
epochs = 500

train_chi2 = []
# implement the training loop
for epoch in range(epochs):
    with tf.GradientTape() as tape:
        obs = thpred(x_input)
        loss = chi2(data, obs[:, 0], invCy)

    # compute gradient wrt training parameters
    grads = tape.gradient(loss, thpred.trainable_weights)
    # update free parameters of the network
    optimizer.apply_gradients(zip(grads, thpred.trainable_weights))
    train_chi2.append(loss.numpy())

# %%
# now the model has been trained.
# We can get the corresponding PDF and use the x_output grid to plot it.
# We can plot NNPDF4.0 as well for reference.

pdf_result = thpred.get_pdf(x_output)
plt.plot(x_output, pdf_result.numpy(), c="green", alpha=0.5, label="fit")
plt.plot(x_input, f / x_input[:, 0], "--", c="black", label="NNPDF4.0")
plt.ylim([-1, 5])
plt.legend()


# %%
# Solution Ex 3
def plot_chi2_vs_epocs(chi2_res, epocs=500, label=r"$\chi^2$"):
    x = np.arange(epocs)
    plt.plot(x, chi2_res, label=label)
    plt.legend()
    # plt.show()


plot_chi2_vs_epocs(train_chi2)
# %%
# Monte Carlo fit


def fit_replica(y, Cy, invCy, epochs, noise=False):
    thpred_NN = Observable(FK)
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)

    if noise:
        yrep = tf.cast(np.random.multivariate_normal(y, Cy), "float32")
    else:
        yrep = tf.cast(y, "float32")

    # Iterate over epochs.
    train_chi2 = []
    for epoch in range(epochs):
        with tf.GradientTape() as tape:
            obs = thpred_NN(x_input)
            loss = chi2(yrep, obs[:, 0], invCy)

        grads = tape.gradient(loss, thpred_NN.trainable_weights)
        optimizer.apply_gradients(zip(grads, thpred_NN.trainable_weights))
        train_chi2.append(loss.numpy())

    return thpred_NN.get_pdf(x_output), np.asarray(train_chi2)


# run the fit
replicas_res = []
chi2_res = []

invCy = invert(Cy)

replicas = 10

for rep in range(replicas):
    print(f"fitting replica {rep}")
    res_, chi2_ = fit_replica(data, Cy, invCy, 500, noise=True)
    replicas_res.append(res_)
    chi2_res.append(chi2_)

# plot the results
for i in range(replicas):
    plt.plot(x_output, replicas_res[i].numpy() / x_output, c="green", alpha=0.5)
plt.ylim([-1, 5])
plt.xscale("log")
plt.plot(x_input, f / x_input[:, 0], "--", c="black", label="NNPDF4.0")
plt.legend()

# %%
