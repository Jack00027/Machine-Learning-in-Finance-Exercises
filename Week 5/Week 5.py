"""
Exercise (2): Recovering sigma from a signature model.

    X_t = c0 + c1 * Sig^1_t + c2 * Sig^2_t
        = c0 + c1 * W_t + c2 * (W_t^2 - t)/2
    with (c0, c1, c2) = (50, 10, 100).

    By Ito:   dX_t = (c1 + c2 W_t) dW_t,
    so       sigma_hat_t = dX_t / dW_t = c1 + c2 W_t.
"""

import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(0)

# ---- model constants ----
c0, c1, c2 = 50.0, 10.0, 100.0

# ---- simulation grid ----
T   = 1.0          # we want u = 0.5 and u = 1.0, so T = 1
N   = 2000         # time steps
R   = 400          # number of paths
dt  = T / N
t   = np.linspace(0.0, T, N + 1)

# ---- (a) simulate (W, X) ----
dW = rng.normal(0.0, np.sqrt(dt), size=(R, N))
W  = np.concatenate([np.zeros((R, 1)), np.cumsum(dW, axis=1)], axis=1)

# X via the closed-form signature representation
X = c0 + c1 * W + c2 * 0.5 * (W**2 - t[None, :])

# ---- (b) sigma_hat from increments: dX_t / dW_t ----
dX = np.diff(X, axis=1)
sigma_hat = dX / dW                       # shape (R, N)
# associate the increment with the LEFT endpoint t_k
t_left = t[:-1]

# theoretical curve for comparison: sigma(W) = c1 + c2 * W
W_grid = np.linspace(W.min(), W.max(), 400)
sig_of_W   = c1 + c2 * W_grid
X_of_W_05  = c0 + c1*W_grid + c2*0.5*(W_grid**2 - 0.5)
X_of_W_10  = c0 + c1*W_grid + c2*0.5*(W_grid**2 - 1.0)

# ---- plot one (W, X) sample path ----
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
for i in range(5):
    ax[0].plot(t, W[i], lw=0.8)
    ax[1].plot(t, X[i], lw=0.8)
ax[0].set_title("Brownian paths $W_t$");  ax[0].set_xlabel("t")
ax[1].set_title("Signature paths $X_t$"); ax[1].set_xlabel("t")
plt.tight_layout()
plt.show()

# ---- (c) scatter of sigma_hat_u over X_u for u = 0.5 and u = 1.0 ----
def idx_of(u):
    # index in t_left whose left-endpoint is closest to u
    return int(np.argmin(np.abs(t_left - u)))

fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
for ax, u in zip(axes, [0.5, 1.0]):
    k = idx_of(u)
    Xu  = X[:, k]                         # X_u across paths
    sgu = sigma_hat[:, k]                 # sigma_hat_u across paths

    mask = np.abs(sgu) < 600   # drop spikes from dW_k ~ 0 in dX/dW
    ax.scatter(Xu[mask], sgu[mask], s=14, alpha=0.5,
               label=r"empirical $(X_u,\hat\sigma_u)$")
    ax.set_ylim(-450, 450); ax.set_xlim(-50, 900)

    # theoretical parabola (parametric in W): X(W,u) vs sigma(W)
    Xth = X_of_W_05 if u == 0.5 else X_of_W_10
    ax.plot(Xth, sig_of_W, "r-", lw=2,
            label=r"theory: $\hat\sigma = c_1 + c_2 W$, $X = c_0+c_1W+c_2(W^2-u)/2$")

    ax.axhline(0, color="k", lw=0.5)
    ax.set_title(fr"$u = {u}$")
    ax.set_xlabel(r"$X_u$")
    ax.legend(loc="lower right", fontsize=9)
axes[0].set_ylabel(r"$\hat\sigma_u = dX_u/dW_u$")
plt.suptitle(r"Recovered local volatility $\hat\sigma_u$ as a function of $X_u$")
plt.tight_layout()
plt.show()

# ---- numerical sanity check ----
print("Mean |sigma_hat - (c1 + c2 W)| over all (path, time):",
      np.mean(np.abs(sigma_hat - (c1 + c2 * W[:, :-1]))))
print("Should be ~0 (it equals 0 in continuous time; tiny here from the increment)")