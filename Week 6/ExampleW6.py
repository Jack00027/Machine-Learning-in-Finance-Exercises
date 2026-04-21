"""
Signature-based regression for controlled stochastic systems.

This code implements a learning framework for stochastic dynamics of the form
    dY_t = a(Y_t) dX_t,
where X is a stochastic input path (e.g. Brownian motion augmented with time),
and Y is generated numerically via an Euler scheme.

The main idea is to approximate the dynamics in signature space by learning
a linear model of the form
    dY_t ≈ sum_w c_w dS_t^w,
where S^w denotes iterated integrals (signature terms) of the input path X
truncated at a given level M.

------------------------------------------------------------
TRAINING PHASE
------------------------------------------------------------
For one training trajectory (X, Y):
    - compute the signature S(X) up to level M
    - form signature increments dS^w
    - fit coefficients c_w using ordinary least squares:
          dY ≈ Φ c
      where Φ=dS is the design matrix
    - store the learned coefficients

------------------------------------------------------------
TEST PHASE
------------------------------------------------------------
For unseen test paths:
    - compute signature S(X)
    - reconstruct an output path Z via:
          Z_0 = Y_0
          dZ_t = sum_w c_w dS_t^w
    - compare Z with the true Euler-simulated path Y using:
          (i) visual comparison of paths
          (ii) mean squared error of increments:
                MSE = mean( |dY_t - dZ_t|^2 )

------------------------------------------------------------
MODEL SELECTION
------------------------------------------------------------
The procedure is repeated for different signature truncation levels M.

For each M:
    - compute training error (fit quality)
    - compute test error (generalisation performance)

Results are plotted jointly to illustrate the bias–variance tradeoff:
    - low M: underfitting
    - intermediate M: best generalisation
    - high M: overfitting / numerical instability and computationally costly
    - Note that the test performance is measured with a low MC average which is not so stable.
"""

#%%

import numpy as np
import matplotlib.pyplot as plt
from itertools import product


# =========================
# PARAMETERS
# =========================
T = 1.0
N = 500
dt = T / N

mu = 0.1
sigma = 0.2

path_num = 4  # number of paths for visualization

M = 2 # Signature levels to use

prec = 4 ## Precision shown


#%%
# =========================
# 1. Simulate X path
# =========================
def simulate_input_path(T=1.0, N=500, seed=None):
    """
    Simulate a 2D path X_t = (W_t, t)

    Parameters
    ----------
    T : float
        Time horizon
    N : int
        Number of time steps
    seed : int or None
        Random seed for reproducibility

    Returns
    -------
    X : array (N+1, 2)
        Path with:
        X[:,0] = Brownian motion W_t
        X[:,1] = time t
    t_grid : array (N+1,)
        Time grid
    """

    if seed is not None:
        np.random.seed(seed)

    dt = T / N
    t_grid = np.linspace(0.0, T, N+1)

    # Brownian increments
    dW = np.sqrt(dt) * np.random.randn(N)

    # Brownian path
    W = np.zeros(N+1)
    W[1:] = np.cumsum(dW)

    # assemble path
    X = np.zeros((N+1, 2))
    X[:, 0] = W
    X[:, 1] = t_grid

   
    return X, t_grid


X, t = simulate_input_path(T=1.0, N=500, seed=42)


plt.plot(t, X[:,0], label="X^0")
plt.plot(t, X[:,1], label="X^1")
plt.legend()
plt.grid()
plt.show()


#%%
# =========================
# 2. Euler scheme for Y based on X: dY = a(Y)dX
# =========================
def simulate_Y_euler(X, Y0):
    """
    Euler scheme for:
        dY_t = a(Y_t) dX_t

    Parameters
    ----------
    X : array (N+1, 2)
        Input path (X^0 = Brownian, X^1 = time)
    Y0 : array (2,)
        Initial condition

    Returns
    -------
    Y : array (N+1, 2)
        Simulated path
    """

    N = X.shape[0] - 1
    dX = X[1:] - X[:-1]

    Y = np.zeros((N+1, 2))
    Y[0] = Y0

    # define coefficient function a(y)
    def a(y):
        """
        Matrix a(y) of shape (2,2)
        """
        y0, y1 = y

        return np.array([
            [y0 * y1, 0.2 * y0],
            [0.0    , np.cos(2*y1)]
        ])

    # Euler scheme
    for t in range(N):
        A = a(Y[t])              # (2,2)
        dXt = dX[t]              # (2,)
        
        # matrix-vector product
        Y[t+1] = Y[t] + A @ dXt

    return Y


# initial condition
Y0 = np.array([1, 0.5])

# simulate Y
Y = simulate_Y_euler(X, Y0)

# plot
import matplotlib.pyplot as plt

plt.plot(t, Y[:,0], label="Y^0")
plt.plot(t, Y[:,1], label="Y^1")
plt.legend()
plt.grid()
plt.show()



#%%
# =========================
# 3. Signature up to level M (manual, left-point Ito)
# =========================
def ito_signature(X, M):
    """
    Compute the Ito signature of a path X up to level M.

    Parameters
    ----------
    X : array of shape (N+1, d)
        The path
    M : int
        Maximum level of the signature

    Returns
    -------
    sig : dict
        Dictionary mapping words (tuples) -> arrays of length N+1
        Example: sig[(0,1)] is the iterated integral along coordinates 0 then 1
    """

    N, d = X.shape[0] - 1, X.shape[1]
    
    # increments
    dX = X[1:] - X[:-1]

    # signature dictionary
    sig = {}

    # level 0 (constant 1)
    sig[()] = np.ones(N+1)

    # level 1
    for i in range(d):
        S = np.zeros(N+1)
        for t in range(N):
            S[t+1] = S[t] + dX[t, i]
        sig[(i,)] = S

    # higher levels
    for level in range(2, M+1):
        for word in product(range(d), repeat=level):
            
            prefix = word[:-1]
            last = word[-1]

            S = np.zeros(N+1)

            for t in range(N):
                # Ito left-point approximation:
                # ∫ S^{prefix}_s dX^{last}_s ≈ S^{prefix}_t * ΔX^{last}_t
                S[t+1] = S[t] + sig[prefix][t] * dX[t, last]

            sig[word] = S

    return sig


print("Calculating the signature up to level:",M)
sig =  ito_signature(X, M)
print("Done.")
K, d = np.shape(X)
print("Total signature dimension:",(d**(M+1)-1)/(d-1))


#%%
# =========================
# 4. Fit d log(Y) ≈ Σ c_w dS^w
# =========================
def fit_signature_ols(X, Y, M):
    """
    Fit dY ≈ sum_w c_w dS^w using OLS.

    Parameters
    ----------
    X : array (N+1, d)
        Input path
    Y : array (N+1,) or (N+1, m)
        Output path (scalar or vector-valued)
    M : int
        Signature level

    Returns
    -------
    coeffs : dict
        Mapping word -> coefficient (scalar or vector)
    """

    # compute signature
    sig = ito_signature(X, M)

    N = X.shape[0] - 1

    # ensure Y is 2D
    Y = np.asarray(Y)
    if Y.ndim == 1:
        Y = Y[:, None]

    # increments of Y
    dY = Y[1:] - Y[:-1]   # shape (N, m)

    # list of words (exclude level 0 = ())
    words = [w for w in sig.keys() if len(w) > 0]
    words.sort(key=lambda w: (len(w), w))  # nice ordering

    # build design matrix
    # each column = dS^w
    Phi = np.zeros((N, len(words)))

    for j, w in enumerate(words):
        S = sig[w]
        dS = S[1:] - S[:-1]   # increments
        Phi[:, j] = dS

    # OLS solve: minimize ||dY - Phi c||^2
    # works also for vector-valued Y
    C, _, _, _ = np.linalg.lstsq(Phi, dY, rcond=None)

    # store coefficients in dictionary
    coeffs = {}
    for j, w in enumerate(words):
        coeffs[w] = C[j]  # shape (m,)

    return coeffs



## Coefficient fitting

coeffs = fit_signature_ols(X, Y, M)

#%%
# =========================
# Reconstruct Sig^c_t
# =========================

def simulate_signature_path(X, coeffs, M, Y0):
    """
    Construct Y_t = sum_w c_w S^w_t from signature coefficients.

    Parameters
    ----------
    X : array (N+1, d)
        Input path
    coeffs : dict
        Output of fit_signature_ols(), mapping word -> coefficient
    M : int
        Signature level (used to recompute signature)
    Y0: Starting value

    Returns
    -------
    Y : array (N+1,) or (N+1, m)
        Simulated path
    """

    sig = ito_signature(X, M)

    N = X.shape[0] - 1

    # detect output dimension
    m = np.shape( coeffs[(0,)] )[0]

    Y = np.zeros((N+1, m)) + Y0

    # build linear combination
    for w, c in coeffs.items():
        S_w = sig[w]  # shape (N+1,)
        
        if m == 1:
            Y[:, 0] += c * S_w
        else:
            # broadcast: (N+1,1) * (m,)
            Y += S_w[:, None] * c[None, :]

    # return 1D if scalar
    if m == 1:
        return Y[:, 0]
    
    return Y





#%%
# =========================
# 5. Run simulation + visualization
# =========================
def compare(X, coeffs, Y0, M, text="", plot=True):
    """
    Compare Euler simulation vs signature-based simulation.

    Parameters
    ----------
    X : array (N+1, d)
        Input path
    coeffs : dict
        Coefficients from fit_signature_ols
    Y0 : array (m,)
        Initial condition
    M : int
        Signature level
    text : plot header text
    plot : logical / True for plotting    
    
    Returns
    -------
    MSE
    """

    # simulate paths
    Y = simulate_Y_euler(X, Y0)
    Z = simulate_signature_path(X, coeffs, M, Y0)

    # ensure shapes are consistent
    Y = np.asarray(Y)
    Z = np.asarray(Z)

    if Y.ndim == 1:
        Y = Y[:, None]
    if Z.ndim == 1:
        Z = Z[:, None]

    m = Y.shape[1]

    # plot each component separately
    if plot:
        for i in range(m):
            plt.figure(figsize=(8, 4))
            
            plt.plot(t, Y[:, i], label="Euler Y", linewidth=2)
            plt.plot(t, Z[:, i], "--", label="Signature Z", linewidth=2)
        
            plt.title(f"Component {i} "+text)
            plt.xlabel("t")
            plt.ylabel(f"Value (dim {i})")
            plt.legend()
            plt.grid()

        plt.show()
 
    # increments
    dY = Y[1:] - Y[:-1]
    dZ = Z[1:] - Z[:-1]

    # Errors of component zero
    err = dY[:,0] - dZ[:,0]
    # Eucliden error
    err = np.sqrt( np.sum( (dY - dZ)**2 , axis=1) )

    if plot:
        plt.figure()
        plt.hist(err, bins=50)
        plt.title("Histogram of increment errors |dY - dZ|")
        plt.grid()
        plt.show()
    return(np.mean(err**2))

    
print("MSE (Training data) * time steps:",np.round( N * compare(X, coeffs, Y0, M,"On TRAINGING data") , prec))


#%%
# =========================
# 6. Simulation new X paths, and Y via both Euler and Signature
# =========================

for _ in range(path_num):
    Xnew, t = simulate_input_path(T,N)
    MSE = compare(Xnew, coeffs, Y0, M, "On test data!")
    print("MSE (Test path) * time steps:",np.round(MSE*N,prec))

    

#%%
# =========================
# 7. Training error for different levels of signature:
# =========================

def f(coeffs):
    return N * compare( simulate_input_path(T,N)[0] , coeffs, Y0, M,"",False)

L = 8
errors = np.zeros(L)
test_errors = np.zeros(L)
print("")
for j in range(L):
    M = j+1
    dims = (d**(M+1)-1)/(d-1)
    print("Calculating training error for signature approximation with ",M,"levels that are ",int(dims)," dimensions ...")
    coeffs = fit_signature_ols(X, Y, M)
    errors[j] = N * compare(X, coeffs, Y0, M,"",False)
    Kmax = 10
    test_errors[j] = np.mean(np.stack([f(coeffs) for _ in range(Kmax)]), axis=0)
print("All done.")

plt.plot(np.linspace(1, L, L), np.log(errors),'o')
plt.plot(np.linspace(1, L, L), np.log(test_errors),'o')
plt.title("Log Errors (MSE)")
plt.xlabel("Signature levels used")
plt.ylabel("Training MSE (blue) // Average MSE (orange)")
plt.show()