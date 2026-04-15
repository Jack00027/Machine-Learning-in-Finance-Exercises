# ================
#  Load libraries
# ================
import numpy as np
import tensorflow as tf
from keras.optimizers import Adam
from keras.layers import Input, Dense
from keras.models import Model
import matplotlib.pyplot as plt
import scipy.stats as scipy
from keras import regularizers
from mpl_toolkits.mplot3d import Axes3D
import math
import pandas as pd

"""In this code the price of an European Call option, i.e., $\mathbb E^Q[e^{-rT}\max(S(T)-K, 0)]$ with strike $K$ (can be a list of various strikes in this code) and maturity $T$ is estimated for three different Stockmodels.

To determine the prices, we consider the SDEs such that $W_t^Q$ is a Brownian motion under the risk-neutral measure $Q$ with interest rate $r$.

Stockmodel = 1 is the Black Scholes model solving the SDE
$dS_t=r S_t dt + \sigma S_t dW_t^Q$
where $\sigma$ is the volatility.

Stockmodel = 2 is the Constant elasticity of variance model (CEV) solving the SDE
$dS_t=r S_t dt + \sigma S_t^\eta dW_t^Q$
where $\sigma$ is the volatility and $\eta$ the elasticity.

Stockmodel = 3 is the Heston model solving the SDE

\begin{aligned}
dS_t &= r\,S_t\,dt \;+\; \sqrt{v_t}\,S_t\,dW_t^{Q}, \\
dv_t &= \eta\,(\theta - v_t)\,dt \;+\; \sigma\,\sqrt{v_t}\,dW_t^{v},\\
d W^{Q}_td W^{v}_t &= \rho\, dt,
\end{aligned}
where $S_t$ is the stock price, $\sqrt{v_t}$ the instantaneous volatility. The parameter $\kappa$ determines the mean reversion speed to the long term variance $\theta$, $\sigma$ the volatility of the volatility, and $\rho$ the constant correlation between the two Brownian moitions $W^Q$ and $W^v$.

In the following code block, the Stockmodel, its parameter range, i.e., $\Gamma$, and the payoff are chosen. You can add other Stockmodels and payoffs here.
"""

# =========================
#  Set parameter and model
# =========================

### Model parameters
N = 20 # time disrectization
S0 = 100 # initial value of the asset
T = 1/12 # maturity

r = 0.02  ## risk neutral interest rate



## The code provides 3 stockmodels
# 1 for Black Scholes model
# 2 for flexible drift/diff function, here CEV
# 3 stochvol model like Heston
Stockmodel = 2


# ===============================================================
#  If you want to add a Stockmodel, add: If Stockmodel = 4: [...]
#  and choose Stockmodel = 4 in the line above
# ===============================================================

## Select a model
if Stockmodel == 1:
    param_range = [ [0.1, 0.2] ] ## A list of tupels. Each tupel contains the parameter range
    param_name = ["volatility"]
    param_symbol = ["σ"]
    model_name = "Black-Scholes model"
if Stockmodel == 2:
    param_range = [ [0.2, 0.8] , [0.7 ,1] ]
    param_name = ["volatility", "elasticity"]
    param_symbol = ["σ","η"]
    model_name = "CEV model"
if Stockmodel == 3:
    param_range = [ [0.2, 0.2] , [-0.8,0.8] , [0.05, 1], [0.01, 0.4] ]
    param_name = ["long term variance", "correlation", "mean reversion rate", "volatility of the volatility"]
    param_symbol = ["θ","ρ","κ","σ"]
    model_name = "Heston model"

## Number of parameters
m = len(param_range)


## Strikes
K = [95, 100, 105]
l = len(K)

## Option payoff function [European call, i.e.  f_T = max( S(T)-K, 0) ]
def f(S):
    R, N, d = np.shape(S); N=N-1
    y = np.zeros((R, l))
    for j in range(l):
        y[:,j] = np.exp(-r * T)*np.maximum( S[:,N,0] - K[j] , 0 )
    return y



## Generate random parameters according to the specifications.
## Here, the choice is according to a uniform distribution on the paramter range
def random_parameter(R=1, param_range=param_range):
    p_num = len(param_range)
    p = np.zeros((R,p_num))
    for j in range(p_num):
        p[:,j] = np.random.uniform( param_range[j][0] , param_range[j][1] , R )
    return p



###### Define the stock price models and select one of them
## If a different vector of length N+1 is used, then the model is generated along the given vector
## This could be intresting if one expects that more frequent rebalancing
## is required near (or far away) from maturity
TimePoints = np.linspace(0,T,N+1)

# ===========================================
#  If you want to add a Stockmodel, add:
#  def path4(S0,Timepoints,param,r=0): [...]
# ===========================================

#### Defining the price model ####
## BS model
def path1(S0,Timepoints,param, r=0):
    R = param.shape[0]
    N = len(Timepoints) - 1
    X = np.zeros((R,N+1)) + np.log(S0)
    r_log = r - param[:,0]**2/2
    for j in range(N):
        dt = Timepoints[j+1] - Timepoints[j]
        dZ = np.random.normal(0,1,R)
        increment = r_log * dt + param[:,0] * dZ * np.sqrt(dt)
        X[:,j+1] = X[:,j] + increment
    S = np.exp(X)
    return np.reshape(S,(R,N+1,1))

## A CEV-type model
def path2(S0,Timepoints,param,r=0):
    sigma = param[:,0]
    eta = param[:,1]
    R = len(sigma)
    N = len(Timepoints) - 1
    S = np.zeros((R,N+1)) + S0
    # drift function
    def beta(s):
        return r*s
    # volatility function
    def a(s):
        return sigma*(s**eta)
    for j in range(N):
        dt = Timepoints[j+1] - Timepoints[j]
        dZ = np.random.normal(0,1,R)
        increment = beta(S[:,j])*dt + a(S[:,j]) * dZ * np.sqrt(dt)
        S[:,j+1] = np.abs( S[:,j] + increment )
    return np.reshape(S,(R,N+1,1))

## A Heston-type model
def path3(S0, Timepoints, param, r=0, max_retries=100):
    theta = param[:,0]  # long-term volatility
    rho = param[:,1]   # correlation
    kappa = param[:,2]   # mean reversion
    sigma = param[:,3]  # volatility of the volatility
    ## correlation is at most 1
    rho = np.minimum(rho, 1)
    R = param.shape[0]
    N = len(Timepoints) - 1
    retries = 0
    while True:
        S = np.zeros((R, N+1)) + S0
        v = np.zeros((R, N+1))
        v[:, 0] = theta / 2 # initial value
        restart = False
        for j in range(N):
            dt = Timepoints[j+1] - Timepoints[j]
            dZ = np.random.normal(0, 1, (R, 2))
            dZ[:,1] = rho*dZ[:,0] + np.sqrt(1 - rho**2) * dZ[:,1]
            increment_S = r*S[:,j]*dt + np.sqrt(v[:,j]) * S[:,j]*dZ[:,0]*np.sqrt(dt)
            increment_v = kappa*(theta-v[:,j])*dt + sigma * np.sqrt(v[:,j]) * dZ[:,1]*np.sqrt(dt)
            S[:,j+1] = S[:,j] + increment_S
            v[:,j+1] = v[:,j] + increment_v
            # Restart if volatility becomes negative
            if np.any(v[:,j+1] < 0):
                restart = True
                break
        if not restart:
            return np.reshape(S, (R, N+1, 1))
        retries += 1
        if retries >= max_retries:
            raise RuntimeError("Negative volatility encountered despite multiple restarts")



# ===================================================================
#  If you want to add a Stockmodel, add the according elif-condition:
#  elif Stockmodel == 4:
#      path = path4
# ===================================================================

if Stockmodel == 1:
    path = path1
elif Stockmodel == 2:
    path = path2
elif Stockmodel == 3:
    path = path3
else:
    raise("Stockmodel", Stockmodel,"Does not exist")

"""Consider the i.i.d. sequence $(\Theta_1,X_1),\dots,(\Theta_n,X_n)$, i.e., $X_i$ is one payoff for the chosen Stockmodel with model parameter $\Theta_i$.

We use this sample to define the quadratic function $V_{N,\lambda_N}:H\rightarrow L^2$ given by
\begin{align*}
    V_{N,\lambda_N}(h) &:=  \frac1N\sum_{i=1}^N (h(\Theta_i)-X_i)^2+\lambda_N\|h\|^2
\end{align*}
for a regularisation $\lambda_N>0$, any $h\in H$, and a suitable Hilbertspace $H\subset C(\Gamma,\mathbb R)$ (A Reproducing kernel Hilbert space satisfying Assumptions 2.1.)

Our main theorem of the paper is Theorem 2.2 which shows that the minimum of $V_{N,\lambda_N}$ exists and converges to the price function $h_0$ for $\lambda_N \to 0$ and $\lambda_NN^{\frac14}\to \infty$.

To improve the simulation results we use a pre-averaging. For each model parameter $\Theta_i$ we sample $M$ i.i.d. random variables $Y_{i,1},\dots,Y_{i,M}$ from this model and define the average
\begin{align*}
    X_i:=\frac1M \sum\limits_{j=1}^MY_{i,j}.
\end{align*}
This results in an i.i.d. sequence $(\Theta_1,X_1),\dots,(\Theta_n,X_n)$ which satisfies the MinMC assumptions.

This Code uses feature maps presented in Example 2.9 and neural networks to determine the minimum of $h_0$.

In these simulations the loss does not converge to zero since
\begin{align*}
    \mathbb{E} [ |X - h(\Theta)|^2 ] =  \int_\Gamma  Var [ X| \Theta=\theta] d \mu  +  \int_\Gamma \big( \mathbb{E} [ X| \Theta=\theta ] - h(\theta )  \big)^2  d \mu(\theta).
\end{align*}
Hence, the theoretic limit of the loss is the first integral.
"""



# =========================
#  Neural Network training
# =========================
## Train/Test Data size
Ktrain = 50000 # Size of training data
M = 5 # Number of pre-averages per training data

## Training parameters for the neural network
epochs = 10
batch_size = 32
learning_rate = 0.0001  # 0.001 is the default value
regularisation_rate = 0 # the regularisation rate is lambda in the paper

## Network structure
activator = "leaky_relu" # Activation function to use in the networks
#activator = "relu" # Activation function to use in the networks
d = 2 # number of hidden layers in strategy
n = 200  # nodes for nodes in hidden layers

## Digits shown in some print commands:
prec = 4
## Show error plots. (False to not show them)
ploterrors = False

#### Generating training data according to our model and specifications ####
# xtrain consists of possible parameters

print("\nCreating training data, the random parameters ...")

xtrain = random_parameter(Ktrain)  ## Produces Ktrain random parameters for the model

for i in range(M):
    Strain = path(S0, TimePoints, xtrain, r) ## Produces one price of the underlying per parameter
    if i==0:
        ytrain = f(Strain)  ## Produces the payoff
    else:
        ytrain += f(Strain)  ## Produces the payoff
ytrain /= M
y_var = np.var(ytrain,axis=0)
print("Variance payoffs:",y_var,"\nMean variance:",np.mean(y_var))
print("Done.")


###### Define the neural networks

#### Architecture of the network --- Expecting a vector with l prices and m parameters are the output ####
inputs = Input((m,))

output = inputs
for i in range(d):
    output = Dense(n,activation=activator)(output)
output = Dense(l, activation='linear', kernel_regularizer=regularizers.l2(regularisation_rate))(output)

pricerstd = Model(inputs=inputs, outputs = output)
print("\n[Below] Network for the pricer:")
pricerstd.summary()

###### Training the model ######
print("\nStart of training ...")
## pre running at learning rate specified earlier

### Choose an optimizer and learning rate.
optimizer = Adam(learning_rate=learning_rate)
pricerstd.compile(optimizer = optimizer, loss = 'mse')


## Output normalisation
mu = np.zeros(len(K))
sigma = np.zeros(len(K))
for j in range(len(K)):
    mu[j] = np.mean(ytrain[:,j])
    sigma[j] = np.std(ytrain[:,j])
y_std = (ytrain - mu)/sigma

## Fit the model
pricerstd.fit(x=xtrain, y=y_std,epochs=epochs, batch_size=batch_size)

def pricer(x):
    return(pricerstd(x)*sigma + mu)

"""The following outputs illustrate the price approximations generated by the neural network pricer under different parameter configurations. Depending on the number of variable parameters, the results are presented either as a single price, one-dimensional comparison plots against Monte Carlo simulations, or three-dimensional price surfaces. These visualizations highlight how the estimated option prices depend on the selected model parameters and strike values.

In the case of a single varying parameter, Monte Carlo prices are computed for Ktest=1000 randomly sampled parameter realizations. Only the selected parameter is varied explicitly, while all remaining parameters are fixed. Each Monte Carlo estimate is based on MC_num=500 simulated paths and is plotted pointwise against the corresponding neural network output.

When multiple parameters are variable, price surfaces are generated by varying two parameters at a time over a two-dimensional grid. All remaining parameters are fixed, either at their predefined values or at randomly sampled constants. For more than two free parameters, this pairwise visualization strategy is repeated for all parameter combinations, allowing the multidimensional dependence of the price approximation to be explored through two-parameter slices.
"""

# ===========================
# Price approximation plots
# ===========================


# ==========================================================
# Detect fixed and variable parameters
# ==========================================================

fixed_mask = [np.isclose(r[1] - r[0], 0.0) for r in param_range]
free_idx = [i for i in range(m) if not fixed_mask[i]]
num_free = len(free_idx)

# Random generator (set seed for reproducibility if desired)
rng = np.random.default_rng()  # e.g. np.random.default_rng(42)


# ================
# Grid for strikes
# ================

def adaptive_grid(n):
    cols = math.ceil(math.sqrt(n))
    rows = math.ceil(n / cols)
    return rows, cols


# ====================================================
# CASE 0: no variable parameters → single price output
# ====================================================

if num_free == 0:

    xtest = np.zeros((1, m))
    for i in range(m):
        xtest[0, i] = param_range[i][0]

    prices = pricer(xtest).numpy().flatten()

    fixed_str = ", ".join(
        f"{param_symbol[i]} = {param_range[i][0]}"
        for i in range(m)
    )

    print("All parameters are fixed.")
    print("Fixed parameters:", fixed_str)
    print("Price approximation:")

    for j in range(l):
        print(f"  Strike {K[j]}: {prices[j]:.{prec}f}")


# ================================================
# CASE 1: exactly one variable parameter → 1D plot
# ================================================

elif num_free == 1:

    a = free_idx[0]

    Ktest = 1000
    xtest_temp = random_parameter(Ktest)
    MC_num = 500

    xtest = xtest_temp.copy()

    # Fix all other parameters
    for i in range(m):
        if fixed_mask[i]:
            xtest[:, i] = param_range[i][0]
        elif i != a:
            xtest[:, i] = rng.uniform(
                param_range[i][0],
                param_range[i][1]
            )

    # Vary only the free parameter
    xtest[:, a] = xtest_temp[:, a]

    # NN prices
    NN_test = pricer(xtest).numpy()

    # Monte Carlo prices
    print("\nMC Pricing with parameter", param_name[a], "...")
    payoffs = np.zeros((Ktest, l, MC_num))
    for j in range(MC_num):
        payoffs[:, :, j] = f(path(S0, TimePoints, xtest, r))
    MC_test = np.mean(payoffs, axis=2)
    print("Done")

    # Plots
    rows, cols = adaptive_grid(l)
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
    axes = np.atleast_1d(axes).flatten()

    fig.suptitle(
        "Price function of the Neural Network (orange)\n"
        f"Each point represents a Monte Carlo simulation from {MC_num} observations",
        fontsize=14,
        y=0.94
    )

    for j in range(l):
        ax = axes[j]
        ax.plot(xtest[:, a], MC_test[:, j], "o", alpha=0.2)
        ax.plot(xtest[:, a], NN_test[:, j], "o")
        ax.set_title(f"Strike {K[j]}", pad=2)
        ax.set_xlabel(param_symbol[a])
        ax.set_ylabel("Price")

    for k in range(l, len(axes)):
        axes[k].axis("off")

    plt.tight_layout(rect=[0, 0, 1, 0.92])
    plt.show()


# ======================================================
# CASE 2: two or more variable parameters → Numeric validation:
# ======================================================

else:
    print("\nCreating test data, the random parameters ...")
    Ktest = 1000
    MC_num = 500
    
    xtest = random_parameter(Ktest)  ## Produces Ktrain random parameters for the model

    for i in range(MC_num):
        Strain = path(S0, TimePoints, xtest, r) ## Produces one price of the underlying per parameter
        if i==0:
            ytest = f(Strain)  ## Produces the payoff
        else:
            ytest += f(Strain)  ## Produces the payoff
    ytest /= MC_num
    print("Done.")
    
    ypredict = np.array(pricer(xtest))

    mean_err = np.zeros(len(K))
    std_err = np.zeros(len(K))
    mean_test = np.zeros(len(K))
    std_test = np.zeros(len(K))
    for j in range(len(K)):
        mean_err[j] = np.mean(ypredict[:,j]-ytest[:,j])
        std_err[j] = np.std(ypredict[:,j]-ytest[:,j])
        mean_test[j] = np.mean(ytest[:,j])
        std_test[j] = np.mean(ytest[:,j])
        plt.hist(ypredict[:,j]-ytest[:,j])
        plt.title("Histogram of errors for Option with strike:" + str(K[j]))
        plt.show()
    print("Mean ytest:",mean_test)
    print("Standard deviation ytest:",mean_test)
    print("Mean errors:",mean_err)
    print("Standard deviations of erros:",std_err)
    




