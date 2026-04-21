# -*- coding: utf-8 -*-
"""ML_week5.py"""

#### Problem desctiption
## We are given R = 1 path of prices with their volatilites [S, V]
## We like to find a simulator to create new paths.



# %%
######## Deep Simulation code ########
###### Load libraries and set parameters.
import numpy as np
#import tensorflow as tf
import math
from keras.layers import Input, Dense
from keras.models import Model
from matplotlib import pyplot as plt

## Digits shown in some print commands:
prec = 3



S0 = 100 # initial value of the asset.
T = 2/12 # maturity in years. Here "2 month" equal T = 2/12
N = int(T*1500) # time steps. T*1500 is approximately 1 timestep per hour while exchange is open

## Number so signature levels used
# longer maturities need more signature levels (or a more refined method!)
M = 5

## Train/Test setup
# R is supposed to be 1.
R = 1 # number of Trajectories --- this is our training sample


#### The code provides 3 stockm odels
# 1 for Black Scholes model
# 2 for flexible drift/diff function, here CEV
# 3 stochvol model like Heston

Stockmodel = 1

mu = 0.02  ## Q-dynamics: mu=0


## Select a model
if Stockmodel == 1:
    param_range = [ [0.1, 0.3] ] ## A list of tupels. Each tupel contains the parameter range
    useBSformula = True ## Use the BS-option price formula for comparisson
    param_name = ["volatility"]
    model_name = "Black-Scholes model"
if Stockmodel == 2:
    param_range = [ [0.2, 0.8] , [0.7 ,1] ]
    useBSformula = False
    param_name = ["volatility", "elasticity"]
    model_name = "CEV model"
if Stockmodel == 3:
    param_range = [ [0.2, 0.8] , [0, 0.1] , [0.15, 0.25], [0.01, 0.03] ]
    useBSformula = False
    param_name = ["volatility", "reversion rate", "long-term volatility", "volvol"]
    model_name = "Heston model"

## Number of parameters
m = len(param_range)


# %%
# Train on log-prices?
UseLogPrice = True



## Network setting for Deep Simulation
# Use quadratic solver?
UseLRsolve = True
# Training setting if not using quadratic solver
epochs = 10
batch_size = 4

learning_rate = 1
learning_rate_decay = 0.8

# %%

## Calculate the increments of a process S
def d_process(S):
    R, N, d = np.shape(S);
    N = N - 1
    dS = np.zeros((R,N,d))
    for j in range(N):
        dS[:,j,:] = S[:,j+1,:] - S[:,j,:]
    return dS

## Calculate a process from its starting value S0 and its increments dS
def process(dS,S0):
    if len(np.shape(dS)) == 2:
        R, N = np.shape(dS);
        dS = np.reshape(dS, (R,N,1))
    R, N, d = np.shape(dS);
    S = np.zeros((R,N+1,d)) + S0
    for j in range(N):
        S[:,j+1,:] = S[:,j,:] + dS[:,j,:]
    return S


# %%
## Generate random parameters according to the specifications.
def random_parameter(R=1, param_range=param_range):
    p_num = len(param_range)
    p = np.zeros((R,p_num))
    for j in range(p_num):
        p[:,j] = np.random.uniform( param_range[j][0] , param_range[j][1] , R )
    return p

# %%
###### Define and select a model
#### Time points ####
## If a different vector of length N+1 is used, then the model is generated
## along the given vector
## This could be intresting if one expects that more frequent rebalancing
## is required near (or far away) from maturity
TimePoints = np.linspace(0,T,N+1)

#### Converter of the time information that is fed into the NN
def TimeConv(t):
     # return np.sqrt(T-t)   ## Works better as input variable in case of diffusion models!
     return T-t  ## The NN expects a time to maturity as information


#### Defining the price model ####
## Brownian motion increments
def BM_paths(R,N,increment=True):
    dt = T/N
    dW = np.random.normal(0,np.sqrt(dt),R*N)
    dW = dW.reshape((R,N))
    return dW

## Brownian motion from its increments (starting value is 0)
def BM(dW):
    R, N = np.shape(dW)
    W = np.zeros((R,N+1))
    for j in range(N):
        W[:,j+1] = W[:,j] + dW[:,j]
    return W

## BS model
def path1(S0,mu,param,Timepoints,R,dW):
    sigma = param
    N = len(Timepoints) - 1
    X = np.zeros((R,N+1)) + np.log(S0) ## the last 1 is for the dimension of the model
    mu_log = mu - sigma**2/2
    for j in range(N):
        dt = Timepoints[j+1] - Timepoints[j]
        increment = mu_log * dt + sigma * dW[:,j]
        X[:,j+1] = X[:,j] + increment
    S = np.exp(X)
    V = sigma * S
    return np.reshape(S,(R,N+1,1)), np.reshape(V,(R,N+1,1))

## A CEV-type model
def path2(S0,mu,param,Timepoints,R,dW):
    sigma, eta = param
    N = len(Timepoints) - 1
    S = np.zeros((R,N+1)) + S0
    V = np.zeros((R,N+1))
    # drift function
    def beta(s):
        return mu*(s**eta)
    # volatility function
    def a(s):
        return sigma*(s**eta)
    for j in range(N):
        dt = Timepoints[j+1] - Timepoints[j]
        V[:,j] = a(S[:,j])
        increment = beta(S[:,j])*dt + V[:,j] * (dW[:,j])
        S[:,j+1] = S[:,j] + increment
    V[:,N] = a(S[:,N])
    return np.reshape(S,(R,N+1,1)), np.reshape(V,(R,N+1,1))

## A Heston-type model
def path3(S0,mu,param,Timepoints,R,dW):
    sigma0, eta, sigma, volvol = param
    rho = 0.3
    N = len(Timepoints) - 1
    S = np.zeros((R,N+1)) + S0
    A = np.zeros((R,N+1)) + sigma0
    for j in range(N):
        dt = Timepoints[j+1] - Timepoints[j]
        dZ = np.random.normal(0,np.sqrt(T/N) ,(R,))
        dB = rho * dW[:,j] + np.sqrt(1-rho**2) * dZ
        increment_S = mu*S[:,j]*dt + (A[:,j]*S[:,j]) * dW[:,j]
        increment_A = eta*(sigma-A[:,j])*dt + volvol * dB
        S[:,j+1] = S[:,j] + increment_S
        A[:,j+1] = A[:,j] + increment_A
    return np.reshape(S,(R,N+1,1)), np.reshape( A*S ,(R,N+1,1))


## Select a model
path = path1
name = "BS model"
if Stockmodel == 2:
    path = path2
    name = "CEV model"
if Stockmodel == 3:
    path = path3
    name = "'Heston' model"


param = random_parameter(1)[0,:]

#### Creating 1 sample paths from the model ####
dW = BM_paths(R,N)
S, V = path(S0,mu,param,TimePoints,R,dW)

for i in range(R):
   plt.figure(figsize=(10,5))
   # Plot Price cooridnate 1
   plt.subplot(1,2,1)
   plt.plot(TimePoints,S[i,:,0])
   plt.title("Sample path/s from "+str(model_name))
   # Plot volatility coordinate 1
   plt.subplot(1,2,2)
   plt.plot(TimePoints,V[i,:,0])
   plt.title("Diffusion 'σ_t * S_t' path")
   plt.show()

# %%
#### Recaputre Noise increments
## Since  dS = V * dB we should have
## dB = dS / V

def find_noise(S,V):
    if UseLogPrice:
        dX = d_process(np.log(S))
        dB = (S[:,0:N,:]*dX) / V[:,0:N,:]
    else:
        dS = d_process(S)
        dB = dS / V[:,0:N,:]
    B = process(dB,0)
    return B, dB

B, dB = find_noise(S,V)


for i in range(R):
   plt.figure(figsize=(10,5))
   # Plot Price cooridnate 1
   plt.subplot(1,2,1)
   plt.plot(TimePoints,S[i,:,0])
   plt.title("Sample path/s from "+str(model_name))
   # Plot volatility coordinate 1
   plt.subplot(1,2,2)
   plt.plot(TimePoints,B[i,:,0])
   plt.title("Recovered Brownian 'dB_t' paths")
   plt.show()

# %%
#### Create the signature according to B
## Hermite polynomials
def p(n,x,t=1):
    l = math.floor(n/2)
    y = 0
    for k in range(l+1):
        d = math.factorial(k)*math.factorial(n-2*k)*2**k
        y += x**(n-2*k)*(-t)**k/d
    return y

## Exact Brownian signature
def signature(B,M=10,T=1):
    R, N, d = np.shape(B)
    Sig = np.zeros((R, N, M+1))
    Sig[:,:,0] =  1
    for k in range(M):
        n = k+1
        for l in range(N):
            dt = T/(N-1)
            Sig[:,l,n] = p(n,B[:,l,0],l*dt)
    return Sig


# def signature_simul(B,M=10,T=1):
#     R, N, d = np.shape(B)
#     dB = d_process(B)
#     Sig = np.zeros((R, N, M+1))
#     Sig[:,:,0] =  1
#     A = np.zeros((M+1,M+1))
#     for l in range(M):
#         A[l,l+1] = 1
#     for n in range(N-1):
#         Sig[:,n+1,:] = Sig[:,n+1,:] + np.dot(Sig[:,n,:], A) * dB[:,n]
#     return(Sig)

print("\nCreating the signature of the BM")
Sig_B = signature(B,M=M,T=T)

#dSig_B = d_process(Sig_B)[:,:,1:M]
print("Done.")

## Show signature paths. Has been deactivated.
if False:
    M_show = min( 3 , M)
    for i in range(M_show+1):
        plt.plot(TimePoints,Sig_B[0,:,i])
    plt.title("Signature of (first) recovered Brownian path up to level "+str(M_show)+".")
    plt.show()

# %%
#### Architercure of the network --- S = Lin(Sig_B) ####
## g( Signature ) = PositionSize
Sig_input = Input(shape=(M,))

output = Sig_input
#for i in range(d):
#    output = Dense(n,activation=activator)(output)
output = Dense(1, activation='linear', use_bias=False)(output)

Linear = Model(inputs=Sig_input,outputs = output)
print("\n[Below] Netowrk for the Deep Simulation:")
Linear.summary()


# %%
#### Prepare training data
def make_x(Sig,R):
    dSig = d_process(Sig)[:,:,1:(M+1)]
    xtrain = dSig.reshape(R*N,M)
    return xtrain

xtrain = make_x(Sig_B,R)

def make_y(S,R):
    if UseLogPrice:
        dX = d_process(np.log(S))
        ytrain = dX.reshape( (R*N,1) )
    else:
        dS = d_process(S)
        ytrain = dS.reshape( (R*N,1) )
    return ytrain

ytrain = make_y(S,R)

def make_S(y,S0,R):
    if UseLogPrice:
        S = np.zeros( (R,N+1,1) ) + np.log(S0)
    else:
        S = np.zeros( (R,N+1,1) ) + S0
    y = y.reshape( (R,N,1) )
    for k in range(N):
        S[:,k+1,:] = S[:,k,:] + y[:,k,:]
    if UseLogPrice:
        return np.exp(S)
    return S



# %%
###### Training the model ######
print("\nStart of training ...")

Linear.compile(optimizer='adam', loss = 'mse')

if UseLRsolve:
    weights, _, _, _ = np.linalg.lstsq(xtrain, ytrain, rcond=None)
    # Set the weights of the model to the linear regression coefficients
    Linear.set_weights([weights])
else:
    optimizer = Linear.optimizer
    optimizer.learning_rate.assign(learning_rate)
    k_count = 5
    for k in range(epochs):
        if k >= k_count:
            print("Epochs:",k_count,"/",epochs," current learning rata:",learning_rate)
            k_count += 5
        optimizer.learning_rate.assign(learning_rate)
        learning_rate = learning_rate * learning_rate_decay
        Linear.fit(x=xtrain, y=ytrain, epochs=1, batch_size=batch_size)
print("Done.")

## Take the model weights into a matrix A
## The model does  ytrain = xtrain A
A = np.array([num for sublist in Linear.get_weights()[0] for num in sublist])

print("Rounded Signature weights are (0th shown entry is for Sig^1(W) and so on):",np.round(A,prec))

ypred = np.array( Linear(xtrain) )

plt.plot(ytrain,ytrain)
plt.plot(ypred,ytrain,'o')
plt.title("Predicted increments against actual increments [on TRAINING data]")
plt.show()

S_pred = make_S(ypred,S0,R)


for i in range(R):
    plt.plot(TimePoints,S[i,:,0])
    plt.plot(TimePoints,S_pred[i,:,0],'o',alpha=0.5,markersize=4)
plt.title(str(R)+" risky asset price pathes and recreated price pathes [on TRAINING data].")
plt.show()

train_err = Linear.evaluate(xtrain,ytrain,verbose=0)
print("\nFinal loss:",train_err)

# %%

## The next function samples K path via the signature method
## It is important to note that this samples is based on the
## initial observations S, V alone!
def SigSample(K,dW=None):
    ## Create new Brownian increments if none have been provided
    if dW is None:
        dW = BM_paths(K,N)
    ## Create BM from increments.
    W = BM(dW)
    Sig = signature(W.reshape(K,N+1,1),M=M,T=T)
    x = make_x(Sig,K)
    y_pred = np.array( Linear(x) )
    S = make_S(y_pred,S0,K)
    return S

## The function below compares K path from the signature method sampler
## Against K path from the Euler scheme of the model.
## It creates a plot with the paths
## In a real data application, this function would not be available!
def Compare(K):
    dW = BM_paths(K,N)
    S_Sig = SigSample(K,dW)
    S_path, V_path = path(S0,mu,param,TimePoints,K, dW)
    for j in range(K):
        plt.plot(TimePoints,S_path[j,:,0])
        plt.plot(TimePoints,S_Sig[j,:,0],'o',alpha=0.5,markersize=4)
        plt.title(str(j)+" path from "+name+" Euler scheme (lines) and from signature method (dots)")
        plt.show()

Compare(3)


print("\n\nWe used: "+str(model_name))
print("With parameters: "+str(', '.join(param_name)))
print("Parameter values:",param)


###### End of DeepSimulation ######
