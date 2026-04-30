# %%    
######## Deep Simulation code ########
###### Load libraries and set parameters.
import numpy as np
#import tensorflow as tf
import math
from keras.layers import Input, Dense, Add, Dot, TimeDistributed, Flatten
from keras.models import Model
from matplotlib import pyplot as plt
#import scipy.stats as scipy



## Digits shown in some print commands:
prec = 3


#### General stock model setup
S0 = 100 # initial value of the asset.
T = 6/12 # maturity in years. Here "2 month" equal T = 2/12
N = int(T*1500) # time steps. T*1500 is approximately 1 timestep per hour while exchange is open


#### Setup for the DEEPSIMULATION
## Number so signature levels used
# longer maturities need more signature levels (or a more refined method!)
M = 10
## Train/Test setup
# R is supposed to be 1.
R = 1 # number of Trajectories --- this is our training sample

## Number of time steps simulated for newly generated paths
Ntrain = int(T*250) # time steps. T*250 is approximately 1 timestep per day.


## Neural network setup for the DEEPHEDGE
## Train/Test setup
Ktrain = 40000 # Size of training data
Ktest = 2000 # Size of test data
epochs = 15
batch_size = 256
activator = "tanh" ## Activation function to use in the networks
## Network structure
learnV0 = True # Learn setup wealth. If set to False, then it uses the MC estimate as initial wealth
d = 2 # number of hidden layers in strategy
n = 200  # nodes for nodes in hidden layers





#### The code provides 3 stock models
# 1 for Black Scholes model
# 2 for flexible drift/diff function, here CEV
# 3 stochvol model like Heston
# We go with a random choice
Stockmodel = int(np.ceil(3*np.random.sample(1)).item())
Stockmodel = 3

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
    param_range = [ [0.2, 0.8] , [0.001, 0.1] , [0.15, 0.25], [0.01, 0.03], [0.3, 0.7] ]
    useBSformula = False
    param_name = ["volatility", "reversion rate", "long-term volatility", "volvol", "correlation"]
    model_name = "Heston model"

## Number of parameters
m = len(param_range)




## Option profile
strike = 100
## payoff [European call, ie  f_T = max( S(T)-strike, 0) ]
def f(S):
    K_loc, N_loc, d_loc = np.shape(S) ## Read realisations, timepoints and dimension from S
    return(np.maximum(S[:,N_loc-1,0]-strike,0))
    #return(np.maximum(strike-S[:,N,0],0))


# %%    


#### Problem desctiption
## We are given R = 1 path of prices with their volatilites [S, V]
## We like to find a hedge for a European call
## 1 paths is insufficient information!
## We will use the signature method to generate new paths
## We train a deep hedging network on the newly generated paths

## For model evaluation:
## We will generate additional paths via the Euler-Monte Carlo method from our 
## stock price model
## The hedge is evaluated on the newly created paths


# Train on log-prices?
UseLogPrice = True
    

## Network setting for Deep Simulation
# Use quadratic solver?
UseLRsolve = True 
# Training setting if not using quadratic solver
epochs_simulator = 10
batch_size_simulator = 4

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
def TimeConv(t, TTM=True):
    if TTM:
        return T-t  ## The hedge NN expects a time to maturity as information
    return t        ## With this setting, the hedging NN expects time as information

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
    sigma0, eta, sigma, volvol, rho = param
    rho = 1
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

## Showing the parameters
def print_table(param_name, param):
    print("\n\nModel in use: "+"\033[1;3;4m" + str(model_name) + "\033[0m")

    # Print the header
    param = np.round(param, prec)
    print(f"{'Parameter':<20} {'Value':<10}")
    print("-" * 30)

    # Print each parameter and its corresponding value
    for name, value in zip(param_name, param):
        print(f"{name:<20} {value:<10}")



#### Creating R = 1 sample paths from the model ####
dW = BM_paths(R,N)
S, V = path(S0,mu,param,TimePoints,R,dW)

for i in range(R):
   plt.plot(TimePoints,S[i,:,0])
   plt.plot(TimePoints,V[i,:,0],'o',markersize=4)
plt.title(str(R)+" sample paths from "+str(model_name)+" and absolute diff.")
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
    dt = T/(N-1)
    for k in range(M):
        n = k+1
        for l in range(N):
            Sig[:,l,n] = p(n,B[:,l,0],l*dt)
    return Sig



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

Linear = Model(inputs=Sig_input,outputs = output, name="DeepSimulation")
print("\n[Below] Netowrk for the Deep Simulation:")
Linear.summary()


# %%
#### Prepare training data
def make_x(Sig,R,N=N):
    dSig = d_process(Sig)[:,:,1:(M+1)]
    xtrain = dSig.reshape(R*N,M)
    return xtrain

xtrain = make_x(Sig_B,R)

def make_y(S,R,N=N):
    if UseLogPrice:
        dX = d_process(np.log(S))
        ytrain = dX.reshape( (R*N,1) )
    else:
        dS = d_process(S)
        ytrain = dS.reshape( (R*N,1) )
    return ytrain
    
ytrain = make_y(S,R)

def make_S(y,S0,R,N=N):
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

#optimizer = SGD(learning_rate=new_learning_rate)
Linear.compile(optimizer='adam', loss = 'mse')

if UseLRsolve:
    weights, _, _, _ = np.linalg.lstsq(xtrain, ytrain, rcond=None)
    # Set the weights of the model to the linear regression coefficients
    Linear.set_weights([weights])
else:
    optimizer = Linear.optimizer
    optimizer.learning_rate.assign(learning_rate)
    k_count = 5
    for k in range(epochs_simulator):
        if k >= k_count:
            print("Epochs:",k_count,"/",epochs_simulator," current learning rata:",learning_rate)
            k_count += 5
        optimizer.learning_rate.assign(learning_rate)
        learning_rate = learning_rate * learning_rate_decay
        Linear.fit(x=xtrain, y=ytrain, epochs=1, batch_size=batch_size_simulator)
print("Done.")

## Take the model weights into a matrix A
## The model does  ytrain = xtrain A
A = np.array([num for sublist in Linear.get_weights()[0] for num in sublist])

print("Rounded Signature weights are (0th shown entry is for Sig^1(W) and so on):",np.round(A,prec))

ypred = np.array( Linear(xtrain) )

## Shows the fitting of increments. Has been deactivated.
if False:
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
def SigSample(K,N,dW=None,echo=False):
    ## Create new Brownian increments if none have been provided
    if dW is None:
        if echo: print("Creating fresh Bornian motions...")
        dW = BM_paths(K,N)
    ## Create BM from increments.
    W = BM(dW)
    if echo: print("Calculating their signatures...")
    Sig = signature(W.reshape(K,N+1,1),M=M,T=T)
    if echo: print("Predicting stock paths based on signature...")
    x = make_x(Sig,K,N)
    y = np.array( Linear(x) )
    S = make_S(y,S0,K,N)
    if echo: print("Done.")
    return S

## The function below compares K path from the signature method sampler
## Against K path from the Euler scheme of the model.
## It creates a plot with the paths
## In a real data application, this function would not be available!
def Compare(K, N=N):
    dW = BM_paths(K,N)
    S_Sig = SigSample(K,N,dW)
    S_path, V_path = path(S0,mu,param,np.linspace(0,T,N+1),K, dW)
    for j in range(K):
        plt.plot(np.linspace(0,T,N+1),S_path[j,:,0])
        plt.plot(np.linspace(0,T,N+1),S_Sig[j,:,0],'o',alpha=0.5,markersize=4)
    plt.title(str(K)+" paths from "+name+" Euler scheme (lines) and from signature method (dots)")
    plt.show()

Compare(3, N)


print_table(param_name, param)


def TestLoss(K, N=N, echo=False):
    dW = BM_paths(K,N)
    print("") #linebreak
    if echo: print("Simulating ",K," paths from signature.")
    S_Sig = SigSample(K,N,dW)
    if echo: print("Done ... simulating ",K," paths via Euler scheme.")
    S_path, V_path = path(S0,mu,param,np.linspace(0,T,N+1),K, dW)
    if echo: print("Done.")    
    ## Error for every realisation and time point (and dimension)
    S_error = S_path - S_Sig 
    ##
    S_path_error = np.max(np.linalg.norm(S_error, axis=2), axis=1)
    ## RMSE of the pathe error
    print("RMSE in units of money on",K,"test pathes:",np.round( np.std(S_path_error) ,prec))
    #print("Maximal path error on",K,"test pathes:",np.round( np.max(S_path_error) ,prec))

TestLoss(1000,N)

###### End of DeepSimulation ######


# %% 
###### Start of DeepCalibration ######



#### Define the neural networks ####
##
m = 1 # dimension of price


### Definition of neural networks for initial wealth ####
## g( InitialPriceUnderlying ) = PriceOption
d_V0 = 0 ## Number of hidden layers
price0 = Input(shape=(m,))
V0 = price0
for i in range(d_V0):
    V0 = Dense(1, activation=activator)(V0)
V0 = Dense(1, activation='linear', trainable=learnV0)(V0)

pi = Model(inputs=price0, outputs=V0, name="V0")
if learnV0:
    print("\n[Below] Network for the initial wealth:")
    pi.summary()
#we = [1,0]
#pi.setweights(we)


#### Architercure of the network --- Expecting (timeToMaturity,price) vector ####
## g( TimeToMaturity , Price ) = PositionSize
timeprice = Input(shape=(1+m,))

output = timeprice
for i in range(d):
    output = Dense(n,activation=activator)(output)
output = Dense(m, activation='sigmoid')(output)

hedge = Model(inputs=timeprice,outputs = output, name="DeepHedge")
print("\n[Below] Netowrk for the hedging position:")
hedge.summary()


#### Architercure of the wealth network --- expecting a price path
# Reading initial price of the risky asset. 'price' stands for current price.
Obs = Input(shape=(Ntrain,1+m))
Incr = Input(shape=(Ntrain,m))

inputs = [Obs,Incr]

V0 = pi( Obs[:,0,1] )

H = TimeDistributed(hedge)(Obs)

H = Flatten()(H)
Incr = Flatten()(Incr)
Gain = Dot(axes=1)([H,Incr])

wealth = Add()([V0,Gain])

    
## Defining the model: inputs -> outputs
model_wealth = Model(inputs=inputs, outputs=wealth, name="TerminalWealth")
model_wealth.compile(optimizer='adam',loss='mean_squared_error')


print("\n\nNetwork for terminal wealth:") # It is large and one shouldn't look at it!
model_wealth.summary()



# %%
#### Generating training data according to our model and specifications ####
# xtrain consists of the price flow of the risky asset 
trainpathes = SigSample(Ktrain,Ntrain,echo=True)

def shape_inputs(pathes):
    K, N, d = np.shape(pathes)
    N = N - 1
    x = [ np.zeros((K,N,1+d)) ]+ [ np.zeros((K,N,d))]
    for i in range(N):
        x[0][:,i,0] = np.repeat( TimeConv(TimePoints[i]) ,K)
    x[0][:,:,1] = pathes[:,0:N,0]
    x[1][:,:,0] = pathes[:,1:(N+1),0] - pathes[:,0:N,0]
    return x
    
xtrain = shape_inputs(trainpathes)
ytrain = f(trainpathes)


## Set pi network to mean payoff 
V0_train = np.mean(ytrain)
print("\nPre-setting initial wealth for NN-hedge to:",V0_train)
weights_new = [ np.array([[0]]) , np.array([V0_train]) ]
pi.set_weights( weights_new )

# %%
###### Training the model ######
print("\nStart of training ...")
model_wealth.fit(x=xtrain, y=ytrain,epochs=epochs, batch_size=batch_size)
print("Done.")


# %%
###### Testing our model ######
## Building test pathes directly from our model
Ntest = Ntrain ## Deep Hedge is trained on this trading frequency
dW_test = BM_paths(Ktest,Ntest)
S_testpath, V_testpath = path(S0,mu,param,np.linspace(0,T,Ntest+1),Ktest,dW_test)

MC_f = f( path(S0,0,param,np.linspace(0,T,Ntest+1),Ktest,dW_test)[0] )
MC_price = np.mean( MC_f )
MC_stderr = np.std( MC_f ) / np.sqrt(Ktest)

## Creating test data set
xtest = shape_inputs(S_testpath)
ytest = f(S_testpath)  ## Option payoffs

#### Visualisation of results ####
NNtest = model_wealth.predict(xtest,verbose=0)[:,0]   ## Terminal wealth NN
difftest = NNtest - ytest   ## Error in terminal wealth

V0test = pi.predict(S_testpath[0,:],verbose=0)[:,0]   ## Initial wealth NN


print_table(param_name, param)


print("\n\nTest data analysis:")
print("\nStandard deviation (payoffs):",round(np.std(ytest),prec))
print("Mean sample (payoffs):",round(np.mean(ytest),prec))

print("\nMean sample error (NN):",round(np.mean(difftest),prec))
print("Standard deviation of errors (NN):",round(np.std(difftest),prec))

k = 3
print("\nSetup cost for the hedge (NN):",round(V0test[0],prec))
err = k*np.std(ytest)/np.sqrt(Ktest)
print("Price (MC):",round(MC_price,prec),"  with",k,"std. error region: (",round(MC_price-k*MC_stderr,prec),",",round(MC_price+k*MC_stderr,prec),")")


## Comparrison of correct BS-hedge at time 1 and NN hedge at time 1
def Comparehedge(t=0.1):
    for i in range(Ntest):
        if i*T/Ntest <= t:
            k = i
    t = k*T/Ntest
    Svals = S_testpath[:,k,0]
    timeprice = np.concatenate( (np.reshape( np.repeat(TimeConv(TimePoints[k]),Ktest) ,(Ktest,1) ) , np.reshape( Svals ,(Ktest,1) )), axis = 1 )
    h_NN = hedge.predict(timeprice,verbose=0)[:,0]
    plt.plot( Svals, h_NN, 'o')
    content_str = str(round(T-t,prec))
    plt.title("NN hedging position (orange) time to maturity: " + content_str)
    plt.show()

tshow = np.array(T) * np.array( (0.1,0.25,0.5,0.75,0.9) )
for t in tshow:
    Comparehedge(t)



## Plot the realised payoffs
sort_indices = np.argsort(S_testpath[:,Ntest,0])
plt.plot( S_testpath[sort_indices,Ntest,0], ytest[sort_indices])
plt.plot( S_testpath[:,Ntest,0], NNtest, 'o', alpha=0.3, markersize=3)
plt.title("Option payoffs (in blue) and NN terminal wealth (in orange)")
plt.show()

# %%
from scipy.stats import norm

def BS_delta(S, K, T_rem, sigma, r=0.0):
    """Analytical Black-Scholes call delta = N(d1)."""
    S = np.asarray(S, dtype=float)
    if T_rem <= 1e-12:
        return (S > K).astype(float)
    d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T_rem) / (sigma*np.sqrt(T_rem))
    return norm.cdf(d1)

# Pick the BS volatility to compare against, depending on the active model
if Stockmodel == 1:
    sigma_bs = param[0]                  # the BS vol itself
elif Stockmodel == 2:
    sigma_bs = param[0] * S0**(param[1]-1)   # CEV local vol at S=S0
else:
    sigma_bs = param[2]                  # Heston long-term vol

def Comparehedge(t=0.1):
    for i in range(Ntest):
        if i*T/Ntest <= t:
            k = i
    t = k*T/Ntest
    Svals = S_testpath[:, k, 0]
    timeprice = np.concatenate(
        (np.reshape(np.repeat(TimeConv(TimePoints[k]), Ktest), (Ktest, 1)),
         np.reshape(Svals, (Ktest, 1))), axis=1)
    h_NN = hedge.predict(timeprice, verbose=0)[:, 0]

    # Sort by spot for a clean BS-delta line
    order = np.argsort(Svals)
    S_sorted = Svals[order]
    bs_d   = BS_delta(S_sorted, strike, T - t, sigma_bs, r=0.0)

    plt.plot(Svals, h_NN, 'o', color='C0', label='NN hedge', alpha=0.5, markersize=3)
    plt.plot(S_sorted, bs_d, '-', color='orange', label=f'BS delta (σ={sigma_bs:.3f})')
    plt.xlabel('Spot S'); plt.ylabel('Position')
    plt.title(f"NN hedge (blue) vs BS delta (orange) — TTM: {round(T-t, prec)}")
    plt.legend()
    plt.show()

for t in tshow:
    Comparehedge(t)
    
    
    
