# %% 
######## Deep Simulation code ########
###### Load libraries and set parameters.
import numpy as np
#import tensorflow as tf
import math
from keras.layers import Input, Dense, Add, Dot, TimeDistributed, Flatten, Lambda
from keras.models import Model
from matplotlib import pyplot as plt

## Digits shown in some print commands:
prec = 4



S0 = 100 # initial value of the asset.
T = 2/12 # maturity in years. Here "2 month" equal T = 2/12
N = int(T*1500) # time steps. T*1500 is approximately 1 timestep per hour while exchange is open

## Number so signature levels used
# longer maturities need more signature levels (or a more refined method!)
M = 20

## Train/Test setup
# R is supposed to be 1.
R = 1 # number of Trajectories --- this is our training sample


#### The code provides 3 stockm odels
# 1 for Black Scholes model
# 2 for flexible drift/diff function, here CEV
# 3 stochvol model like Heston
# We go with a random choice
Stockmodel = int(np.ceil(3*np.random.sample(1)))
mu = 0.08  ## Q-dynamics: mu=0

## Parameters for model 1, 2 and 3:
if Stockmodel == 1:
    sigma = 0.2 # volatility
    
if Stockmodel == 2:
    sigma = 0.2 # volatility 
    eta = 0.8 ## ellasticity

if Stockmodel == 3:    
    sigma = 0.2 # volatility 
    eta = 0.05 ## Speed of mean reversion.
    rho = 0.3 ## Noise correlation
    volvol = 0.02 ## Volatility of the volatility 'volvol'

    
## payoff [European call, ie  f_T = max( S(T)-strike, 0) ]
strike = 100
def f(S):
    return(np.maximum(S[:,N,0]-strike,0))
    #return(np.maximum(strike-S[:,N,0],0))    


#### Problem desctiption
## We are given R = 1 path of prices with their volatilites [S, V]
## We like to find the "hedge" for the option with payoff f from above.

## Strategy:
## Use DeepSimulation to generate many sample paths
## Use DeepHedging on the many sample paths to find the hedge

## Testing method
## Nr.1: Simulate additional paths via DeepSimulation and use them to test the hedge
## Nr.2: Simulate more paths from the model chosen above and test the hedge on them
## The first strategy we could follow even if [S, V] is coming from a data input
## The second strategy requires that we already know the model.
    
    

## Network setting for Deep Simulation
# Use quadratic solver?
UseLRsolve = False 

# Training setting if not using quadratic solver
epochs = 20
batch_size = 4

learning_rate = 1
learning_rate_decay = 0.8




###### Deep Heding setup

## Train/Test setup
Ktrain = 10000 # Size of training data
epochs_hedge = 10
batch_size_hedge = 128

lr_hedge = 0.001
lr_hedge_decay = 0.9

Ktest = 2000 # Size of test data

#### Network setup for the DeepHedge
activator = "tanh" ## Activation function to use in the networks
## Network structure
learnV0 = True # Learn setup wealth. If set to False, then it uses the MC estimate as initial wealth
L = 2 # number of hidden layers in strategy
n = 200  # nodes for nodes in hidden layers



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
    R, N, d = np.shape(dS); 
    S = np.zeros((R,N+1,d)) + S0
    for j in range(N):
        S[:,j+1,:] = S[:,j,:] + dS[:,j,:]
    return S


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
def path1(S0,mu,sigma,Timepoints,R,dW):
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
def path2(S0,mu,sigma,Timepoints,R,dW):
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
def path3(S0,mu,sigma,Timepoints,R,dW):
    N = len(Timepoints) - 1
    S = np.zeros((R,N+1)) + S0
    A = np.zeros((R,N+1)) + sigma
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
    


#### Creating 1 sample paths from the model ####
dW = BM_paths(R,N)
S, V = path(S0,mu,sigma,TimePoints,R,dW)

for i in range(R):
   plt.plot(TimePoints,S[i,:,0])
   plt.plot(TimePoints,V[i,:,0],'o',markersize=4)
plt.title(str(R)+" sample paths from 'some' model and absolute vols.")
plt.show()



# %%
#### Recaputre Noise increments
## Since  dS = V * dB we should have 
## dB = dS / V

def find_noise(S,V):
    dS = d_process(S)
    dB = dS / V[:,0:N,:]
    B = process(dB,0)
    return B, dB

B, dB = find_noise(S,V)


for i in range(R):
    plt.plot(TimePoints,B[i,:,0])
plt.title(str(R)+" recovered Noise paths (BM with drift).")
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


def signature_simul(B,M=10,T=1):
    R, N, d = np.shape(B)
    dB = d_process(B)
    Sig = np.zeros((R, N, M+1))
    Sig[:,:,0] =  1
    A = np.zeros((M+1,M+1))
    for l in range(M):
        A[l,l+1] = 1
    for n in range(N-1):
        Sig[:,n+1,:] = Sig[:,n+1,:] + np.dot(Sig[:,n,:], A) * dB[:,n]
    return(Sig)

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
    dS = d_process(S)
    ytrain = dS.reshape( (R*N,1) )
    return ytrain
    
ytrain = make_y(S,R)

def make_S(y,S0,R):
    S = np.zeros( (R,N+1,1) ) + S0
    y = y.reshape( (R,N,1) )
    for k in range(N):
        S[:,k+1,:] = S[:,k,:] + y[:,k,:]
    return S



# %%
###### Training the model ######
print("\nStart of training ...")

#optimizer = SGD(learning_rate=new_learning_rate)
Linear.compile(optimizer='adam', loss = 'mse')

print("\nStart of training ...")
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
    S_path, V_path = path(S0,mu,sigma,TimePoints,K, dW)
    for j in range(K):
        plt.plot(TimePoints,S_path[j,:,0])
        plt.plot(TimePoints,S_Sig[j,:,0],'o',alpha=0.5,markersize=4)
    plt.title(str(K)+" paths from "+name+" Euler scheme (lines) and from signature method (dots)")
    plt.show()


###### End of DeepSimulation ######
"""
 We now have a simulator which is solely based on our data [S, V]. 
 We will use the simulator (SigSample) to generatea large number of paths and
 employ a deephedging method to obtain a hedge.
"""
###### Start of DeepHedging ######


#%%
print("\nCreating",Ktrain,"artificial sample paths for training a DeepHedge...")
S_new = SigSample(Ktrain)
print("Done.")


                 
# %%
###### Define the neural networks
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

pi = Model(inputs=price0, outputs=V0)
if learnV0:
    print("\n[Below] Network for the initial wealth:")
    pi.summary()
#we = [1,0]
#pi.setweights(we)


#### Architercure of the network --- Expecting (timeToMaturity,price) vector ####
## g( TimeToMaturity , Price ) = PositionSize
timeprice = Input(shape=(1+m,))

output = timeprice
for i in range(L):
    output = Dense(n,activation=activator)(output)
output = Dense(m, activation='linear')(output)

hedge = Model(inputs=timeprice,outputs = output)
print("\n[Below] Netowrk for the hedging position:")
hedge.summary()




#### Architercure of the wealth network --- expecting a price path
# Reading initial price of the risky asset. 'price' stands for current price.
Obs = Input(shape=(N,1+m))
Incr = Input(shape=(N,m))

inputs = [Obs,Incr]

V0 = pi( Obs[:,0,1] )

H = TimeDistributed(hedge)(Obs)

H = Flatten()(H)
Incr = Flatten()(Incr)
Gain = Dot(axes=1)([H,Incr])

wealth = Add()([V0,Gain])

    
## Defining the model: inputs -> outputs
model_wealth = Model(inputs=inputs, outputs=wealth)
model_wealth.compile(optimizer='adam',loss='mean_squared_error')


print("\n\nNetwork for terminal wealth:") # It is large and one shouldn't look at it!
model_wealth.summary()


# %%
def shape_inputs(pathes):
    K, N, d = np.shape(pathes)
    N = N - 1
    x = [ np.zeros((K,N,1+d)) ]+ [ np.zeros((K,N,d))]
    for i in range(N):
        x[0][:,i,0] = np.repeat( TimeConv(TimePoints[i]) ,K)
    x[0][:,:,1] = pathes[:,0:N,0]
    x[1][:,:,0] = pathes[:,1:(N+1),0] - pathes[:,0:N,0]
    return x
    
xtrain = shape_inputs(S_new)
ytrain = f(S_new)




# %%
###### Training the model ######

## A learning rate scheduler would actually be good here

print("\nStart of training for the DeepHedge ...\n")

## Set pi network to mean payoff 
V0_train = np.mean(ytrain)
print("\nPre-setting initial wealth for NN-hedge to:",V0_train,"\n")
weights_new = [ np.array([[0]]) , np.array([V0_train]) ]
pi.set_weights( weights_new )


optimizer = model_wealth.optimizer
optimizer.learning_rate.assign(lr_hedge)
k_count = 5
for k in range(epochs_hedge):
    if k >= k_count:
        print("Epochs:",k_count,"/",epochs," current learning rata:",lr_hedge)
        k_count += 5
    optimizer.learning_rate.assign(lr_hedge)
    lr_hedge = lr_hedge * lr_hedge_decay
    model_wealth.fit(x=xtrain, y=ytrain, epochs=1, batch_size=batch_size_hedge)

#model_wealth.fit(x=xtrain, y=ytrain,epochs=epochs, batch_size=batch_size)
print("Done.")

NNtrained = model_wealth.predict(xtrain,verbose=0)

plt.plot( ytrain, NNtrained, 'o',)
plt.plot( ytrain, ytrain)
plt.title("Scatter plot, payoffs/NN-hedge (in blue) and ideal line (in orange)\n[on artificial TRAINING data]")
plt.show()


# %%
###### Self-check on the path we can generate ######
print("\nCreating",Ktest,"artificial sample paths for testing.")
dW_test = BM_paths(Ktest,N)
S_test = SigSample(Ktest,dW_test)
print("Done.")

# %%
xtest = shape_inputs(S_test) ## prepare network input
NNtest = model_wealth.predict(xtest,verbose=0)[:,0] ## Terminal wealth NN-hedge

ytest = f(S_test)  ## Option payoffs

difftest = NNtest - ytest   ## Error in terminal wealth

V0test = pi.predict(S_test[0,:],verbose=0)[:,0]   ## Initial wealth NN

print("\n\nTest data analysis (on artificial paths):")
print("\nMean sample (payoffs):",round(np.mean(ytest),prec))
print("Standard deviation (payoffs):",round(np.std(ytest),prec))

print("\nMean sample error (NN):",round(np.mean(difftest),prec))
print("Standard deviation of errors (NN):",round(np.std(difftest),prec))

plt.plot( ytest, NNtest, 'o',)
plt.plot( ytest, ytest)
plt.title("Scatter plot, artificial payoffs/NN-hedge (in blue) and ideal line (in orange)\n[on artificial TEST data]")
plt.show()

print("\nEND OF TEST that can be done in a real situation!")
print("______________________________________________")


# %%
print("\n\nSTART of tests that use model information")
print("\nData was generated by: ->",name,"<-")
print("\nCreating more data")

## Create model path according to model
S_model, V_model = path(S0,mu,sigma,TimePoints, Ktest, dW_test)

## Create option payoffs according to model path
## These are the payoffs we should actually hit!
y_true = f(S_model)

## Creating realised wealth along the true paths
x_true = shape_inputs(S_model) ## prepare network input
NN_Wealth_true = model_wealth.predict(x_true,verbose=0)[:,0] ## Terminal wealth NN-hedge



y_error = ytest - y_true
true_errors = NN_Wealth_true - y_true

plt.hist(true_errors, bins=50)
plt.title("Histogram of errors: NN-wealth minus true payoff")
plt.show()

print("\n\nTest data analysis (on model paths):")
print("\n---> Mean sample (payoffs):",round(np.mean(y_true),prec))
print("---> Standard deviation (payoffs):",round(np.std(y_true),prec))

print("\n---> True mean sample hedging error (NN):",round(np.mean(true_errors),prec))
print("---> True standard deviation of hedging errors (NN):",round(np.std(true_errors),prec))

print("\nMean sample (Artifial paths):",round(np.mean(ytest),prec))
print("Standard deviation (Artificial paths):",round(np.std(ytest),prec))
print("Mean sample error (Artifial paths):",round(np.mean(y_error),prec))
print("Standard deviation of errors (Artificial paths):",round(np.std(y_error),prec))



plt.plot(y_true, NN_Wealth_true, 'o')
plt.plot(y_true,ytest, 'o')
plt.plot(y_true, y_true)
plt.title("Scatter plot, true payoffs/NN-hedge (in blue), true payoffs/artifical payoffs (in orange)\n and ideal line (in green)")
plt.show()
