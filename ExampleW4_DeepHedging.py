# %% 
######## Deep Hedging code ########
###### Load libraries and set parameters.
import numpy as np
#import tensorflow as tf

from keras.layers import Input, Dense, Add, Dot, TimeDistributed, Flatten

from keras.models import Model

import matplotlib.pyplot as plt


# Trajectories of the Black scholes model
# Let it run to initialize the following parameters, the trajectories 
# are not needed afterwards

N = 20 # time disrectization
S0 = 150 # initial value of the asset in multiples of np.exp(5)
strike = 150 # strike for the call option
T = 2 # maturity
sigma = 0.2 # volatility 
mu = 0.02  ## excess return --- Q-dynamics: mu=0
R = 3 # number of shown Trajectories --- not used for training/testing


## Train/Test setup
Ktrain = 40000 # Size of training data
Ktest = 2000 # Size of test data
epochs = 20
batch_size = 128
activator = "tanh" ## Activation function to use in the networks


## Network structure
learnV0 = True # Learn setup wealth. If set to False, then it uses the MC estimate as initial wealth
d = 2 # number of hidden layers in strategy
n = 200  # nodes for nodes in hidden layers


## Digits shown in some print commands:
prec = 2

## Volatility assumed for the BS-hedge
sigma_BS = sigma

## The code provides 3 stockmodels
# 1 for Black Scholes model
# 2 for flexible drift/diff function, here CEV
# 3 stochvol model like Heston
# 4 mixture model of 1 to 3
Stockmodel = 1




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
     # return t  ## The NN expects actual time as information. The results are the same as 'time to maturity'
                 ## but the information will be stored differently in the NN.


#### Defining the price model ####
## BS model
def path1(S0,mu,sigma,Timepoints,R):
    N = len(Timepoints) - 1
    X = np.zeros((R,N+1)) + np.log(S0) ## the last 1 is for the dimension of the model
    mu_log = mu - sigma**2/2
    for j in range(N):
        dt = Timepoints[j+1] - Timepoints[j]
        dZ = np.random.normal(0,1,R)
        increment = mu_log * dt + sigma * dZ * np.sqrt(dt)
        X[:,j+1] = X[:,j] + increment
    S = np.exp(X)
    return np.reshape(S,(R,N+1,1))

## A CEV-type model
def path2(S0,mu,sigma,Timepoints,R):
    N = len(Timepoints) - 1
    S = np.zeros((R,N+1)) + S0
    # drift function
    eta = 0.8 ## CEV rate
    def beta(s):
        return mu*(s**eta)
    # volatility function
    def a(s):
        return sigma*(s**eta)
    for j in range(N):
        dt = Timepoints[j+1] - Timepoints[j]
        dZ = np.random.normal(0,1,R)
        increment = beta(S[:,j])*dt + a(S[:,j]) * dZ * np.sqrt(dt)
        S[:,j+1] = S[:,j] + increment
    return np.reshape(S,(R,N+1,1))

## A Heston-type model
def path3(S0,mu,sigma,Timepoints,R,ret_both=False):
    N = len(Timepoints) - 1
    S = np.zeros((R,N+1)) + S0
    A = np.zeros((R,N+1)) + sigma
    rho = 0.3 ## Noise correlation
    eta = 0.05 ## Speed of mean reversion for volatility
    volvol = 0.02 ## Volatility of the volatility A
    for j in range(N):
        dt = Timepoints[j+1] - Timepoints[j]
        dZ = np.random.normal(0,1,(R,2))
        dZ[:,1] = rho*dZ[:,0] + np.sqrt(1-rho**2)*dZ[:,1]
        increment_S = mu*S[:,j]*dt + A[:,j] * S[:,j]*dZ[:,0]*np.sqrt(dt)
        increment_A = eta*(sigma-A[:,j])*dt + volvol * dZ[:,1]*np.sqrt(dt)
        S[:,j+1] = S[:,j] + increment_S
        A[:,j+1] = A[:,j] + increment_A
    if ret_both:
        return S,A
    return np.reshape(S,(R,N+1,1))

## A mixture type model
def path4(S0,mu,sigma,Timepoints,R):
    props = [0.2,0.4,0.4]
    select = np.random.choice([0,1,2], size=R, p=props)
    S = np.zeros((R,N+1,1))
    R = 0
    L = 0
    for i in range(3):
        R = sum(select == i)
        if i == 0:
            S[0:R] = path1(S0,mu,sigma,Timepoints,R)
        if i == 1:    
            S[L:(L+R)] = path2(S0,mu,sigma,Timepoints,R)
        if i == 2:    
            S[L:(L+R)] = path3(S0,mu,sigma,Timepoints,R)
        L = L + R
    return S

## Select a model
path = path1
if Stockmodel == 2:
    path = path2
if Stockmodel == 3:
    path = path3
if Stockmodel == 4:
    path = path4
    


#### Creating R sample paths from the model ####
S = path(S0,mu,sigma,TimePoints,R)

for i in range(R):
   plt.plot(TimePoints,S[i,:,0])
plt.title(str(R)+" sample paths from our model.")
plt.show()

# %%
#### Defining the payoff as well as implementing the BS-formula for it
import scipy.stats as scipy
#from scipy.stats import norm

# Black Scholes price formula (European call)
def BS(S0, strike, T, sigma=sigma_BS):
    return S0*scipy.norm.cdf((np.log(S0/strike)+0.5*T*sigma**2)/(np.sqrt(T)*sigma))-strike*scipy.norm.cdf((np.log(S0/strike)-0.5*T*sigma**2)/(np.sqrt(T)*sigma))
# Black Scholes Delta hedge
def deltaBS(S0, strike, T, sigma=sigma_BS):
    return scipy.norm.cdf((np.log(S0/strike)+0.5*T*sigma**2)/(np.sqrt(T)*sigma))

# Computes terminal wealth of the BS-delta hedge for a call on a trajectory
def terminalWealth_BS(S,sigma=sigma_BS,V=BS(S0,strike,T,sigma_BS)):
    for k in range(S.shape[1]-1):
        V = V + deltaBS(S[:,k], strike, (N-k)*T/N ,sigma) * (S[:,k+1]-S[:,k])
    return V

## payoff [European call, ie  f_T = max( S(T)-strike, 0) ]
def f(S):
    return(np.maximum(S[:,N,0]-strike,0))
    #return(np.maximum(strike-S[:,N,0],0))

priceBS=BS(S0,strike,T,sigma)
print('BS-Price of a Call option in the Black scholes model with initial price', round(S0,prec), 'strike', strike, 'maturity', T , 'and volatility' , sigma, 'is equal to', round(BS(S0,strike,T,sigma),prec))



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
for i in range(d):
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
#### Generating training data according to our model and specifications ####
# xtrain consists of the price flow of the risky asset 
trainpathes = path(S0, mu,sigma, TimePoints, Ktrain)

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
## Building test pathes
testpathes = path(S0, mu,sigma, TimePoints, Ktest)

## Creating test data set
xtest = shape_inputs(testpathes)
ytest = f(testpathes)  ## Option payoffs


#### Visualisation of results ####
NNtest = model_wealth.predict(xtest,verbose=0)[:,0]   ## Terminal wealth NN
difftest = NNtest - ytest   ## Error in terminal wealth

BStest = terminalWealth_BS(testpathes, sigma_BS)[:,0] ## Black SCholes terminal wealth

V0test = pi.predict(testpathes[0,:],verbose=0)[:,0]   ## Initial wealth NN
V0correct = BS(testpathes[:,0],strike,T,sigma_BS)   ## Initial wealth BS
V0diff = np.mean( V0test - V0correct )   ## mean error of initial wealth








print("\n\nTest data analysis:")
print("\nStandard deviation (payoffs):",round(np.std(ytest),prec))
print("Mean sample (payoffs):",round(np.mean(ytest),prec))

print("\nMean sample error (NN):",round(np.mean(difftest),prec))
print("Standard deviation of errors (NN):",round(np.std(difftest),prec))

print("\nMean sample error (BS):",round(np.mean(ytest-BStest),prec))
print("Standard deviation of errors (BS):",round(np.std(ytest-BStest),prec))

k = 3
print("\nSetup cost (BS):",round(V0correct[0,0],prec))
print("Setup cost (NN):",round(V0test[0],prec))
err = k*np.std(ytest)/np.sqrt(Ktest)
print("Expected payoff (MC):",round(np.mean(ytest),prec),"  with",k,"std. error region: (",round(np.mean(ytest)-err,prec),",",round(np.mean(ytest)+err,prec),")")
#print("Setup cost (NN minus BS):",round(V0diff,prec))


## Comparrison of correct BS-hedge at time 1 and NN hedge at time 1
def Comparehedge(t=0,showBS=True,sameArea=False):
    for i in range(N):
        if TimePoints[i] <= t:
            k = i
    t = TimePoints[k]
    Svals = testpathes[:,k,0]
    if sameArea:
        a = np.min(testpathes)
        b = np.max(testpathes)
        Svals = np.linspace(a,b,Ktest)
    h_BS = deltaBS(Svals,strike,T-t,sigma_BS)
    timeprice = np.concatenate( (np.reshape( np.repeat(TimeConv(TimePoints[k]),Ktest) ,(Ktest,1) ) , np.reshape( Svals ,(Ktest,1) )), axis = 1 )
    h_NN = hedge.predict(timeprice,verbose=0)[:,0]
    if showBS:
        plt.plot( Svals, h_BS, 'o')
    plt.plot( Svals, h_NN, 'o')
    content_str = str(round(T-t,prec))
    plt.title("BS hedging position (blue) vs NN hedging position (orange) time to maturity: " + content_str)
    plt.show()

tshow = np.array(T) * np.array( (0.1,0.25,0.5,0.75,0.9) )
for t in tshow:
    Comparehedge(t,True,False)

plt.plot( ytest, BStest, 'o', ytest, NNtest, 'o',)
plt.plot( ytest, ytest)
plt.title("Scatter plot, payoffs/NN-hedge (in orange) and \npayoffs/BS-hedge (in blue), ideal line (in green)")
plt.show()


## Plot the realised payoffs
sort_indices = np.argsort(testpathes[:,N,0])
plt.plot( testpathes[sort_indices,N,0], ytest[sort_indices])
plt.plot( testpathes[:,N,0], BStest, 'o', alpha=0.5)
plt.plot( testpathes[:,N,0], NNtest, 'o', alpha=0.75)
plt.title("Option payoffs (in blue), BS terminal wealth (in orange) \n and NN terminal wealth (in green)")
plt.show()