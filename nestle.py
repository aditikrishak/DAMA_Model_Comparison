
#@author: aditi


import numpy as np
import nestle

path=input("enter file path ")    
t=open(path, "r")
data = np.loadtxt(t)                    
t.close()
data=np.transpose(data)
x=data[0]
y=data[1]
xerr=data[2]
yerr=data[3]

#to find A_lim value
if np.abs(np.min(data[1]))>np.abs(np.max(data[1])): A_lim=np.abs(np.min(data[1]))
else: A_lim=np.abs(np.max(data[1]))


#====================FOR CONSTANT MODEL====================================
def const(k, x):
    return k*x**0

# The likelihood function:
def loglike1(k):
    yM = const(k,x)
    return -0.5 * np.sum(np.log(2 * np.pi * yerr ** 2)
                         + (y - yM) ** 2 / yerr ** 2)

# Defines a flat prior
def prior_transform1(k):
    return A_lim*k

def nestle_multi1():
    # Run nested sampling
    res = nestle.sample(loglike1, prior_transform1, 1, method='multi',
                    npoints=2000)
    print(res.summary())

    # weighted average and covariance:
    pm, covm = nestle.mean_and_cov(res.samples, res.weights)

    # re-scale weights to have a maximum of one
    nweights = res.weights/np.max(res.weights)

    # get the probability of keeping a sample from the weights
    keepidx = np.where(np.random.rand(len(nweights)) < nweights)[0]

    # get the posterior samples
    samples_nestle = res.samples[keepidx,:]

    print(np.mean(samples_nestle[:,0]))      # mean of k samples
    print(np.std(samples_nestle[:,0]))       # standard deviation of k samples
    print(len(samples_nestle))              # number of posterior samples
    
    return res.logz
    
Z1=nestle_multi1()

#============================FOR COSINE MODEL===================================

def cosine(x,theta):
	A = theta[0]
	w = theta[1]
	t_0= theta[2]
	return A*np.cos(w*(x-t_0))

# The likelihood function:
def loglike2(theta,data=data):
        A = theta[0]
        w = theta[1]
        t_0= theta[2]
        sigma=np.zeros(len(x))
        for i in range(len(x)):
            sigma[i]=np.sqrt((data[3,i]**2)+(A*w*np.sin(w*(x[i]-t_0))*data[2,i])**2)
        yM= cosine(x,theta)
        return -0.5 * np.sum(np.log(2 * np.pi * sigma ** 2)
                         + (y - yM) ** 2 / sigma ** 2)

# Defines a flat prior for each parameter:
def prior_transform2(theta):
    A = theta[0]
    w = theta[1]
    t_0= theta[2]
    return np.array([A_lim*A , 2*np.pi*w/(365.25) , t_0*2*np.pi/w])

def nestle_multi2():
    # Run nested sampling
    res = nestle.sample(loglike2, prior_transform2, 3, method='multi',
                    npoints=2000)
    print(res.summary())

    # weighted average and covariance:
    pm, covm = nestle.mean_and_cov(res.samples, res.weights)

    # re-scale weights to have a maximum of one
    nweights = res.weights/np.max(res.weights)

    # get the probability of keeping a sample from the weights
    keepidx = np.where(np.random.rand(len(nweights)) < nweights)[0]

    # get the posterior samples
    samples_nestle = res.samples[keepidx,:]
   
    
    print(np.mean(samples_nestle[:,0]))      # mean of A samples
    print(np.std(samples_nestle[:,0])      )# standard deviation of A samples
    print( np.mean(samples_nestle[:,1])    )  # mean of w samples
    print( np.std(samples_nestle[:,1])    )  # standard deviation of w samples
    print( np.mean(samples_nestle[:,2])    )  # mean of t0 samples
    print( np.std(samples_nestle[:,2])    )  # standard deviation of t0 samples
    print(len(samples_nestle))              # number of posterior samples
    
    return res.logz
    
Z2 = nestle_multi2()

Z=np.exp(Z2-Z1)
print ('Bayes factor=',Z)
