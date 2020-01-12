#This code calculates the Bayes factor after creating synthetic data containing background+signal and also for synthetic data contaiinng only background
from scipy import stats
from scipy.optimize import curve_fit
import numpy as np
from matplotlib.pylab import plt
import nestle 
np.random.seed(0)


x=np.arange(0,1840,10)
# t is center of each time bin 
t=0.5*(x[1:]+x[:-1])
print (len(t))

y=np.random.normal(0,0.035,len(t))
erry=np.random.normal(0.031,0.000175,len(t))
xerr=10*np.ones(len(t))
s=0.02*np.cos((2*np.pi/365.0)*(t-156))
# Add signal to the background. If you want synthetic data consisting of only background then the next line is not needed.
y=y+s



def cosine(x,theta):
    A=theta[0]
    t_0=theta[1]
    w=2*np.pi/365.0
    return A*np.cos(w*(x-t_0))

def loglike2(theta):
    yM=cosine(t1,theta)
    return -0.5 * np.sum(np.log(2 * np.pi * erry1 ** 2)
                         + (y1 - yM) ** 2 / erry1 ** 2)
def prior_transform2(theta):
    A = theta[0]
    t_0= theta[1]
    A_lim=0.3
    return np.array([A_lim*(2*A -1),  t_0*365])


def const(k):
    return k*(t1**0)


def loglike1(k):
    yM = const(k)
    return -0.5 * np.sum(np.log(2 * np.pi * erry1 ** 2)
                         + (y1 - yM) ** 2 / erry1 ** 2)
def prior_transform1(k):
    A_lim=0.3
    return A_lim*(2*k-1)


def nestle_multi1():
    # Run nested sampling
    res = nestle.sample(loglike1, prior_transform1, 1, method='multi',
                    npoints=2000)

    # weighted average and covariance:
    pm, covm = nestle.mean_and_cov(res.samples, res.weights)

    # re-scale weights to have a maximum of one
    nweights = res.weights/np.max(res.weights)

    # get the probability of keeping a sample from the weights
    keepidx = np.where(np.random.rand(len(nweights)) < nweights)[0]

    # get the posterior samples
    samples_nestle = res.samples[keepidx,:]

    return res.logz
    


def nestle_multi2():
    # Run nested sampling
    res = nestle.sample(loglike2, prior_transform2, 2, method='multi',
                    npoints=2000)

    # weighted average and covariance:
    pm, covm = nestle.mean_and_cov(res.samples, res.weights)

    # re-scale weights to have a maximum of one
    nweights = res.weights/np.max(res.weights)

    # get the probability of keeping a sample from the weights
    keepidx = np.where(np.random.rand(len(nweights)) < nweights)[0]

    # get the posterior samples
    samples_nestle = res.samples[keepidx,:]

    return res.logz
    


t1=t
y1=y
erry1=erry
Z1=nestle_multi1()
print (Z1)
Z2=nestle_multi2()
print (Z2)
print (np.exp(Z2-Z1))


