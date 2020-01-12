# This code generates simulated background and Cosine signal for a mock dark matter experiment and tests for background and calculates frequentist p-value, AIC, BIC. Bayes factor calculation in a different script 
from scipy import stats
from scipy.optimize import curve_fit
import numpy as np
from matplotlib.pylab import plt
np.random.seed(0)
#dist=stats.poisson(2.7)
#1830 = 5 years. You need one more
#r=dist.rvs(1830)
#generate 5 years worth of time data
x=np.arange(0,1840,10)
# t is center of each time bin 
t=0.5*(x[1:]+x[:-1])
# y is the binned average of r in every 10 day bin.
#y=[np.mean(r[p:p+9]) for p in x[:-1]]
#erry= error in y. obtained from median error/error from Anais count rate data
#erry=np.abs(y)*0.01
# do a mean subtraction
#y=y-np.mean(y)
y=np.random.normal(0,0.035,len(t))
erry=np.random.normal(0.031,0.000175,len(t))
xerr=10*np.ones(len(t))
#print xerr
s=0.01*np.cos((2*np.pi/365.0)*(t-156))
y=y+s
#model 2 : 
#erry=np.abs(y)*1.45
#print erry.shape
#print y.shape
#plt.plot(t,y,'o')
#plt.errorbar(t, y,yerr=erry,fmt='.k')
#plt.plot(t,y,'r.')


def fit_cosine(x,C,A,t_0):
    w=2*np.pi/365.0
    return C+A*np.cos(w*(x-t_0))

# year 5 results

def results(t,y,err): 
    p0=np.array([0.35,0.1,100])
    popt,pcov =curve_fit(fit_cosine,t,y,sigma=err)
#    plt.plot(t,fit_cosine(t,popt[0],popt[1]),'m')
#    plt.show()
    resid= (y-fit_cosine(t,popt[0],popt[1]))/err
    chi_cosine=np.sum(resid**2)
    print (chi_cosine/(len(t)-2))
    ymean=np.sum(y/err**2)/np.sum(1/err**2)
    resid2=(y-ymean)/err
    chi_const=np.sum(resid2**2)
    aic_const=chi_const + 1
    aic_cosine=chi_cosine+2
    print (" chi2/ DOF for constant model", chi_const,"/",len(t)-1)
    print( " chi2/ DOF for cosine model", chi_cosine,"/",len(t)-2)
    print ("chi2 pdf for constant model",stats.chi2(len(t)-1).pdf(chi_const))
    print ("chi2 pdf for cosine model",stats.chi2(len(t)-2).pdf(chi_cosine))
    print ("delta chi-square between the two models",chi_cosine-chi_const)
    p=stats.chi2(1).sf(np.abs(chi_const-chi_cosine))
    print ("p-value =",p)
    print ("Z-score=",stats.norm.isf(p))
    print ("AIC for constant model",aic_const) 
    print ("AIC for cosine model", aic_cosine)
    print ("difference in AIC for cosine and constant",aic_const-aic_cosine)
    bic_const=chi_const+np.log(len(t))
    bic_cosine=chi_cosine+2*np.log(len(t))
    print ("BIC for constant model",bic_const) 
    print ("BIC for cosine model", bic_cosine)
    print ("difference in BIC for cosine and constant",bic_const-bic_cosine)
    p=stats.chi2(1).sf(np.abs(chi_const-chi_cosine))
    return None
print( "year 5 results")
#For year 1 results use
t1=t[0:38]
y1=y[0:38]
err1=erry[0:38]
#t1=t[0:110]
#y1=y[0:110]
#err1=erry[0:110]
results(t1,y1,err1)
