#@author: aditi

import numpy as np 
from scipy import optimize , stats

path=input("enter file path : ")
t=open(path, "r")
data = np.loadtxt(t)                    
t.close()
data=np.transpose(data)
x=data[0]
y=data[1]
xerr=data[2]
yerr=data[3]

#constant model
def fit_const(x,k):
	return k*x**0

#cosine model
def fit_cosine(x,A,w,t_0):
	return A*np.cos(w*(x-t_0))

def cosine(x,theta):
	A = theta[0]
	w = theta[1]
	t_0= theta[2]
	return A*np.cos(w*(x-t_0))

#log of likelihood function
def logL(theta, model, data=data):
    if model==cosine or model==fit_cosine:
        A = theta[0]
        w = theta[1]
        t_0= theta[2]
        sigma=np.zeros(len(x))
        for i in range(len(x)):
            sigma[i]=np.sqrt((data[3,i]**2)+(A*w*np.sin(w*(x[i]-t_0))*data[2,i])**2)
    else:
        sigma=yerr
    y_fit = model(x,theta)
    return sum(stats.norm.logpdf(*args) for args in zip(y, y_fit, sigma))

#chi square value
def chi2_val(theta,model,data=data):
    if model==cosine or model==fit_cosine:
        A = theta[0]
        w = theta[1]
        t_0= theta[2]
        sigma=np.zeros(len(x))
        for i in range(len(x)):
            sigma[i]=np.sqrt((data[3,i]**2)+(A*w*np.sin(w*(x[i]-t_0))*data[2,i])**2)
    else:
        sigma=yerr
    y_fit=model(x,theta)
    r = (y - y_fit)/sigma
    return np.sum(r**2)

#degrees of freedom
def dof_val(theta,data=data):
  return len(x) - len(theta)

#chi squared likelihood function
def chi2L(theta,model,data=data):
  chi2 = chi2_val(theta,model)
  dof = dof_val(theta)
  return stats.chi2(dof).pdf(chi2)



#negative log likelihood function
cos_neg_logL = lambda theta: -logL(theta, cosine, data)
const_neg_logL = lambda k: -logL(k, fit_const, data)

#initial guess
k_init = 0
par_init =[0.0167, 0.0172, 152.5]

#negative likelihood minimization
k_fin = optimize.fmin_bfgs(const_neg_logL, k_init, disp=False)
par_fin = optimize.fmin_bfgs(cos_neg_logL, par_init, disp=False)

A = par_fin[0]
w = par_fin[1]
t = par_fin[2]

c1=chi2_val(par_fin,cosine)
c2=chi2_val(k_fin,fit_const)
d1=dof_val(par_fin)
d2=dof_val(k_fin)

print("\nCosine : Amplitude= ",A,"  w= ",w," /days ","  initial phase= ",t," days")
print("Constant k= ",k_fin)
print("\nCosine :  Chi-Square likelihood:" , chi2L(par_fin,cosine)," ; Chi square value=",c1,"\nConstant :  Chi-Square likelihood:" , chi2L(k_fin,fit_const)," ; Chi square value=",c2)

p1=stats.chi2(d1).sf(c1)
print("cosine ",'dof',d1,'pval',p1,'sigma' ,stats.norm.isf(p1))
p2=stats.chi2(d2).sf(c2)
print("constant ",'dof',d2,'pval',p2,'sigma',stats.norm.isf(p2))

d=np.abs(c1-c2)
print("difference in chi square values = ",d)
p=stats.chi2(2).sf(d)
print ("p value=",p)
print("Confidence level : ",stats.norm.isf(p),'\u03C3','\n')

#=============================================================================

#AIC calculation
aic_const=-2*logL(k_fin,fit_const) + 2
aic_cosine=-2*logL(par_fin,cosine) +2*3
del_aic= np.abs(aic_const-aic_cosine)
print("AIC cosine=",'%.2f'%aic_cosine,"AIC const=",'%.2f'%aic_const)
print ("diff in AIC values = ",'%.2f'%del_aic)


#BIC calculation
bic_const=-2*logL(k_fin,fit_const) + np.log(len(x))
bic_cosine=-2*logL(par_fin,cosine) +3*np.log(len(x))
del_bic= np.abs(bic_const-bic_cosine)
print("BIC cosine=",'%.2f'%bic_cosine,"BIC const=",'%.2f'%bic_const)
print ("diff in BIC values = ",'%.2f'%del_bic)
