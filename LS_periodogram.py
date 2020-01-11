import numpy as np
from scipy import optimize 
from astropy.timeseries import LombScargle
import matplotlib.pyplot as plt 
plt.style.use('ggplot')

def fit_cosine(x,A,w,t_0):
	return A*np.cos(w*(x-t_0))


def residuals(x,y,xerr,yerr):
    par_init =[0.0167, 0.0172, 152.5]
    [par_fin , sig] = optimize.curve_fit(fit_cosine, x, y, par_init)
    A,w,t = par_fin[0], par_fin[1], par_fin[2]
    dA , dw , dt = sig[0,0],sig[1,1],sig[2,2]
    """
    DAMA values:
    A, w, t = 0.0103, 2.*np.pi/(0.999*365.25), 145
    dA, dw, dt = 0.0008, w*0.001/(0.999*365.25), 5
    """
    r = (y - fit_cosine(y,A,w,t))

    #taking error only in x and y
    sig = (np.sqrt( (yerr)**2  + (-xerr*A*w*np.sin(w*(x-t)))**2 ))
    
    return r, sig

def LS_plot(file,x,y,xerr,yerr,n): 
    fmin=1e-3
    fmax=18
    #fmax=1/np.median(xerr)
    #fmax=1/np.mean(xerr)

    frequency, power = LombScargle(x, y, dy=yerr, nterms=n).autopower(minimum_frequency=fmin, maximum_frequency=fmax)

    plt.plot(frequency, power)
    #plt.xlim(0,2.5) 
    plt.xlabel('Frequency (per year)',fontsize=15)
    plt.ylabel('Lomb-Scargle Power',fontsize =15)
    plt.tick_params(axis='both',labelsize=15)
    plt.grid(color='w')
    plt.minorticks_on()
    plt.grid(which='minor', linestyle='--', linewidth='0.5', color='white')
    plt.show()

def LS (file):
    t=open(file+'.txt', "r")
    data = np.loadtxt(t)                    
    t.close()
    data=np.transpose(data)
    x=data[0]
    y=data[1]
    xerr=data[2]
    yerr=data[3]
    print(file+' : mean=','%.2f'%np.mean(xerr),'  , median=','%.2f'%np.median(xerr),' ,  max=','%.2f'%np.max(xerr))
    #for data 
    LS_plot (file, x/365.25 , y ,xerr/365.25 , yerr , 2)
     
    # for residuals
    resd ,sig_resd = residuals(x,y,xerr,yerr)
    LS_plot (file, x/365.25 , resd ,xerr/365.25, sig_resd , 1)

#============================================================================
    
LS('2-6keV')

#LS('2018_1-6keV')

#============================================================================
