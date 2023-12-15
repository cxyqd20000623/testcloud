"""Close Form Formulas for the pricing of  Financial Derivatives.

Most formulas have rather narrow ranges of validity (i.e. they may assume constant interest rates and
a flat dividend yield, etc) so they are  mainly useful in testing the pde calculators. 
"""

from numpy import exp,sqrt,log,power,pi,fabs
import numpy as np
from scipy.special import ndtr
from scipy import integrate
from scipy.optimize import root_scalar


epsilon=1e-12

def bs_price_fwd(isCall, K, T, F, sigma):
    """ Black's pricing formula
    
    European option  forward price as a function of
    the asset's forward.
    
    :param isCall: True for calls , False for Puts
    :type isCall: Boolean
    :param K: option strike
    :param T: option expiry in years
    :param F: forward of the options underlying asset
    :param sigma: underlying's  volatility
    :return: option's forward price
    """
    ds=np.maximum(0.000001,sigma*sqrt(T))
    dsig=0.5*ds*ds
    d2=(log(F/np.maximum(K,epsilon))-dsig)/ds
    d1=d2+ds
    if isCall:
        opt= F*ndtr(d1) - K*ndtr(d2)
    else:
        opt= K*ndtr(-d2) - F*ndtr(-d1)
    return opt

    
class BSPriceTarget:
    def __init__(self,isCall,K,T,F,p):
        self.isCall=isCall
        self.K=K
        self.T=T
        self.F=F
        self.p=p
    def __call__(self,sigma):
        #print(sigma,bs_price_fwd(self.isCall,self.K,self.T,self.F,sigma))
        return  bs_price_fwd(self.isCall,self.K,self.T,self.F,sigma)-self.p

def bs_implied_fwd(isCall, K, T, F, p, tol=1e-8,max_iter=1000, max_sigma=4):
    """ Implied Blacks scholes volatility.
    
    European option implied volatility as a function of  forward option price and
    underlying forward.
    
    :param isCall: True for calls , False for Puts
    :type isCall: Boolean
    :param K: option strike
    :param T: option expiry in years
    :param F: forward of the options underlying asset
    :parm p: forward option price to expiry.
    :param sigma: initial guess of the underlying's  volatility
    :param tol: tolerance, the implied volatility returned by this function when used in :meth:`bs_price_fwd` function will match *p* with at most *tol* error.
    :aram max_sigma: maximum level of volatility to attempt, defaults to 200%
    :return: option's implied volatility
    """
    func=BSPriceTarget(isCall,K,T,F,p)
    sol=root_scalar(func,bracket=[0.0,max_sigma],x0=0.2,xtol=tol,maxiter=max_iter)
    return sol.root

def bs_price(isCall,K,T, spot, rate, div_yield,sigma):
    """ Black-Scholes pricing formula
    
    European option  price as a function of
    the asset's spot

    :param isCall: True for calls , False for Puts
    :type isCall: Boolean
    :param K: option strike
    :param T: option expiry in years
    :param spot: underlying asset spot price
    :param rate: risk free interest rate (assumed `act/365`, continously compounded).
    :param div_yield: asset dividend yield (assumed `act/365`, continously compounded).
    :param sigma: asset  volatility
    :return: option's price
    """
    df=exp(-T*rate)
    F=spot*exp(T*(rate-div_yield))
    return df*bs_price_fwd(isCall,K,T,F,sigma)

def solve_function(func,  y, x,  x0,  x1,tol,max_steps):
    f0=func(x0)-y
    if (fabs(f0)<tol): return x0
    f1=func(x1)-y
    if (fabs(f1)<tol): return x1
    if (f0*f1>0):
       raise Exception(f"solveSS(): values do not  bracket level {y} between f({x0})={y+f0} and f({x1})={y+f1}")
    
    steps=0
    f=func(x)-y
    while fabs(f)>tol: 
        if ((x<=x0) or (x>=x1)):
            x=(x0+x1)/2
            f=func(x)-y    
        if (f*f1<0):
            x0=x
            f0=f
        else:
            x1=x
            f1=f
        m=(f1-f0)/(x1-x0)
        x=x0-f0/m
        f=func(x)-y
        steps+=1
        if (steps==max_steps):
           raise Exception(f"solveSS(): secant method did not converge for level {y}  giving up after {steps} iterations. f({x0})={y+f0} and f({x1})={y+f1}")
    return x

def bs_d1(strike,T,s,rate,div_yield,sigma):
        F=s*exp((rate-div_yield)*T)
        var=max(1e-8,sigma*sigma*T)
        return (log(F/strike)+0.5*var)/sqrt(var)
  
def am_price(isCall,strike,T,spot, rate, div_yield,sigma):
    """ Barone-Adesi and Whaley [Adesi]_ American option pricing formula.
    
    This formula is a small variance quadratic approximation. It is only valid for short maturities 
    or low volatities.

    :param isCall: True for calls , False for Puts
    :type isCall: Boolean
    :param K: option strike
    :param T: option expiry in years
    :param spot: underlying asset spot price
    :param rate: risk free interest rate (assumed `act/365`, continously compounded).
    :param div_yield: asset dividend yield (assumed `act/365`, continously compounded).
    :param sigma: asset  volatility
    :return: option's price


    .. [Adesi] Giovanni Barone-Adesi and Robert E Whaley.
               Efficient analytic approximation of American option values.
               Journal of Finance, 42(2):-20, June 1987.
 
    """
    b=rate-div_yield
    r=rate
    M=2.0*r/sigma/sigma
    N=2.0*b/sigma/sigma
    K=1-exp(-T*r)
    if r!=0:
        mk=M/K
    else:
        mk=2.0/(sigma*sigma*T)
    if isCall:
        q=-(N-1)/2.0+sqrt((N-1)**2+4.*mk)/2.0
        q_inf=-(N-1)/2.0+sqrt((N-1)**2+4.0*M)/2.0
        s0=strike
        s1=strike/max(0.01,1-1/q_inf)
        h=-(b*T+2*sigma*sqrt(T))*(s1-strike)
        s_g=strike+(s1-strike)*(1-exp(h))
        def f(s):
            d1=bs_d1(strike,T,s,rate,div_yield,sigma)
            val=s
            val-=bs_price(isCall,strike,  T   ,s,rate,div_yield,sigma)
            val-=(1-exp(-div_yield*T)*ndtr(d1))*s/q
            return val
        
        s_b=solve_function(f,strike,s_g,s0,s1,1e-2,500)
        if (spot>s_b): return spot-strike
        else:
            d1=bs_d1(strike,T,s_b,rate,div_yield,sigma)
            A=(s_b/q)*(1-exp(-div_yield*T)*ndtr(d1))
            euro_call=bs_price(isCall,strike,T,spot,
                                  rate,div_yield,sigma)
            return euro_call+A*(spot/s_b)**q
    else: # Put option
        q=-(N-1)/2.0-sqrt((N-1)**2+4.*mk)/2.0
        q_inf=-(N-1)/2.0-sqrt((N-1)**2+4.0*M)/2.0
        s0=strike
        s1=strike/(1-1/min(-0.0001,q_inf))
        h=(b*T+2-sigma*sqrt(T))*s1/(strike-s1)
        s_g=s1+(strike-s1)*exp(h)
        def f(s):
            d1=bs_d1(strike,T,s,rate,div_yield,sigma)
            val=s
            val+=bs_price(isCall,strike,  T   ,s,rate,div_yield,sigma)
            val-=(1-exp(-div_yield*T)*ndtr(-d1))*s/q
            return val
        s_b=solve_function(f,strike,s_g,s0,s1,1e-2,500)
        if (spot<s_b): return strike-spot
        else:
            d1=bs_d1(strike,T,s_b,rate,div_yield,sigma)
            A=-(s_b/q)*(1-exp(-div_yield*T)*ndtr(-d1))
            euro_call=bs_price(isCall,strike,T,spot,
                                  rate,div_yield,sigma)
            return euro_call+A*(spot/s_b)**q

_coef = { 
#  X<H    C/P    Dn/Up  In/Out     nu phi  a   b   c   d   e   f
  (True,  True,  True,  True):   (  1,  1, 1, -1,  0,  1,  1,  0), # Down-In Call
  (True,  True,  True,  False):  (  1,  1, 0,  1,  0, -1,  0,  1), # Down-Out Call
  (True,  True,  False, True):   ( -1,  1, 0,  1, -1,  1,  1,  0), # Up-In Call
  (True,  True,  False, False):  ( -1,  1, 1, -1,  1, -1,  0,  1), # Up-Out Call
  (True,  False, True,  True):   (  1, -1, 1,  0,  0,  0,  1,  0), # Down-In Put
  (True,  False, True,  False):  (  1, -1, 0,  0,  0,  0,  0,  1), # Down-Out Put
  (True,  False, False, True):   ( -1, -1, 0,  0,  1,  0,  1,  0), # Up-In Put
  (True,  False, False, False):  ( -1, -1, 1,  0, -1,  0,  0,  1), # Up-Out Put
#  X<H    C/P    Dn/Up  In/Out     nu phi  a   b   c   d   e   f
  (False, True,  True,  True):   (  1,  1, 0,  0,  1,  0,  1,  0), # Down-In Call
  (False, True,  True,  False):  (  1,  1, 1,  0, -1,  0,  0,  1), # Down-Out Call
  (False, True,  False, True):   ( -1,  1, 1,  0,  0,  0,  1,  0), # Up-In Call
  (False, True,  False, False):  ( -1,  1, 0,  0,  0,  0,  0,  1), # Up-Out Call
  (False, False, True,  True):   (  1, -1, 0,  1, -1,  1,  1,  0), # Down-In Put
  (False, False, True,  False):  (  1, -1, 1, -1,  1, -1,  0,  1), # Down-Out Put
  (False, False, False, True):   ( -1, -1, 1, -1,  0,  1,  1,  0), # Up-In Put
  (False, False, False, False):  ( -1, -1, 0,  1,  0, -1,  0,  1), # Up-Out Put
}

def bar_price(isCall,isDown,isIn,K,H,R,T,spot,rate,div_yield,sig):
    """Reiner and Rubinstein's European Barrier option price.

    This function returns the price of a Barrier option with European Excercise.
    It prices both calls and puts with either a single up or down barrier that can be 
    knock-in or knock-out. The rebate *K* is paid inmediately upen touching the barrier
    for a knock-out option  or at maturity for a knock-in option than never reaches the barrier.
    The implementation uses Haug's [HaugBarriers]_ formalization of Reiner and Rubinstein's [RR]_ 
    close form solutions. 
     
    
    
    :param isCall: True for calls , False for Puts
    :type isCall: Boolean
    :param isDown: True for  a down an out barrier, False for an up and in barrier
    :type isDown: Boolean
    :param isIn: True for a Knock-in option, False for a Knock-out option
    :type isIn: Boolean
    :param K: option strike
    :param H: option barrier level
    :param R: option rebate, paid inmediately after touching the barrier for a knockout option, 
              and on expiry for a knockin in than never touches the barrier.
    :param T: option expiry in years
    :param F: forward of the options underlying asset
    :param sigma: underlying's  volatility
    :return: option's forward price

    .. [RR] Reiner, E., & Rubinstein, M., "Exotic Options", Working Paper 1992
    .. [HaugBarriers] Haug, E., "Complete Guide to Option Pricing Formulas", McGraw Hill (1998)

    """
    if (isDown and spot<=H):
        if isIn: return bs_price(isCall,K,T,spot,rate,div_yield,sig)
        else: return R
    elif (not(isDown) and spot>=H):
        if isIn: return bs_price(isCall,K,T,spot,rate,div_yield,sig)
        else: return R

    var=sig*sig
    sqrT=sig*sqrt(T)
    m=(rate-div_yield-0.5*var)/var 
    la=sqrt(m*m+2*(rate-div_yield*0)/var) # probably a mistake the missing div_yield?
    temp1=(1+m)*sqrT
    x1=log(spot*1.0/K)/(sqrT)+temp1
    x2=log(spot*1.0/H)/(sqrT)+temp1
    y1=log((H*H)*1.0/K/spot)/sqrT+temp1
    y2=log(H*1.0/spot)/sqrT+temp1
    z=log(H*1.0/spot)/sqrT+la*sqrT

    nu,phi, a,b,c,d,e,f = _coef[(K<H,isCall,isDown,isIn)]
    
    edt=exp(-div_yield*T)
    ert=exp(-rate*T)
    hs=H*1.0/spot
    N=ndtr
    A=phi*(spot*edt*N(phi*x1)-K*ert*N(phi*(x1-sqrT)))
    B=phi*(spot*edt*N(phi*x2)-K*ert*N(phi*(x2-sqrT)))
    C=phi*(spot*edt*power(hs,2*(m+1))*N(nu*y1)-K*ert*power(hs,2*m)*N(nu*(y1-sqrT)))
    D=phi*(spot*edt*power(hs,2*(m+1))*N(nu*y2)-K*ert*power(hs,2*m)*N(nu*(y2-sqrT)))
    E=R*ert*(N(nu*(x2-sqrT))-power(hs,2*m)*N(nu*(y2-sqrT)))
    F=R*(power(hs,m+la)*N(nu*z)+power(hs,m-la)*N(nu*(z-2*la*sqrT)))
  
    return a*A+b*B+c*C+d*D+e*E+f*F



epsilon=1e-16
def heston_price_fwd(is_call,K,T,F,v,v_inf,lamb,nu,rho):
    x=np.log(F/K)
    tau=T
    i=complex(0,1)
    def Pj(k,alpha,beta):
        gamma=nu*nu/2.0
        d=np.sqrt(beta**2-4*alpha*gamma)
        r_p=(beta+d)/(2.0*gamma)
        r_m=(beta-d)/(2.0*gamma)
        ed=np.exp(-d*tau)
        g=r_m/r_p
        D=r_m*(1-ed)/(1-g*ed)
        C=lamb*(r_m*tau-2.0/nu**2*np.log((1-g*ed)/(1-g)))
        return 1.0/(i*k)*np.exp(C*v_inf+D*v)
    def P0(k):
       alpha=-1.0/2*k*(k+i)
       beta= lamb-rho*nu*i*k
       return Pj(k,alpha,beta)
    def P1(k):
        alpha=-1.0/2*k*(k+i)+i*k
        beta=lamb-rho*nu-rho*nu*i*k
        return Pj(k,alpha,beta)
    class P:
        def __init__(self,p):
            self.p=p
        def __call__(self,k):
            r=self.p(k)*exp(i*x*k)
            return r.real
    p0=P(P0)
    p1=P(P1)
    k=np.linspace(epsilon,100,1000)
    f0=np.empty_like(k)
    f1=np.empty_like(k)
    for i1 in range(len(k)):
        f0[i1]=p0(k[i1])
        f1[i1]=p1(k[i1])
    v0=integrate.quad(p0,0,1000)
    v1=integrate.quad(p1,0,1000)
    c0=v0[0]/pi
    c1=v1[0]/pi
    c=c1*exp(x)-c0
    c=c.real
    if is_call:
        c+=0.5*(exp(x)-1)
    else:
        c-=0.5*(exp(x)-1)
    return K*c