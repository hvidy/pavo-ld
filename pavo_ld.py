"""If we have a complex limb darkening law, we can fit a linear limb darkening law
by matching the Hankel transform at a particular spatial frequency.

This module accomplishes this.
"""
from __future__ import division, print_function
import numpy as np
import matplotlib.pyplot as plt
import scipy.special as sp
from scipy import integrate
from scipy import interpolate
import pdb
plt.ion()

def hankel_func(r, u, func):
    return func(r)*sp.jv(0,2*np.pi*u*r)*r

def hankel(fs, rs=None, mus=None, us=np.linspace(0,3,100)):
    """Find the Hankel transform of y(r)
    
    Note that mu is the same as r.
    
    FIXME:
    It is also a dodgy formula, because the interpolation should be done
    in mu, not in r
    """
    if rs is None and mus is None:
        raise UserWarning("Must set rs or mus")
    if rs is None:
        rs = np.sqrt(1-mus**2)
    Fs = []
    func = interpolate.interp1d(rs,fs, kind='quadratic')
    for u in us:
        F = 2*np.pi*integrate.quad(hankel_func, rs[0], rs[-1], args=(u,func))[0]
        Fs.append(F)
    Fs = np.array(Fs)
    return us, Fs
    