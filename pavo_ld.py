"""If we have a complex limb darkening law, we can fit a linear limb darkening law
by matching the Hankel transform at a particular spatial frequency.

This module attempts to accomplish this

TODO

Based on V_from_claret, we need to fit a linear limb darkening coefficient to
x=5.4, based on a more complex law.
"""
from __future__ import division, print_function
import numpy as np
import matplotlib.pyplot as plt
import scipy.special as sp
from scipy import integrate
from scipy import interpolate
from scipy import optimize
from scipy.spatial import Delaunay
from scipy.interpolate import CloughTocher2DInterpolator

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
    
def V_from_claret(xs, ks, cs):
    """Find visibility as a function of x, using the formula from
    Quirrenbach (1996)
    
    Parameters
    ----------
    x: array-like
        Input x ordinate
    """
    if len(ks) != len(cs.T):
        raise UserWarning("Need one coefficient per k")
    try:
        lc = cs.T.shape[1]
    except IndexError:
        lc = 1
    Vs = np.zeros((len(xs),lc))
    norm = np.zeros(lc)
    cs0 = np.append(cs.T,1-np.sum(cs.T,axis=0)).reshape(len(cs.T)+1,lc).T
    ks0 = np.append(ks, 0)
    for k, c in zip(ks0, cs0.T):
        Vs += np.outer(sp.jv(k/2+1,xs)/xs**(k/2+1),c*2**(k/2)*sp.gamma(k/2+1))
        #Vs += c*2**(k/2)*sp.gamma(k/2+1)*sp.jv(k/2+1,xs)/xs**(k/2+1)
        norm += c/(k+2)
    return Vs/norm

def get_elc(ks,cs):
    """Find the equivalent linear coefficient for a given set of coefficients
    
    Returns:
        elc - equivalent linear coefficient
        scl - scale factor for angular diameter
    
    """
    
    if len(ks) != len(cs):
        raise UserWarning("Need one coefficient per k")
    
    # Determine how the maximum visbility in the sidelobe varies with the linear coefficient.
    # Currently by fitting a polynomial to numerically calculated max visibilities.
    # This won't change, so could hardcode the polynomical coefficients instead of recalculating.
    xs = np.arange(5.1,5.8,0.0001)
    #us = np.arange(0,1.001,0.001)
    #
    #mv0s = []
    #
    #for u in zip(us):
    #    vis0 = V_from_claret(xs,[1.0],u)
    #    mv0s.append(vis0.min())
    #mv0s = np.array(mv0s)
    
    #z = np.polyfit(us,mv0s**2,11)
    z = [2.61861994e-06,1.87429278e-05,-6.07351449e-05,9.25521846e-05,-6.72508875e-05,8.32951276e-05,1.48187938e-04,4.58261148e-04,8.69003259e-04,-8.78512128e-05,-1.15292637e-02,1.74978628e-02]
    p = np.poly1d(z)
    
    # Find maximum visibility for the higher-order term law
    cs = np.vstack(cs).T
    
    elcs = []
    scls = []
    
    vis = V_from_claret(xs,ks,cs)
    mv = vis.min(axis=0)
    mx = xs[vis.argmin(axis=0)]
    
    for mvs in mv:
        elc = optimize.brentq(p-mvs**2,0,1)
        elcs.append(elc)
    
    vis0 = V_from_claret(xs,[1.0],np.array([elcs]).T)
    mx0 = xs[vis0.argmin(axis=0)]
    scls = mx0/mx
        
    return elcs, scls

def elc_interp(inteff,inlogg):
    """ Find the equivalent limb-darkening coefficients for a given teff and logg
    
    """
    
    grid = read_regner('PAVO.tab')
    ks = [0.5,1.0,1.5,2.0]
    
    # Triangulate the grid
    loggs = []
    teffs = []
    a1s = []
    a2s = []
    a3s = []
    a4s = []
    for d in grid:
        logg = d['logg']
        teff = d['teff']
        a1 = d['00']['a1']
        a2 = d['00']['a2']
        a3 = d['00']['a3']
        a4 = d['00']['a4']    
        loggs.append(logg)
        teffs.append(teff)
        a1s.append(a1)
        a2s.append(a2)
        a3s.append(a3)
        a4s.append(a4)

    points2D = np.vstack([np.log10(teffs),np.log10(loggs)]).T
    tri = Delaunay(points2D)
    
    #Values to be interpolated to
    t1 = np.log10(inteff)
    l1 = np.log10(inlogg)
    
    # Define wavelength channels
    channels = d.keys()
    channels.remove('teff')
    channels.remove('sigteff')
    channels.remove('logg')
    channels.remove('feh')
    channels.sort()

    wls = []
    elcs = []
    scls = []
    ftcs = []

    # Loop over wavlength channels and perform interpolation
    for c in channels:
        a1s = []
        a2s = []
        a3s = []
        a4s = []
        for d in grid:
            ch = d[c]
            a1s.append(ch['a1'])
            a2s.append(ch['a2'])
            a3s.append(ch['a3'])
            a4s.append(ch['a4'])

#         resLa1 = LinearNDInterpolator(tri,a1s)
#         resLa2 = LinearNDInterpolator(tri,a2s)
#         resLa3 = LinearNDInterpolator(tri,a3s)
#         resLa4 = LinearNDInterpolator(tri,a4s)

        resCTa1 = CloughTocher2DInterpolator(tri,a1s)
        resCTa2 = CloughTocher2DInterpolator(tri,a2s)
        resCTa3 = CloughTocher2DInterpolator(tri,a3s)
        resCTa4 = CloughTocher2DInterpolator(tri,a4s)

#         csL = [resLa1(t1,l1),resLa2(t1,l1),resLa3(t1,l1),resLa4(t1,l1)]
        csCT = [resCTa1(t1,l1),resCTa2(t1,l1),resCTa3(t1,l1),resCTa4(t1,l1)]
        wls.append(ch['wl'])
        ftcs.append(np.asarray(csCT))
#         elc, scl = get_elc(ks,csL)
#         elcs.append(elc)
#         scls.append(scl)
        elc, scl = get_elc(ks,csCT)
        elcs.append(elc)
        scls.append(scl)
        
    return wls,elcs,scls,ftcs


def elc_interp_sig(inteff,insigteff,inlogg,insiglogg,nit):
    """ Find the equivalent limb-darkening coefficients for a given teff and logg and their uncertainties
    
    """
    grid = read_regner('PAVO.tab')
    ks = [0.5,1.0,1.5,2.0]
    
    # Triangulate the grid
    loggs = []
    teffs = []
    for d in grid:
        logg = d['logg']
        teff = d['teff']
        loggs.append(logg)
        teffs.append(teff)

    points2D = np.vstack([np.log10(teffs),np.log10(loggs)]).T
    tri = Delaunay(points2D)
    
    #Values to be interpolated to
    t1 = np.log10(insigteff * np.random.randn(nit) + inteff)
    l1 = np.log10(insiglogg * np.random.randn(nit) + inlogg)
    
    # Define wavelength channels
    channels = d.keys()
    channels.remove('teff')
    channels.remove('sigteff')
    channels.remove('logg')
    channels.remove('feh')
    channels.sort()

    wls = []
    elcs = []
    scls = []
    ftcs = []

    # Loop over wavlength channels and perform interpolation
    for c in channels:
        a1s = []
        a2s = []
        a3s = []
        a4s = []
        for d in grid:
            ch = d[c]
            a1s.append(ch['a1'])
            a2s.append(ch['a2'])
            a3s.append(ch['a3'])
            a4s.append(ch['a4'])

        resCTa1 = CloughTocher2DInterpolator(tri,a1s)
        resCTa2 = CloughTocher2DInterpolator(tri,a2s)
        resCTa3 = CloughTocher2DInterpolator(tri,a3s)
        resCTa4 = CloughTocher2DInterpolator(tri,a4s)

        csCT = [resCTa1(t1,l1),resCTa2(t1,l1),resCTa3(t1,l1),resCTa4(t1,l1)]
        wls.append(ch['wl'])
        ftcs.append(np.asarray(csCT))
        elc, scl = get_elc(ks,csCT)
        elcs.append(elc)
        scls.append(scl)
        
    melcs = np.nanmean(elcs,axis=1)
    sigelcs = np.nanstd(elcs,axis=1)
    mscls = np.nanmean(scls,axis=1)
    sigscls = np.nanstd(scls,axis=1)
    mftcs = np.nanmean(ftcs,axis=2)
    sigftcs = np.nanstd(ftcs,axis=2)
        
    return wls,melcs,sigelcs,mscls,sigscls,mftcs,sigftcs



def read_regner(file):
    """ Read Regner's data file containing 4-term limb-darkening coefficients
        for each pavo wavelength channel    
    """
    
    data = []

    with open(file,'r') as f:
        header = f.readline()
        for line in f:
            line = line.strip()
            columns = line.split()
            model = {}
            model['00'] = {}
            model['01'] = {}
            model['02'] = {}
            model['03'] = {}
            model['04'] = {}
            model['05'] = {}
            model['06'] = {}
            model['07'] = {}
            model['08'] = {}
            model['09'] = {}
            model['10'] = {}
            model['11'] = {}
            model['12'] = {}
            model['13'] = {}
            model['14'] = {}
            model['15'] = {}
            model['16'] = {}
            model['17'] = {}
            model['18'] = {}
            model['19'] = {}
            model['20'] = {}
            model['21'] = {}
            model['22'] = {}
            model['23'] = {}
            model['24'] = {}
            model['25'] = {}
            model['26'] = {}
            model['27'] = {}
            model['28'] = {}
            model['29'] = {}
            model['30'] = {}
            model['31'] = {}
            model['32'] = {}
            model['33'] = {}
            model['34'] = {}
            model['35'] = {}
            model['36'] = {}
            model['37'] = {}
            model['teff'] = float(columns[0])
            model['sigteff'] = float(columns[1])
            model['logg'] = float(columns[2])
            model['feh'] = float(columns[3])
            model['00']['fl'] = float(columns[4])
            model['00']['sigf'] = float(columns[5])
            model['00']['a1'] = float(columns[6])
            model['00']['a2'] = float(columns[7])
            model['00']['a3'] = float(columns[8])
            model['00']['a4'] = float(columns[9])
            model['01']['fl'] = float(columns[10])
            model['01']['sigf'] = float(columns[11])
            model['01']['a1'] = float(columns[12])
            model['01']['a2'] = float(columns[13])
            model['01']['a3'] = float(columns[14])
            model['01']['a4'] = float(columns[15])
            model['02']['fl'] = float(columns[16])
            model['02']['sigf'] = float(columns[17])
            model['02']['a1'] = float(columns[18])
            model['02']['a2'] = float(columns[19])
            model['02']['a3'] = float(columns[20])
            model['02']['a4'] = float(columns[21])
            model['03']['fl'] = float(columns[22])
            model['03']['sigf'] = float(columns[23])
            model['03']['a1'] = float(columns[24])
            model['03']['a2'] = float(columns[25])
            model['03']['a3'] = float(columns[26])
            model['03']['a4'] = float(columns[27])
            model['04']['fl'] = float(columns[28])
            model['04']['sigf'] = float(columns[29])
            model['04']['a1'] = float(columns[30])
            model['04']['a2'] = float(columns[31])
            model['04']['a3'] = float(columns[32])
            model['04']['a4'] = float(columns[33])
            model['05']['fl'] = float(columns[34])
            model['05']['sigf'] = float(columns[35])
            model['05']['a1'] = float(columns[36])
            model['05']['a2'] = float(columns[37])
            model['05']['a3'] = float(columns[38])
            model['05']['a4'] = float(columns[39])
            model['06']['fl'] = float(columns[40])
            model['06']['sigf'] = float(columns[41])
            model['06']['a1'] = float(columns[42])
            model['06']['a2'] = float(columns[43])
            model['06']['a3'] = float(columns[44])
            model['06']['a4'] = float(columns[45])
            model['07']['fl'] = float(columns[46])
            model['07']['sigf'] = float(columns[47])
            model['07']['a1'] = float(columns[48])
            model['07']['a2'] = float(columns[49])
            model['07']['a3'] = float(columns[50])
            model['07']['a4'] = float(columns[51])
            model['08']['fl'] = float(columns[52])
            model['08']['sigf'] = float(columns[53])
            model['08']['a1'] = float(columns[54])
            model['08']['a2'] = float(columns[55])
            model['08']['a3'] = float(columns[56])
            model['08']['a4'] = float(columns[57])
            model['09']['fl'] = float(columns[58])
            model['09']['sigf'] = float(columns[59])
            model['09']['a1'] = float(columns[60])
            model['09']['a2'] = float(columns[61])
            model['09']['a3'] = float(columns[62])
            model['09']['a4'] = float(columns[63])
            model['10']['fl'] = float(columns[64])
            model['10']['sigf'] = float(columns[65])
            model['10']['a1'] = float(columns[66])
            model['10']['a2'] = float(columns[67])
            model['10']['a3'] = float(columns[68])
            model['10']['a4'] = float(columns[69])
            model['11']['fl'] = float(columns[70])
            model['11']['sigf'] = float(columns[71])
            model['11']['a1'] = float(columns[72])
            model['11']['a2'] = float(columns[73])
            model['11']['a3'] = float(columns[74])
            model['11']['a4'] = float(columns[75])
            model['12']['fl'] = float(columns[76])
            model['12']['sigf'] = float(columns[77])
            model['12']['a1'] = float(columns[78])
            model['12']['a2'] = float(columns[79])
            model['12']['a3'] = float(columns[80])
            model['12']['a4'] = float(columns[81])
            model['13']['fl'] = float(columns[82])
            model['13']['sigf'] = float(columns[83])
            model['13']['a1'] = float(columns[84])
            model['13']['a2'] = float(columns[85])
            model['13']['a3'] = float(columns[86])
            model['13']['a4'] = float(columns[87])
            model['14']['fl'] = float(columns[88])
            model['14']['sigf'] = float(columns[89])
            model['14']['a1'] = float(columns[90])
            model['14']['a2'] = float(columns[91])
            model['14']['a3'] = float(columns[92])
            model['14']['a4'] = float(columns[93])
            model['15']['fl'] = float(columns[94])
            model['15']['sigf'] = float(columns[95])
            model['15']['a1'] = float(columns[96])
            model['15']['a2'] = float(columns[97])
            model['15']['a3'] = float(columns[98])
            model['15']['a4'] = float(columns[99])
            model['16']['fl'] = float(columns[100])
            model['16']['sigf'] = float(columns[101])
            model['16']['a1'] = float(columns[102])
            model['16']['a2'] = float(columns[103])
            model['16']['a3'] = float(columns[104])
            model['16']['a4'] = float(columns[105])
            model['17']['fl'] = float(columns[106])
            model['17']['sigf'] = float(columns[107])
            model['17']['a1'] = float(columns[108])
            model['17']['a2'] = float(columns[109])
            model['17']['a3'] = float(columns[110])
            model['17']['a4'] = float(columns[111])
            model['18']['fl'] = float(columns[112])
            model['18']['sigf'] = float(columns[113])
            model['18']['a1'] = float(columns[114])
            model['18']['a2'] = float(columns[115])
            model['18']['a3'] = float(columns[116])
            model['18']['a4'] = float(columns[117])
            model['19']['fl'] = float(columns[118])
            model['19']['sigf'] = float(columns[119])
            model['19']['a1'] = float(columns[120])
            model['19']['a2'] = float(columns[121])
            model['19']['a3'] = float(columns[122])
            model['19']['a4'] = float(columns[123])
            model['20']['fl'] = float(columns[124])
            model['20']['sigf'] = float(columns[125])
            model['20']['a1'] = float(columns[126])
            model['20']['a2'] = float(columns[127])
            model['20']['a3'] = float(columns[128])
            model['20']['a4'] = float(columns[129])
            model['21']['fl'] = float(columns[130])
            model['21']['sigf'] = float(columns[131])
            model['21']['a1'] = float(columns[132])
            model['21']['a2'] = float(columns[133])
            model['21']['a3'] = float(columns[134])
            model['21']['a4'] = float(columns[135])
            model['22']['fl'] = float(columns[136])
            model['22']['sigf'] = float(columns[137])
            model['22']['a1'] = float(columns[138])
            model['22']['a2'] = float(columns[139])
            model['22']['a3'] = float(columns[140])
            model['22']['a4'] = float(columns[141])
            model['23']['fl'] = float(columns[142])
            model['23']['sigf'] = float(columns[143])
            model['23']['a1'] = float(columns[144])
            model['23']['a2'] = float(columns[145])
            model['23']['a3'] = float(columns[146])
            model['23']['a4'] = float(columns[147])
            model['24']['fl'] = float(columns[148])
            model['24']['sigf'] = float(columns[149])
            model['24']['a1'] = float(columns[150])
            model['24']['a2'] = float(columns[151])
            model['24']['a3'] = float(columns[152])
            model['24']['a4'] = float(columns[153])
            model['25']['fl'] = float(columns[154])
            model['25']['sigf'] = float(columns[155])
            model['25']['a1'] = float(columns[156])
            model['25']['a2'] = float(columns[157])
            model['25']['a3'] = float(columns[158])
            model['25']['a4'] = float(columns[159])
            model['26']['fl'] = float(columns[160])
            model['26']['sigf'] = float(columns[161])
            model['26']['a1'] = float(columns[162])
            model['26']['a2'] = float(columns[163])
            model['26']['a3'] = float(columns[164])
            model['26']['a4'] = float(columns[165])
            model['27']['fl'] = float(columns[166])
            model['27']['sigf'] = float(columns[167])
            model['27']['a1'] = float(columns[168])
            model['27']['a2'] = float(columns[169])
            model['27']['a3'] = float(columns[170])
            model['27']['a4'] = float(columns[171])
            model['28']['fl'] = float(columns[172])
            model['28']['sigf'] = float(columns[173])
            model['28']['a1'] = float(columns[174])
            model['28']['a2'] = float(columns[175])
            model['28']['a3'] = float(columns[176])
            model['28']['a4'] = float(columns[177])
            model['29']['fl'] = float(columns[178])
            model['29']['sigf'] = float(columns[179])
            model['29']['a1'] = float(columns[180])
            model['29']['a2'] = float(columns[181])
            model['29']['a3'] = float(columns[182])
            model['29']['a4'] = float(columns[183])
            model['30']['fl'] = float(columns[184])
            model['30']['sigf'] = float(columns[185])
            model['30']['a1'] = float(columns[186])
            model['30']['a2'] = float(columns[187])
            model['30']['a3'] = float(columns[188])
            model['30']['a4'] = float(columns[189])
            model['31']['fl'] = float(columns[190])
            model['31']['sigf'] = float(columns[191])
            model['31']['a1'] = float(columns[192])
            model['31']['a2'] = float(columns[193])
            model['31']['a3'] = float(columns[194])
            model['31']['a4'] = float(columns[195])
            model['32']['fl'] = float(columns[196])
            model['32']['sigf'] = float(columns[197])
            model['32']['a1'] = float(columns[198])
            model['32']['a2'] = float(columns[199])
            model['32']['a3'] = float(columns[200])
            model['32']['a4'] = float(columns[201])
            model['33']['fl'] = float(columns[202])
            model['33']['sigf'] = float(columns[203])
            model['33']['a1'] = float(columns[204])
            model['33']['a2'] = float(columns[205])
            model['33']['a3'] = float(columns[206])
            model['33']['a4'] = float(columns[207])
            model['34']['fl'] = float(columns[208])
            model['34']['sigf'] = float(columns[209])
            model['34']['a1'] = float(columns[210])
            model['34']['a2'] = float(columns[211])
            model['34']['a3'] = float(columns[212])
            model['34']['a4'] = float(columns[213])
            model['35']['fl'] = float(columns[214])
            model['35']['sigf'] = float(columns[215])
            model['35']['a1'] = float(columns[216])
            model['35']['a2'] = float(columns[217])
            model['35']['a3'] = float(columns[218])
            model['35']['a4'] = float(columns[219])
            model['36']['fl'] = float(columns[220])
            model['36']['sigf'] = float(columns[221])
            model['36']['a1'] = float(columns[222])
            model['36']['a2'] = float(columns[223])
            model['36']['a3'] = float(columns[224])
            model['36']['a4'] = float(columns[225])
            model['37']['fl'] = float(columns[226])
            model['37']['sigf'] = float(columns[227])
            model['37']['a1'] = float(columns[228])
            model['37']['a2'] = float(columns[229])
            model['37']['a3'] = float(columns[230])
            model['37']['a4'] = float(columns[231])
            model['37']['wl'] = 0.881000
            model['36']['wl'] = 0.871759
            model['35']['wl'] = 0.862519
            model['34']['wl'] = 0.853278
            model['33']['wl'] = 0.844088
            model['32']['wl'] = 0.835334
            model['31']['wl'] = 0.826603
            model['30']['wl'] = 0.818335
            model['29']['wl'] = 0.810068
            model['28']['wl'] = 0.801800
            model['27']['wl'] = 0.793532
            model['26']['wl'] = 0.785719
            model['25']['wl'] = 0.777938
            model['24']['wl'] = 0.770584
            model['23']['wl'] = 0.763289
            model['22']['wl'] = 0.756394
            model['21']['wl'] = 0.749585
            model['20']['wl'] = 0.742776
            model['19']['wl'] = 0.735968
            model['18']['wl'] = 0.729159
            model['17']['wl'] = 0.722350
            model['16']['wl'] = 0.715860
            model['15']['wl'] = 0.709537
            model['14']['wl'] = 0.703506
            model['13']['wl'] = 0.697670
            model['12']['wl'] = 0.691833
            model['11']['wl'] = 0.685997
            model['10']['wl'] = 0.680398
            model['09']['wl'] = 0.675048
            model['08']['wl'] = 0.669698
            model['07']['wl'] = 0.664348
            model['06']['wl'] = 0.658999
            model['05']['wl'] = 0.653649
            model['04']['wl'] = 0.648454
            model['03']['wl'] = 0.643590
            model['02']['wl'] = 0.638854
            model['01']['wl'] = 0.634477
            model['00']['wl'] = 0.630000
            data.append(model)
            
    return data