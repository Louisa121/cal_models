#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 8 10:10:41 2021

@author: Luzhang

import modul for optical models

"""


import os
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pylab as pltl
import matplotlib.dates as mdate
from scipy import stats
import math
import PyMieScatt as ps
from scipy.integrate import trapz
from matplotlib.ticker import LogFormatter 
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,NullFormatter,ScalarFormatter,
                               AutoMinorLocator)



def CS_2D(BC_dis,coating_new,mCore,mShell,dlogdbc,wavelength,ams):
 
    dShell = np.arange(0,222,2)
    Abs = np.zeros(len(BC_dis.index))
    Sca = np.zeros(len(BC_dis.index))
    Ext = np.zeros(len(BC_dis.index))
    _length = np.size(dShell)
    _length_B = np.size(coating_new.rBCCore_dia_nm)
    

    for k in range(len(BC_dis.index)):
    #     print(k)
        Q_ext = np.zeros(_length)
        Q_sca = np.zeros(_length)
        Q_abs = np.zeros(_length)
        Q_pr = np.zeros(_length)
        Q_back = np.zeros(_length)
        Q_ratio = np.zeros(_length)
        g = np.zeros(_length)
        aSDn = np.zeros(_length)
        Bext= np.zeros((_length_B))
        Bsca= np.zeros((_length_B))
        Babs= np.zeros((_length_B))
        Bback= np.zeros((_length_B))
        Bratio= np.zeros((_length_B))
        bigG= np.zeros((_length_B))
        Bpr = np.zeros((_length_B))

        for i in coating_new.index:
            Dc = coating_new.rBCCore_dia_nm[i]
            coating_Dc = coating_new.iloc[i,3:]
            coating_Dc_sum = np.sum(coating_Dc)
            for j in range(len(dShell)):  
                Pj = coating_Dc[j]/coating_Dc_sum if coating_Dc_sum>0 else 0 
                                                                # number fraction of Dc with coating_Dc coating
                Dp = Dc+2*dShell[j]
                aSDn[j] = np.pi*((Dp/2)**2)*BC_dis.iloc[k,2+i]*Pj*(1e-6) # area * N
                if aSDn[j]>0:
                    Q_ext[j],Q_sca[j],Q_abs[j],g[j],Q_pr[j],Q_back[j],Q_ratio[j]=ps.MieQCoreShell(mCore, mShell.iloc[k,0],wavelength,Dc,Dp)    
                                                                # Wavelength and particle diameters: nanometers, 
                                                                # efficiencies: unitless, cross-sections: nm2, 
                                                                # coefficients: Mm-1, 
                                                                # size distribution concentration: cm-3
                else:
                    Q_ext[j] = 0
                    Q_sca[j] = 0
                    Q_abs[j] = 0
            Bext[i] = np.sum(Q_ext*aSDn)
            Bsca[i] = np.sum(Q_sca*aSDn)
            Babs[i] = Bext[i]-Bsca[i]
    #         Bback[i] = np.sum(Q_back*aSDn)
    #         Bratio[i] = np.sum(Q_ratio*aSDn)
    #         bigG[i] = np.sum(g*Q_sca*aSDn)/trapz(Q_sca*aSDn)


        Ext[k] = np.nansum(Bext*dlogdbc.values)
        Abs[k] = np.nansum(Babs*dlogdbc.values)
        Sca[k] = Ext[k]-Abs[k]
    cal_coef_cs = pd.DataFrame([Ext,Sca,Abs]).T
    cal_coef_cs.columns = ['Ext','Sca','Abs']
    cal_coef_cs.index = ams.index
    return(cal_coef_cs)


def Mie_MG(psd,psd_BC,mCore,mShell,logdp,wavelength,ams):

    dp = psd.columns.astype('float').values                  
    logdp = np.log10(dp)
    _length = np.size(dp)
    Q_ext = np.zeros(_length)
    Q_sca = np.zeros(_length)
    Q_abs = np.zeros(_length)
    Q_pr = np.zeros(_length)
    Q_back = np.zeros(_length)
    Q_ratio = np.zeros(_length)
    g = np.zeros(_length)
    Bext= np.zeros(len(ams))
    Bsca= np.zeros(len(ams))
    Babs= np.zeros(len(ams))
    Bback= np.zeros(len(ams))
    Bratio= np.zeros(len(ams))
    bigG= np.zeros(len(ams))
    Bpr = np.zeros(len(ams))
    R = {}
    cal_coefi = {}
    perm1 = mCore**2
    
    for k in range(len(mShell.columns)):
        for i in range(len(ams)):
            perm2 = mShell.iloc[i,k]**2
            aSDn = np.pi*((dp/2)**2)*psd.iloc[i,:]*(1e-6)
            for j,Dpj in enumerate(dp):
                f1 = (psd_BC.iloc[i,j]/psd.iloc[i,j])**3 if psd.iloc[i,j]>0 else 0
                m_MGj = ((perm1+2*perm2+2*f1*(perm1-perm2))/(perm1+2*perm2-f1*(perm1-perm2))*perm2)**0.5
                Q_ext[j],Q_sca[j],Q_abs[j],g[j],Q_pr[j],Q_back[j],Q_ratio[j] = ps.MieQ(m_MGj, wavelength, Dpj)    
                                                                # Wavelength and particle diameters: nanometers, 
                                                                # efficiencies: unitless, cross-sections: nm2, 
                                                                # coefficients: Mm-1, 
                                                                # size distribution concentration: cm-3
            Bext[i] = trapz(Q_ext*aSDn,logdp)
            Bsca[i] = trapz(Q_sca*aSDn,logdp)
            Babs[i] = Bext[i]-Bsca[i]
            Bback[i] = trapz(Q_back*aSDn,logdp)
            Bratio[i] = trapz(Q_ratio*aSDn,logdp)
            bigG[i] = trapz(g*Q_sca*aSDn,logdp)/trapz(Q_sca*aSDn,logdp)
            Bpr[i] = Bext[i] - bigG[i]*Bsca[i]
        cal_coefi[k] = pd.DataFrame([Bext,Bsca,Babs],index=['Ext', 'Sca', 'Abs']).T
    if len(cal_coefi) > 1:
        cal_coef = pd.concat([cal_coefi[0],cal_coefi[1]],axis=1)
        for k in range(2,len(mShell.columns)):
            cal_coef = pd.concat([cal_coef,cal_coefi[k]],axis=1)
    else:
        cal_coef = cal_coefi[0]
    cal_coef.index = ams.index
    return cal_coef




def Mie_BG(psd,psd_BC,mCore,mShell,logdp,wavelength,ams):

    dp = psd.columns.astype('float').values                  
    logdp = np.log10(dp)
    _length = np.size(dp)
    Q_ext = np.zeros(_length)
    Q_sca = np.zeros(_length)
    Q_abs = np.zeros(_length)
    Q_pr = np.zeros(_length)
    Q_back = np.zeros(_length)
    Q_ratio = np.zeros(_length)
    g = np.zeros(_length)
    Bext= np.zeros(len(ams))
    Bsca= np.zeros(len(ams))
    Babs= np.zeros(len(ams))
    Bback= np.zeros(len(ams))
    Bratio= np.zeros(len(ams))
    bigG= np.zeros(len(ams))
    Bpr = np.zeros(len(ams))
    R = {}
    cal_coefi = {}
    perm1 = mCore**2
    
    for k in range(len(mShell.columns)):
        for i in range(len(ams)):
            perm2 = mShell.iloc[i,k]**2
            for j,Dpj in enumerate(dp):
                f1 = (psd_BC.iloc[i,j]/psd.iloc[i,j])**3 if psd.iloc[i,j]>0 else 0
                bb = 3*f1*(perm1-perm2)+2*perm2-perm1
                m_BGj = ((bb+(bb**2+8*perm1*perm2)**.5)/4)**.5
                Q_ext[j],Q_sca[j],Q_abs[j],g[j],Q_pr[j],Q_back[j],Q_ratio[j] = ps.MieQ(m_BGj, wavelength, Dpj)    
                                                                # Wavelength and particle diameters: nanometers, 
                                                                # efficiencies: unitless, cross-sections: nm2, 
                                                                # coefficients: Mm-1, 
                                                                # size distribution concentration: cm-3
            Bext[i] = trapz(Q_ext*aSDn,logdp)
            Bsca[i] = trapz(Q_sca*aSDn,logdp)
            Babs[i] = Bext[i]-Bsca[i]
            Bback[i] = trapz(Q_back*aSDn,logdp)
            Bratio[i] = trapz(Q_ratio*aSDn,logdp)
            bigG[i] = trapz(g*Q_sca*aSDn,logdp)/trapz(Q_sca*aSDn,logdp)
            Bpr[i] = Bext[i] - bigG[i]*Bsca[i]
        cal_coefi[k] = pd.DataFrame([Bext,Bsca,Babs],index=['Ext', 'Sca', 'Abs']).T
    if len(cal_coefi) > 1:
        cal_coef = pd.concat([cal_coefi[0],cal_coefi[1]],axis=1)
        for k in range(2,len(mShell.columns)):
            cal_coef = pd.concat([cal_coef,cal_coefi[k]],axis=1)
    else:
        cal_coef = cal_coefi[0]
    cal_coef.index = ams.index
    return cal_coef



def Mie_aps(aps,m,wavelength,ams,fileid,fname):
    dp_aps = aps.columns.astype('float').values
    def cal_end(dp):    
        width = np.diff(dp)
        bin_end = dp[:-1]-0.5*width
        bin_end = np.insert(bin_end,len(dp)-1,[dp[-2]+0.5*width[-1],dp[-1]+0.5*width[-1]])
        return bin_end
    bin_end_aps = cal_end(dp_aps)
    R = {}
    for i in range(len(ams)):
        R[i] = [ps.Mie_SD_ZL(m.iloc[i,k],wavelength,dp_aps,aps.iloc[i,:],bin_end_aps[:], asDict=True) for k in range(len(m.columns))]
                                                                # Wavelength and particle diameters: nanometers, 
                                                                # efficiencies: unitless, cross-sections: nm2, 
                                                                # coefficients: Mm-1, 
                                                                # size distribution concentration: cm-3 per log10
    print(R)
    Ext = []
    Sca = []
    Abs = []
    bigG = []
    cal_coef = []

    for k in range(len(m.columns)):
        Ext.append(pd.DataFrame([R[i][k]['Bext'] for i in R], columns = ['Ext']))
        Sca.append(pd.DataFrame([R[i][k]['Bsca'] for i in R], columns = ['Sca']))
        Abs.append(pd.DataFrame([R[i][k]['Babs'] for i in R], columns = ['Abs']))
        bigG.append(pd.DataFrame([R[i][k]['bigG'] for i in R], columns = ['bigG']))
        cal_coef.append(pd.concat([Ext[k], Sca[k], Abs[k], bigG[k]], axis=1))
    cal_coef = pd.concat(cal_coef,axis=1)
    cal_coef.index = ams.index
    return cal_coef










