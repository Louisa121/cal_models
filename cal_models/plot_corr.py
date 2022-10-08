#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 8 10:10:41 2021

@author: Luzhang

import modul for plot settings

"""
import os
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pylab as pltl
import seaborn as sns
import matplotlib.dates as mdate
from scipy import stats
import math
import PyMieScatt as ps
from matplotlib import ticker, cm
from scipy.optimize import curve_fit
from scipy.integrate import trapz
from sympy.solvers import solve
from sympy import Symbol
from matplotlib.ticker import LogFormatter 
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,NullFormatter,ScalarFormatter,
                               AutoMinorLocator)

def plot_coef(cal_c, y):    
    eq = []
    slopee = []
    interceptt = []
    r2 = []
    std_errr = []

    h = plt.scatter(y,cal_c)
    # linear regression
    yy = cal_c
    xx = y
    mask = ~np.isnan(xx) & ~np.isnan(yy)
    slope, intercept, r, p, std_err = stats.linregress(xx[mask], yy[mask]) # remove null
    def linearr(x):
        return slope * x + intercept
    mymodel = list(map(linearr, xx))        
    plt.plot(xx, mymodel)
    eq.append('y={:.2f}*x+{:.2f}  $R^2$={:.2f}'.format(slope, intercept,r**2))
    slopee.append(slope)
    interceptt.append(intercept)
    r2.append(r**2)
    std_errr.append(std_err)
#     plt.legend(eq+['BC + non-abs OC', 'BC + moderately abs BrC', 'BC + strongly abs BrC'])
    # add a y=x line
    ax1 = plt.gca()
#     lims = [np.min([ax1.get_xlim(), ax1.get_ylim()]),  # min of both axes
#             np.max([ax1.get_xlim(), ax1.get_ylim()]),  # max of both axes
#            ]
    lims = [np.min([0, 0]),  # min of both axes
            np.max([ax1.get_xlim(), ax1.get_ylim()]),  # max of both axes
           ]
    # now plot both limits against each other
    l = ax1.plot(lims, lims, linestyle='dashed', color = '#c63a33', linewidth=2, alpha=0.75, zorder=0)
    ax1.set_aspect('equal')
    ax1.set_xlim(lims)
    ax1.set_ylim(lims)
    plt.legend(eq)
    plt.ioff()
    sq = np.nansum(((y-yy)/y)**2)
    return(slope,intercept,r**2,std_err,sq,h)



# draw scattering plot of predicted and measured optical properties
def linear_r(cal_coef, mea_coef, ams,fname,i,filter2018):
    print('Creating figure ...')
    # create a figure and axis
    fig = plt.figure(figsize=(12,4))
#     font = {'family': 'san-serif',
#             'color':  'k',
#             'weight': 'normal',
#             'size': 13,
#             }
    font = {'color':  'k',
            'weight': 'normal',
            'size': 13,
            }
    

    # extinction coefficient
    ax1 = plt.subplot(131)
    plot_coef(cal_coef.iloc[:,0],mea_coef.Ext)

#     ax1.set_title('Extinction coefficient RF06', fontdict=font)
    ax1.set_title('Extinction coefficient {}'.format(filter2018.Filter[i]), fontdict=font) # set a title and labels
    ax1.set_ylabel('Calc $\sigma_{ext}\,[Mm^{-1}]$', fontdict=font)
    ax1.set_xlabel('Mea $\sigma_{ext}\,[Mm^{-1}]$', fontdict=font)  

    # scattering coefficient
    ax2 = plt.subplot(132)
    plot_coef(cal_coef.iloc[:,1],mea_coef.Sca)
#     ax2.set_title('Scattering coefficient {}'.format(filter2018.Filter[i]), fontdict=font) # set a title and labels
    ax2.set_ylabel('Calc $\sigma_{sca}\,[Mm^{-1}]$', fontdict=font)
    ax2.set_xlabel('Mea $\sigma_{sca}\,[Mm^{-1}]$', fontdict=font)

    # absorption coefficient
    ax3 = plt.subplot(133)
    plot_coef(cal_coef.iloc[:,2],mea_coef.Abs)
#     [slope3,intercept3,r23] = plot_coef([2, 5, 8],mea_coef.Abs)
#     ax3.set_title('Absorption coefficient {}'.format(filter2018.Filter[i]), fontdict=font) # set a title and labels
    ax3.set_ylabel('Calc $\sigma_{abs}\,[Mm^{-1}]$', fontdict=font)
    ax3.set_xlabel('Mea $\sigma_{abs}\,[Mm^{-1}]$', fontdict=font)
    plt.tight_layout()
#     return(slope1,intercept1,r21,slope2,intercept2,r22,slope3,intercept3,r23)
#     plt.savefig('{}/{}/Closure_{}.pdf'.format(out_dir,filter2018.Filter[i],fname))
#     cal_coef.to_csv('{}/{}/Closure_{}.csv'.format(dataout_dir,filter2018.Filter[fileid],fname))

    
def plt_mea_cal(cal_coeff,mea_coeff,model,filter2018,out_dir,fname,fileid):
    fig = plt.figure(figsize=(14,5))
    #     font = {'family': 'san-serif',
    #             'color':  'k',
    #             'weight': 'normal',
    #             'size': 13,
    #             }
    font = {'color':  'k',
            'weight': 'normal',
            'size': 13,
            }

    # extinction coefficient
    ax1 = plt.subplot(131)
    [k1,b1,r21,std1,e1,h1] = plot_coef(cal_coeff.iloc[:,0],mea_coeff.Ext_450)
    [k2,b2,r22,std2,e2,h2] = plot_coef(cal_coeff.iloc[:,0+3*1],mea_coeff.Ext_470)
    [k3,b3,r23,std3,e3,h3] = plot_coef(cal_coeff.iloc[:,0+3*2],mea_coeff.Ext_530)
    [k4,b4,r24,std4,e4,h4] = plot_coef(cal_coeff.iloc[:,0+3*3],mea_coeff.Ext_550)
    [k5,b5,r25,std5,e5,h5] = plot_coef(cal_coeff.iloc[:,0+3*4],mea_coeff.Ext_660)
    [k6,b6,r26,std6,e6,h6] = plot_coef(cal_coeff.iloc[:,0+3*5],mea_coeff.Ext_700)
    eq1 = '450 nm  y={:.2f}*x{:+.2f}  $R^2$={:.2f}'.format(k1,b1,r21)
    eq2 = '470 nm  y={:.2f}*x{:+.2f}  $R^2$={:.2f}'.format(k2,b2,r22)
    eq3 = '530 nm  y={:.2f}*x{:+.2f}  $R^2$={:.2f}'.format(k3,b3,r23)
    eq4 = '550 nm  y={:.2f}*x{:+.2f}  $R^2$={:.2f}'.format(k4,b4,r24)
    eq5 = '660 nm  y={:.2f}*x{:+.2f}  $R^2$={:.2f}'.format(k5,b5,r25)
    eq6 = '700 nm  y={:.2f}*x{:+.2f}  $R^2$={:.2f}'.format(k6,b6,r26)
    plt.legend([h1,h2,h3,h4,h5,h6],(eq1,eq2,eq3,eq4,eq5,eq6))
    ax1.set_xlim(np.min(cal_coeff.iloc[:,0+3*4])/2)
    ax1.set_ylim(np.min(cal_coeff.iloc[:,0+3*4])/2)
    ax1.set_ylabel('Calc $\sigma_{ext}\,(Mm^{-1})$', fontdict=font)
    ax1.set_xlabel('Mea $\sigma_{ext}\,(Mm^{-1})$', fontdict=font) 

    ax2 = plt.subplot(132)
    [k1,b1,r21,std1,e1,h1] = plot_coef(cal_coeff.iloc[:,1],mea_coeff.Sca_450)
    [k2,b2,r22,std2,e2,h2] = plot_coef(cal_coeff.iloc[:,1+3*1],mea_coeff.Sca_470)
    [k3,b3,r23,std3,e3,h3] = plot_coef(cal_coeff.iloc[:,1+3*2],mea_coeff.Sca_530)
    [k4,b4,r24,std4,e4,h4] = plot_coef(cal_coeff.iloc[:,1+3*3],mea_coeff.Sca_550)
    [k5,b5,r25,std5,e5,h5] = plot_coef(cal_coeff.iloc[:,1+3*4],mea_coeff.Sca_660)
    [k6,b6,r26,std6,e6,h6] = plot_coef(cal_coeff.iloc[:,1+3*5],mea_coeff.Sca_700)
    eq1 = '450 nm  y={:.2f}*x{:+.2f}  $R^2$={:.2f}'.format(k1,b1,r21)
    eq2 = '470 nm  y={:.2f}*x{:+.2f}  $R^2$={:.2f}'.format(k2,b2,r22)
    eq3 = '530 nm  y={:.2f}*x{:+.2f}  $R^2$={:.2f}'.format(k3,b3,r23)
    eq4 = '550 nm  y={:.2f}*x{:+.2f}  $R^2$={:.2f}'.format(k4,b4,r24)
    eq5 = '660 nm  y={:.2f}*x{:+.2f}  $R^2$={:.2f}'.format(k5,b5,r25)
    eq6 = '700 nm  y={:.2f}*x{:+.2f}  $R^2$={:.2f}'.format(k6,b6,r26)
    plt.legend([h1,h2,h3,h4,h5,h6],(eq1,eq2,eq3,eq4,eq5,eq6))
    ax2.set_xlim(np.min(cal_coeff.iloc[:,1+3*4])/2)
    ax2.set_ylim(np.min(cal_coeff.iloc[:,1+3*4])/2)
    ax2.set_ylabel('Calc $\sigma_{sca}\,(Mm^{-1})$', fontdict=font)
    ax2.set_xlabel('Mea $\sigma_{sca}\,(Mm^{-1})$', fontdict=font) 

    ax3 = plt.subplot(133)
    [k1,b1,r21,std1,e1,h1] = plot_coef(cal_coeff.iloc[:,2],mea_coeff.Abs_450)
    [k2,b2,r22,std2,e2,h2] = plot_coef(cal_coeff.iloc[:,2+3*1],mea_coeff.Abs_470)
    [k3,b3,r23,std3,e3,h3] = plot_coef(cal_coeff.iloc[:,2+3*2],mea_coeff.Abs_530)
    [k4,b4,r24,std4,e4,h4] = plot_coef(cal_coeff.iloc[:,2+3*3],mea_coeff.Abs_550)
    [k5,b5,r25,std5,e5,h5] = plot_coef(cal_coeff.iloc[:,2+3*4],mea_coeff.Abs_660)
    [k6,b6,r26,std6,e6,h6] = plot_coef(cal_coeff.iloc[:,2+3*5],mea_coeff.Abs_700)
    eq1 = '450 nm  y={:.2f}*x{:+.2f}  $R^2$={:.2f}'.format(k1,b1,r21)
    eq2 = '470 nm  y={:.2f}*x{:+.2f}  $R^2$={:.2f}'.format(k2,b2,r22)
    eq3 = '530 nm  y={:.2f}*x{:+.2f}  $R^2$={:.2f}'.format(k3,b3,r23)
    eq4 = '550 nm  y={:.2f}*x{:+.2f}  $R^2$={:.2f}'.format(k4,b4,r24)
    eq5 = '660 nm  y={:.2f}*x{:+.2f}  $R^2$={:.2f}'.format(k5,b5,r25)
    eq6 = '700 nm  y={:.2f}*x{:+.2f}  $R^2$={:.2f}'.format(k6,b6,r26)
    plt.legend([h1,h2,h3,h4,h5,h6],(eq1,eq2,eq3,eq4,eq5,eq6))
    ax3.set_xlim(np.min(cal_coeff.iloc[:,2+3*3])/2)
    ax3.set_ylim(np.min(cal_coeff.iloc[:,2+3*4])/2)
    ax3.set_ylabel('Calc $\sigma_{abs}\,(Mm^{-1})$', fontdict=font)
    ax3.set_xlabel('Mea $\sigma_{abs}\,(Mm^{-1})$', fontdict=font) 

    plt.tight_layout()
    plt.savefig('{}/{}/{}/Closure_{}.pdf'.format(out_dir,filter2018.Filter[fileid],model,fname))
    # plt.close()
    # plt.clf()





