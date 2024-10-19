import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr
import matplotlib as mpl
cmap = mpl.cm.get_cmap("viridis", 10)
smap = mpl.cm.get_cmap("spring",10)
amap=mpl.cm.get_cmap("RdBu",10)
tmap=mpl.cm.get_cmap("terrain",10)
# 读取Excel文件

df1 = pd.read_excel('/Users/duan/Desktop/sep/SolarErupDB_Kazachenko2023_V1.xlsx')

# 定义速度的范围和标签
bins = [0, 359, float('inf')]
labels = [1, 2]

# 使用 pd.cut() 进行分类
width = df1['width(deg)']






flux = df1['Flare peak X-ray flux[W/m^2]']
cmespeed = df1['Vcme(km/s)']
T = df1['Peak temperature [MK]']
sep = df1['SEP']
EM = df1['Peak EM [1e48 cm-3]']
dur = df1['GOES duration [sec]']
Unsigned = df1['Unsigned AR flux [Mx]']
recflux = df1['Reconnection flux [Mx]']
recrate = df1['Reconnection flux rate [Mx/s]']
recrate = pd.to_numeric(recrate, errors='coerce')
ARarea = df1['AR area [cm^2]']
Rbarea = df1['Ribbon area [cm^2]']
Rbarea = pd.to_numeric(Rbarea,errors='coerce')
fluxratio = df1['Flux: ratio [%]']
arearatio = df1['Area: ratio [%]']
kb=1.380649e-16
thermal=3/np.power(8,0.25)*np.sqrt(EM)*np.power(Rbarea,0.75)*T*kb*10**30
thermal = pd.to_numeric(thermal,errors='coerce')
mask_cme =  ~np.isnan(cmespeed)
mask_sep = sep !=1
mask_sep1 = sep == 1


widthtype = pd.cut(width[mask_cme], bins=bins, labels=labels, right=False)
print(len(flux[mask_cme&mask_sep]))
print(len(flux[mask_sep1]))
print(len(T[mask_cme]))

'''corr_coefficient1, p_value1 = pearsonr(np.log10(flux[mask_cme&mask_sep]), np.log10(cmespeed[mask_cme&mask_sep]))
corr_coefficient2, p_value2 = pearsonr(np.log10(flux[mask_sep1]),np.log10(cmespeed[mask_sep1]))
corr_coefficient3, p_value3 = pearsonr(T[mask_cme&mask_sep],np.log10(cmespeed[mask_cme&mask_sep]))
corr_coefficient4, p_value4 = pearsonr(np.log10(EM[mask_cme&mask_sep]),np.log10(cmespeed[mask_cme&mask_sep]))
corr_coefficient5, p_value5 = pearsonr(np.log10(recflux[mask_cme&mask_sep]),np.log10(cmespeed[mask_cme&mask_sep]))
corr_coefficient6, p_value6 = pearsonr(np.log10(Unsigned[mask_cme&mask_sep]),np.log10(cmespeed[mask_cme&mask_sep]))
corr_coefficient7, p_value7 = pearsonr(np.log10(recrate[mask_cme&mask_sep]),np.log10(cmespeed[mask_cme&mask_sep]))
corr_coefficient8, p_value8 = pearsonr(np.log10(ARarea[mask_cme&mask_sep]),np.log10(cmespeed[mask_cme&mask_sep]))
corr_coefficient9, p_value9 = pearsonr(np.log10(Rbarea[mask_cme&mask_sep]),np.log10(cmespeed[mask_cme&mask_sep]))
corr_coefficient10, p_value10 = pearsonr(fluxratio[mask_cme&mask_sep],np.log10(cmespeed[mask_cme&mask_sep]))
corr_coefficient11, p_value11 = pearsonr(arearatio[mask_cme&mask_sep],np.log10(cmespeed[mask_cme&mask_sep]))
corr_coefficient12, p_value12 = pearsonr(thermal[mask_cme&mask_sep],np.log10(cmespeed[mask_cme&mask_sep]))'''

'''print(f"{corr_coefficient1} {p_value1}\n{corr_coefficient2} {p_value2}\n{corr_coefficient3} {p_value3}\n {corr_coefficient4} {p_value4}\n"
      f"{corr_coefficient5} {p_value5}\n {corr_coefficient6} {p_value6}\n {corr_coefficient7} {p_value7}\n {corr_coefficient8} {p_value8}\n"
      f"{corr_coefficient9} {p_value9}\n {corr_coefficient10} {p_value10}\n{corr_coefficient11} {p_value11} \n{corr_coefficient12} {p_value12}")'''




fig= plt.figure(figsize=(20,15))
ax1 = fig.add_subplot(3,4,1)
ax2 = fig.add_subplot(3,4,2)
ax3 = fig.add_subplot(3,4,3)
ax4 = fig.add_subplot(3,4,4)
ax5 = fig.add_subplot(3,4,5)
ax6 = fig.add_subplot(3,4,6)
ax7 = fig.add_subplot(3,4,7)
ax8 = fig.add_subplot(3,4,8)
ax9 = fig.add_subplot(3,4,9)
ax10 = fig.add_subplot(3,4,10)
ax11 = fig.add_subplot(3,4,11)
ax12 = fig.add_subplot(3,4,12)
for type in widthtype.unique():
    mask = widthtype == type
    if type ==1:
       ax1.scatter(flux[mask_cme&mask],cmespeed[mask_cme&mask],color=cmap(5),marker='o',facecolor='none',s=30,label='None Halo')
       ax1.scatter(flux[mask & mask_sep1], cmespeed[mask & mask_sep1], color=cmap(1), marker='o', s=30)
       coefficients = np.polyfit(np.log10(flux[mask_cme & mask]), np.log10(cmespeed[mask_cme & mask]), 1)
       corr_coefficient_none,p_value_none=pearsonr(np.log10(flux[mask_cme&mask]),np.log10(cmespeed[mask_cme&mask]))
       polynomial = np.poly1d(coefficients)
       x_fit = np.linspace(np.log10(flux[mask_cme].min()), np.log10(flux[mask_cme].max()), 100)
       y_fit = polynomial(x_fit)
       ax1.plot(10**x_fit, 10**y_fit, color=smap(1), linestyle='solid',alpha=1,linewidth=0.5,label='r={:.2f}-None Halo'.format(corr_coefficient_none))
    elif type ==2:
        ax1.scatter(flux[mask_cme & mask], cmespeed[mask_cme & mask], color=cmap(5), marker='^',s=30,
                    label='Halo')
        ax1.scatter(flux[mask&mask_sep1],cmespeed[mask&mask_sep1],color=cmap(1),marker='^',s=30,label='SEP')
        coefficients = np.polyfit(np.log10(flux[mask_cme & mask]), np.log10(cmespeed[mask_cme & mask]), 1)
        corr_coefficient_halo, p_value_halo = pearsonr(np.log10(flux[mask_cme & mask]), np.log10(cmespeed[mask_cme & mask]))
        polynomial = np.poly1d(coefficients)
        y_fit = polynomial(x_fit)
        ax1.plot(10**x_fit, 10 ** y_fit, color=smap(5), linestyle='solid', alpha=1, linewidth=0.5,
                 label='r={:.2f}-Halo'.format(corr_coefficient_halo))


for type in widthtype.unique():
    mask = widthtype == type
    if type ==1:
       ax2.scatter(T[mask_cme&mask],cmespeed[mask_cme&mask],color=cmap(5),marker='o',facecolor='none',s=30)
       ax2.scatter(T[mask & mask_sep1], cmespeed[mask & mask_sep1], color=cmap(1), marker='o', s=30)
       coefficients3 = np.polyfit(T[mask_cme & mask], np.log10(cmespeed[mask_cme & mask]), 1)
       corr_coefficient_none,p_value_none=pearsonr(T[mask_cme&mask],np.log10(cmespeed[mask_cme&mask]))
       polynomial3 = np.poly1d(coefficients3)
       x_fit = np.linspace(T[mask_cme].min(), T[mask_cme].max(), 100)
       y_fit = polynomial3(x_fit)
       ax2.plot(x_fit, 10**y_fit, color=smap(1), linestyle='solid',alpha=1,linewidth=0.5,label='r={:.2f}-None Halo'.format(corr_coefficient_none))
    elif type ==2:
        ax2.scatter(T[mask_cme & mask], cmespeed[mask_cme & mask], color=cmap(5), marker='^',s=30)
        ax2.scatter(T[mask&mask_sep1],cmespeed[mask&mask_sep1],color=cmap(1),marker='^',s=30)
        coefficients4 = np.polyfit(T[mask_cme & mask], np.log10(cmespeed[mask_cme & mask]), 1)
        corr_coefficient_halo, p_value_halo = pearsonr(T[mask_cme & mask], np.log10(cmespeed[mask_cme & mask]))
        polynomial4 = np.poly1d(coefficients4)
        y_fit = polynomial4(x_fit)
        ax2.plot(x_fit, 10 ** y_fit, color=smap(5), linestyle='solid', alpha=1, linewidth=0.5,
                 label='r={:.2f}-Halo'.format(corr_coefficient_halo))


for type in widthtype.unique():
    mask = widthtype == type
    if type ==1:
       ax3.scatter(EM[mask_cme&mask],cmespeed[mask_cme&mask],color=cmap(5),marker='o',facecolor='none',s=30)
       ax3.scatter(EM[mask & mask_sep1], cmespeed[mask & mask_sep1], color=cmap(1), marker='o', s=30)
       coefficients = np.polyfit(np.log10(EM[mask_cme & mask]), np.log10(cmespeed[mask_cme & mask]), 1)
       corr_coefficient_none,p_value_none=pearsonr(np.log10(EM[mask_cme&mask]),np.log10(cmespeed[mask_cme&mask]))
       polynomial = np.poly1d(coefficients)
       x_fit = np.linspace(np.log10(EM[mask_cme].min()), np.log10(EM[mask_cme].max()), 100)
       y_fit = polynomial(x_fit)
       ax3.plot(10**x_fit, 10**y_fit, color=smap(1), linestyle='solid',alpha=1,linewidth=0.5,label='r={:.2f}-None Halo'.format(corr_coefficient_none))
    elif type ==2:
        ax3.scatter(EM[mask_cme & mask], cmespeed[mask_cme & mask], color=cmap(5), marker='^',s=30)
        ax3.scatter(EM[mask&mask_sep1],cmespeed[mask&mask_sep1],color=cmap(1),marker='^',s=30)
        coefficients = np.polyfit(np.log10(EM[mask_cme & mask]), np.log10(cmespeed[mask_cme & mask]), 1)
        corr_coefficient_halo, p_value_halo = pearsonr(np.log10(EM[mask_cme & mask]), np.log10(cmespeed[mask_cme & mask]))
        polynomial = np.poly1d(coefficients)
        y_fit = polynomial(x_fit)
        ax3.plot(10**x_fit, 10 ** y_fit, color=smap(5), linestyle='solid', alpha=1, linewidth=0.5,
                 label='r={:.2f}-Halo'.format(corr_coefficient_halo))

for type in widthtype.unique():
    mask = widthtype == type
    if type ==1:
       ax4.scatter(dur[mask_cme&mask]/60,cmespeed[mask_cme&mask],color=cmap(5),marker='o',facecolor='none',s=30)
       ax4.scatter(dur[mask & mask_sep1]/60, cmespeed[mask & mask_sep1], color=cmap(1), marker='o', s=30)
       coefficients = np.polyfit(dur[mask_cme & mask]/60, np.log10(cmespeed[mask_cme & mask]), 1)
       corr_coefficient_none,p_value_none=pearsonr(dur[mask_cme&mask]/60,np.log10(cmespeed[mask_cme&mask]))
       polynomial = np.poly1d(coefficients)
       x_fit = np.linspace(dur[mask_cme].min()/60, dur[mask_cme].max()/60, 100)
       y_fit = polynomial(x_fit)
       ax4.plot(x_fit, 10**y_fit, color=smap(1), linestyle='solid',alpha=1,linewidth=0.5,label='r={:.2f}-None Halo'.format(corr_coefficient_none))
    elif type ==2:
       ax4.scatter(dur[mask_cme & mask]/60, cmespeed[mask_cme & mask], color=cmap(5), marker='^',s=30)
       ax4.scatter(dur[mask&mask_sep1]/60,cmespeed[mask&mask_sep1],color=cmap(1),marker='^',s=30)
       coefficients = np.polyfit(dur[mask_cme & mask]/60, np.log10(cmespeed[mask_cme & mask]), 1)
       corr_coefficient_halo, p_value_halo = pearsonr(dur[mask_cme & mask]/60, np.log10(cmespeed[mask_cme & mask]))
       polynomial = np.poly1d(coefficients)
       y_fit = polynomial(x_fit)
       ax4.plot(x_fit, 10 ** y_fit, color=smap(5), linestyle='solid', alpha=1, linewidth=0.5,
                 label='r={:.2f}-Halo'.format(corr_coefficient_halo))

for type in widthtype.unique():
    mask = widthtype == type
    if type ==1:
       ax5.scatter(Unsigned[mask_cme&mask],cmespeed[mask_cme&mask],color=cmap(5),marker='o',facecolor='none',s=30)
       ax5.scatter(Unsigned[mask&mask_sep1], cmespeed[mask & mask_sep1], color=cmap(1), marker='o', s=30)
       coefficients = np.polyfit(np.log10(Unsigned[mask_cme&mask]), np.log10(cmespeed[mask_cme & mask]), 1)
       corr_coefficient_none,p_value_none=pearsonr(np.log10(Unsigned[mask_cme&mask]),np.log10(cmespeed[mask_cme&mask]))
       polynomial = np.poly1d(coefficients)
       x_fit = np.linspace(np.log10(Unsigned[mask_cme].min()), np.log10(Unsigned[mask_cme].max()), 100)
       y_fit = polynomial(x_fit)
       ax5.plot(10**x_fit, 10**y_fit, color=smap(1), linestyle='solid',alpha=1,linewidth=0.5,label='r={:.2f}-None Halo'.format(corr_coefficient_none))
    elif type ==2:
       ax5.scatter(Unsigned[mask_cme&mask],cmespeed[mask_cme&mask], color=cmap(5), marker='^',s=30)
       ax5.scatter(Unsigned[mask&mask_sep1],cmespeed[mask&mask_sep1],color=cmap(1),marker='^',s=30)
       coefficients = np.polyfit(np.log10(Unsigned[mask_cme&mask]), np.log10(cmespeed[mask_cme & mask]), 1)
       corr_coefficient_halo, p_value_halo = pearsonr(np.log10(Unsigned[mask_cme&mask]), np.log10(cmespeed[mask_cme & mask]))
       polynomial = np.poly1d(coefficients)
       y_fit = polynomial(x_fit)
       ax5.plot(10**x_fit, 10 ** y_fit, color=smap(5), linestyle='solid', alpha=1, linewidth=0.5,
                 label='r={:.2f}-Halo'.format(corr_coefficient_halo))



for type in widthtype.unique():
    mask = widthtype == type
    if type ==1:
       ax6.scatter(recflux[mask_cme&mask],cmespeed[mask_cme&mask],color=cmap(5),marker='o',facecolor='none',s=30)
       ax6.scatter(recflux[mask&mask_sep1], cmespeed[mask & mask_sep1], color=cmap(1), marker='o', s=30)
       coefficients = np.polyfit(np.log10(recflux[mask_cme&mask]), np.log10(cmespeed[mask_cme & mask]), 1)
       corr_coefficient_none,p_value_none=pearsonr(np.log10(recflux[mask_cme&mask]),np.log10(cmespeed[mask_cme&mask]))
       polynomial = np.poly1d(coefficients)
       x_fit = np.linspace(np.log10(recflux[mask_cme].min()), np.log10(recflux[mask_cme].max()), 100)
       y_fit = polynomial(x_fit)
       ax6.plot(10**x_fit, 10**y_fit, color=smap(1), linestyle='solid',alpha=1,linewidth=0.5,label='r={:.2f}-None Halo'.format(corr_coefficient_none))
    elif type ==2:
       ax6.scatter(recflux[mask_cme&mask],cmespeed[mask_cme&mask], color=cmap(5), marker='^',s=30)
       ax6.scatter(recflux[mask&mask_sep1],cmespeed[mask&mask_sep1],color=cmap(1),marker='^',s=30)
       coefficients = np.polyfit(np.log10(recflux[mask_cme&mask]), np.log10(cmespeed[mask_cme & mask]), 1)
       corr_coefficient_halo, p_value_halo = pearsonr(np.log10(recflux[mask_cme&mask]), np.log10(cmespeed[mask_cme & mask]))
       polynomial = np.poly1d(coefficients)
       y_fit = polynomial(x_fit)
       ax6.plot(10**x_fit, 10 ** y_fit, color=smap(5), linestyle='solid', alpha=1, linewidth=0.5,
                 label='r={:.2f}-Halo'.format(corr_coefficient_halo))


for type in widthtype.unique():
    mask = widthtype == type
    if type ==1:
       ax7.scatter(recrate[mask_cme&mask],cmespeed[mask_cme&mask],color=cmap(5),marker='o',facecolor='none',s=30)
       ax7.scatter(recrate[mask&mask_sep1], cmespeed[mask & mask_sep1], color=cmap(1), marker='o', s=30)
       coefficients = np.polyfit(np.log10(recrate[mask_cme&mask]), np.log10(cmespeed[mask_cme & mask]), 1)
       corr_coefficient_none,p_value_none=pearsonr(np.log10(recrate[mask_cme&mask]),np.log10(cmespeed[mask_cme&mask]))
       polynomial = np.poly1d(coefficients)
       x_fit = np.linspace(np.log10(recrate[mask_cme].min()), np.log10(recrate[mask_cme].max()), 100)
       y_fit = polynomial(x_fit)
       ax7.plot(10**x_fit, 10**y_fit, color=smap(1), linestyle='solid',alpha=1,linewidth=0.5,label='r={:.2f}-None Halo'.format(corr_coefficient_none))
    elif type ==2:
       ax7.scatter(recrate[mask_cme&mask],cmespeed[mask_cme&mask], color=cmap(5), marker='^',s=30)
       ax7.scatter(recrate[mask&mask_sep1],cmespeed[mask&mask_sep1],color=cmap(1),marker='^',s=30)
       coefficients = np.polyfit(np.log10(recrate[mask_cme&mask]), np.log10(cmespeed[mask_cme & mask]), 1)
       corr_coefficient_halo, p_value_halo = pearsonr(np.log10(recrate[mask_cme&mask]), np.log10(cmespeed[mask_cme & mask]))
       polynomial = np.poly1d(coefficients)
       y_fit = polynomial(x_fit)
       ax7.plot(10**x_fit, 10 ** y_fit, color=smap(5), linestyle='solid', alpha=1, linewidth=0.5,
                 label='r={:.2f}-Halo'.format(corr_coefficient_halo))


for type in widthtype.unique():
    mask = widthtype == type
    if type ==1:
       ax8.scatter(ARarea[mask_cme&mask],cmespeed[mask_cme&mask],color=cmap(5),marker='o',facecolor='none',s=30)
       ax8.scatter(ARarea[mask&mask_sep1], cmespeed[mask & mask_sep1], color=cmap(1), marker='o', s=30)
       coefficients = np.polyfit(np.log10(ARarea[mask_cme&mask]), np.log10(cmespeed[mask_cme & mask]), 1)
       corr_coefficient_none,p_value_none=pearsonr(np.log10(ARarea[mask_cme&mask]),np.log10(cmespeed[mask_cme&mask]))
       polynomial = np.poly1d(coefficients)
       x_fit = np.linspace(np.log10(ARarea[mask_cme].min()), np.log10(ARarea[mask_cme].max()), 100)
       y_fit = polynomial(x_fit)
       ax8.plot(10**x_fit, 10**y_fit, color=smap(1), linestyle='solid',alpha=1,linewidth=0.5,label='r={:.2f}-None Halo'.format(corr_coefficient_none))
    elif type ==2:
       ax8.scatter(ARarea[mask_cme&mask],cmespeed[mask_cme&mask], color=cmap(5), marker='^',s=30)
       ax8.scatter(ARarea[mask&mask_sep1],cmespeed[mask&mask_sep1],color=cmap(1),marker='^',s=30)
       coefficients = np.polyfit(np.log10(ARarea[mask_cme&mask]), np.log10(cmespeed[mask_cme & mask]), 1)
       corr_coefficient_halo, p_value_halo = pearsonr(np.log10(ARarea[mask_cme&mask]), np.log10(cmespeed[mask_cme & mask]))
       polynomial = np.poly1d(coefficients)
       y_fit = polynomial(x_fit)
       ax8.plot(10**x_fit, 10 ** y_fit, color=smap(5), linestyle='solid', alpha=1, linewidth=0.5,
                 label='r={:.2f}-Halo'.format(corr_coefficient_halo))

for type in widthtype.unique():
    mask = widthtype == type
    if type ==1:
       ax9.scatter(Rbarea[mask_cme&mask],cmespeed[mask_cme&mask],color=cmap(5),marker='o',facecolor='none',s=30)
       ax9.scatter(Rbarea[mask&mask_sep1], cmespeed[mask & mask_sep1], color=cmap(1), marker='o', s=30)
       coefficients = np.polyfit(np.log10(Rbarea[mask_cme&mask]), np.log10(cmespeed[mask_cme & mask]), 1)
       corr_coefficient_none,p_value_none=pearsonr(np.log10(Rbarea[mask_cme&mask]),np.log10(cmespeed[mask_cme&mask]))
       polynomial = np.poly1d(coefficients)
       x_fit = np.linspace(np.log10(Rbarea[mask_cme].min()), np.log10(Rbarea[mask_cme].max()), 100)
       y_fit = polynomial(x_fit)
       ax9.plot(10**x_fit, 10**y_fit, color=smap(1), linestyle='solid',alpha=1,linewidth=0.5,label='r={:.2f}-None Halo'.format(corr_coefficient_none))
    elif type ==2:
       ax9.scatter(Rbarea[mask_cme&mask],cmespeed[mask_cme&mask], color=cmap(5), marker='^',s=30)
       ax9.scatter(Rbarea[mask&mask_sep1],cmespeed[mask&mask_sep1],color=cmap(1),marker='^',s=30)
       coefficients = np.polyfit(np.log10(Rbarea[mask_cme&mask]), np.log10(cmespeed[mask_cme & mask]), 1)
       corr_coefficient_halo, p_value_halo = pearsonr(np.log10(Rbarea[mask_cme&mask]), np.log10(cmespeed[mask_cme & mask]))
       polynomial = np.poly1d(coefficients)
       y_fit = polynomial(x_fit)
       ax9.plot(10**x_fit, 10 ** y_fit, color=smap(5), linestyle='solid', alpha=1, linewidth=0.5,
                 label='r={:.2f}-Halo'.format(corr_coefficient_halo))


for type in widthtype.unique():
    mask = widthtype == type
    if type ==1:
       ax10.scatter(fluxratio[mask_cme&mask],cmespeed[mask_cme&mask],color=cmap(5),marker='o',facecolor='none',s=30)
       ax10.scatter(fluxratio[mask&mask_sep1], cmespeed[mask & mask_sep1], color=cmap(1), marker='o', s=30)
       coefficients = np.polyfit(fluxratio[mask_cme&mask], np.log10(cmespeed[mask_cme & mask]), 1)
       corr_coefficient_none,p_value_none=pearsonr(fluxratio[mask_cme&mask],np.log10(cmespeed[mask_cme&mask]))
       polynomial = np.poly1d(coefficients)
       x_fit = np.linspace(fluxratio[mask_cme].min(), fluxratio[mask_cme].max(), 100)
       y_fit = polynomial(x_fit)
       ax10.plot(x_fit, 10**y_fit, color=smap(1), linestyle='solid',alpha=1,linewidth=0.5,label='r={:.2f}-None Halo'.format(corr_coefficient_none))
    elif type ==2:
       ax10.scatter(fluxratio[mask_cme&mask],cmespeed[mask_cme&mask], color=cmap(5), marker='^',s=30)
       ax10.scatter(fluxratio[mask&mask_sep1],cmespeed[mask&mask_sep1],color=cmap(1),marker='^',s=30)
       coefficients = np.polyfit(fluxratio[mask_cme&mask], np.log10(cmespeed[mask_cme & mask]), 1)
       corr_coefficient_halo, p_value_halo = pearsonr(fluxratio[mask_cme&mask], np.log10(cmespeed[mask_cme & mask]))
       polynomial = np.poly1d(coefficients)
       y_fit = polynomial(x_fit)
       ax10.plot(x_fit, 10 ** y_fit, color=smap(5), linestyle='solid', alpha=1, linewidth=0.5,
                 label='r={:.2f}-Halo'.format(corr_coefficient_halo))

for type in widthtype.unique():
    mask = widthtype == type
    if type ==1:
       ax11.scatter(arearatio[mask_cme&mask],cmespeed[mask_cme&mask],color=cmap(5),marker='o',facecolor='none',s=30)
       ax11.scatter(arearatio[mask&mask_sep1], cmespeed[mask & mask_sep1], color=cmap(1), marker='o', s=30)
       coefficients = np.polyfit(arearatio[mask_cme&mask], np.log10(cmespeed[mask_cme & mask]), 1)
       corr_coefficient_none,p_value_none=pearsonr(arearatio[mask_cme&mask],np.log10(cmespeed[mask_cme&mask]))
       polynomial = np.poly1d(coefficients)
       x_fit = np.linspace(arearatio[mask_cme].min(), arearatio[mask_cme].max(), 100)
       y_fit = polynomial(x_fit)
       ax11.plot(x_fit, 10**y_fit, color=smap(1), linestyle='solid',alpha=1,linewidth=0.5,label='r={:.2f}-None Halo'.format(corr_coefficient_none))
    elif type ==2:
       ax11.scatter(arearatio[mask_cme&mask],cmespeed[mask_cme&mask], color=cmap(5), marker='^',s=30)
       ax11.scatter(arearatio[mask&mask_sep1],cmespeed[mask&mask_sep1],color=cmap(1),marker='^',s=30)
       coefficients = np.polyfit(arearatio[mask_cme&mask], np.log10(cmespeed[mask_cme & mask]), 1)
       corr_coefficient_halo, p_value_halo = pearsonr(arearatio[mask_cme&mask], np.log10(cmespeed[mask_cme & mask]))
       polynomial = np.poly1d(coefficients)
       y_fit = polynomial(x_fit)
       ax11.plot(x_fit, 10 ** y_fit, color=smap(5), linestyle='solid', alpha=1, linewidth=0.5,
                 label='r={:.2f}-Halo'.format(corr_coefficient_halo))


for type in widthtype.unique():
    mask = widthtype == type
    if type ==1:
       ax12.scatter(thermal[mask_cme&mask],cmespeed[mask_cme&mask],color=cmap(5),marker='o',facecolor='none',s=30)
       ax12.scatter(thermal[mask&mask_sep1], cmespeed[mask & mask_sep1], color=cmap(1), marker='o', s=30)
       coefficients = np.polyfit(np.log10(thermal[mask_cme&mask]), np.log10(cmespeed[mask_cme & mask]), 1)
       corr_coefficient_none,p_value_none=pearsonr(np.log10(thermal[mask_cme&mask]),np.log10(cmespeed[mask_cme&mask]))
       polynomial = np.poly1d(coefficients)
       x_fit = np.linspace(np.log10(thermal[mask_cme&mask].min()), np.log10(thermal[mask_cme&mask].max()), 100)
       y_fit = polynomial(x_fit)
       ax12.plot(10**x_fit, 10**y_fit, color=smap(1), linestyle='solid',alpha=1,linewidth=0.5,label='r={:.2f}-None Halo'.format(corr_coefficient_none))
    elif type ==2:
       ax12.scatter(thermal[mask_cme&mask],cmespeed[mask_cme&mask], color=cmap(5), marker='^',s=30)
       ax12.scatter(thermal[mask&mask_sep1],cmespeed[mask&mask_sep1],color=cmap(1),marker='^',s=30)
       coefficients = np.polyfit(np.log10(thermal[mask_cme&mask]), np.log10(cmespeed[mask_cme & mask]), 1)
       corr_coefficient_halo, p_value_halo = pearsonr(np.log10(thermal[mask_cme&mask]), np.log10(cmespeed[mask_cme & mask]))
       polynomial = np.poly1d(coefficients)
       y_fit = polynomial(x_fit)
       ax12.plot(10**x_fit, 10 ** y_fit, color=smap(5), linestyle='solid', alpha=1, linewidth=0.5,
                 label='r={:.2f}-Halo'.format(corr_coefficient_halo))
ax1.plot(np.linspace(0,1e-2,100),np.linspace(0,1e-2,100)*0+1000,linestyle='--',alpha=1,linewidth=0.8,color=cmap(1),label='V=1000km/s')
ax1.plot(np.linspace(0,1e-2,100),np.linspace(0,1e-2,100)*0+500,linestyle='-',alpha=1,linewidth=0.8,color=cmap(1),label='V=500km/s')
ax1.set_xscale('log')
ax1.set_yscale('log')
ax1.set_xlim(1e-6,1e-3)
ax1.set_xticks([10**(-6), 10**(-5), 10**(-4), 10**(-3)])
ax1.set_yticks([10**2,10**3,10**4])
ax1.set_xticklabels(['-6', '-5', '-4', '-3'])
ax1.set_yticklabels(['2','3','4'])
ax1.set_xlabel('Peak SXR-Flux(log)')
ax1.set_ylabel('Vcme(km/s)')
ax1.legend(fontsize=8)


ax2.plot(np.linspace(0,30),np.linspace(0,30)*0+1000,linestyle='--',alpha=1,linewidth=0.8,color=cmap(1))
ax2.plot(np.linspace(0,30),np.linspace(0,30)*0+500,linestyle='-',alpha=1,linewidth=0.8,color=cmap(1))
#ax2.set_xscale('log')
ax2.set_xlim(7,25)
ax2.set_yscale('log')
ax2.set_xticks([10,20])
ax2.set_yticks([10**2,10**3,10**4])
ax2.set_xticklabels(['10', '20'])
ax2.set_yticklabels(['2','3','4'])
ax2.set_xlabel('Peak Temperature[MK]')
ax2.legend(fontsize=8)


ax3.plot(np.linspace(0,200),np.linspace(0,30)*0+1000,linestyle='--',alpha=1,linewidth=0.8,color=cmap(1))
ax3.plot(np.linspace(0,200),np.linspace(0,30)*0+500,linestyle='-',alpha=1,linewidth=0.8,color=cmap(1))
ax3.set_xscale('log')
ax3.set_yscale('log')
ax3.set_xlim(1e-3,100)
ax3.set_xticks([10**(0), 10**(1)])
ax3.set_yticks([10**2,10**3,10**4])
ax3.set_xticklabels(['0', '1'])
ax3.set_yticklabels(['2','3','4'])
ax3.set_xlabel('Peak EM [1e48 cm-3]')
ax3.legend(fontsize=8)

ax4.plot(np.linspace(0,600),np.linspace(0,500)*0+1000,linestyle='--',alpha=1,linewidth=0.8,color=cmap(1))
ax4.plot(np.linspace(0,600),np.linspace(0,500)*0+500,linestyle='-',alpha=1,linewidth=0.8,color=cmap(1))
ax4.set_xlim(0,450)
ax4.set_yscale('log')
ax4.set_xticks([0,200,400])
ax4.set_yticks([10**2,10**3,10**4])
ax4.set_xticklabels(['0','200','400'])
ax4.set_yticklabels(['3','3','4'])
ax4.set_xlabel('GOES duration [min]')
ax4.legend(fontsize=8)


ax5.plot(np.linspace(10e21,10e24),np.linspace(10e21,10e24)*0+1000,linestyle='--',alpha=1,linewidth=0.8,color=cmap(1))
ax5.plot(np.linspace(10e21,10e24),np.linspace(10e21,10e24)*0+500,linestyle='-',alpha=1,linewidth=0.8,color=cmap(1))
ax5.set_xlim(1e23,10**23.2)
ax5.set_xscale('log')
ax5.set_yscale('log')
ax5.set_xticks([10**(22.0),10**(23.0)])
ax5.set_yticks([10**2,10**3,10**4])
ax5.set_xticklabels(['22','23'])
ax5.set_yticklabels(['2','3','4'])
ax5.set_xlabel('Unsigned AR flux [Mx]')
ax5.legend(fontsize=8)


ax6.plot(np.linspace(10e19,10e23),np.linspace(10e19,10e24)*0+1000,linestyle='--',alpha=1,linewidth=0.8,color=cmap(1))
ax6.plot(np.linspace(10e19,10e23),np.linspace(10e19,10e24)*0+500,linestyle='-',alpha=1,linewidth=0.8,color=cmap(1))
ax6.set_xscale('log')
ax6.set_yscale('log')
ax6.set_xlim(10**20.5,10**22.8)
ax6.set_xticks([10**(21.0),10**(22.0)])
ax6.set_yticks([10**2,10**3,10**4])
ax6.set_xticklabels(['21','22'])
ax6.set_yticklabels(['2','3','4'])
ax6.set_xlabel('Reconnection flux [Mx]')
ax6.legend(fontsize=8)


ax7.plot(np.linspace(10e16,10e23),np.linspace(10e17,10e24)*0+1000,linestyle='--',alpha=1,linewidth=0.8,color=cmap(1))
ax7.plot(np.linspace(10e16,10e23),np.linspace(10e17,10e24)*0+500,linestyle='-',alpha=1,linewidth=0.8,color=cmap(1))
ax7.set_xscale('log')
ax7.set_yscale('log')
ax7.set_xlim(10**17.8,10**20.4)
ax7.set_xticks([10**(18.0),10**(19.0),10**(20.0)])
ax7.set_yticks([10**2,10**3,10**4])
ax7.set_xticklabels(['18','19','20'])
ax7.set_yticklabels(['2','3','4'])
ax7.set_xlabel('Reconnection flux rate [Mx/s]')
ax7.legend(fontsize=8)

ax8.plot(np.linspace(10e16,10e23),np.linspace(10e17,10e24)*0+1000,linestyle='--',alpha=1,linewidth=0.8,color=cmap(1))
ax8.plot(np.linspace(10e16,10e23),np.linspace(10e17,10e24)*0+500,linestyle='-',alpha=1,linewidth=0.8,color=cmap(1))
ax8.set_xlim(10**19.4,10**20.3)
ax8.set_xscale('log')
ax8.set_yscale('log')
ax8.set_xticks([10**(19.5),10**(20.0),10**(20.5)])
ax8.set_yticks([10**2,10**3,10**4])
ax8.set_xticklabels(['19.5','20.0','20.5'])
ax8.set_yticklabels(['2','3','4'])
ax8.set_xlabel('AR area [cm^2]')
ax8.legend(fontsize=8)

ax9.plot(np.linspace(10e16,10e23),np.linspace(10e17,10e24)*0+1000,linestyle='--',alpha=1,linewidth=0.8,color=cmap(1))
ax9.plot(np.linspace(10e16,10e23),np.linspace(10e17,10e24)*0+500,linestyle='-',alpha=1,linewidth=0.8,color=cmap(1))
ax9.set_xscale('log')
ax9.set_yscale('log')
ax9.set_xlim(10**18.1,10**19.6)
ax9.set_xticks([10**(18.5),10**(19.0),10**(19.5)])
ax9.set_yticks([10**2,10**3,10**4])
ax9.set_xticklabels(['18.5','19.0','19.5'])
ax9.set_yticklabels(['2','3','4'])
ax9.set_xlabel('Ribbon area [cm^2]')
ax9.legend(fontsize=8)

ax10.plot(np.linspace(0,80),np.linspace(0,80)*0+1000,linestyle='--',alpha=1,linewidth=0.8,color=cmap(1))
ax10.plot(np.linspace(0,80),np.linspace(0,80)*0+500,linestyle='-',alpha=1,linewidth=0.8,color=cmap(1))
ax10.set_yscale('log')
ax10.set_xlim(0,50)
ax10.set_xticks([10,20,30,40,50])
ax10.set_yticks([10**2,10**4,10**4])
ax10.set_xticklabels(['10','20','30','40','50'])
ax10.set_yticklabels(['2','3','4'])
ax10.set_xlabel('Flux: ratio [%]')
ax10.legend(fontsize=8)


#ax9.set_xscale('log')
ax11.plot(np.linspace(0,80),np.linspace(0,80)*0+1000,linestyle='--',alpha=1,linewidth=0.8,color=cmap(1))
ax11.plot(np.linspace(0,80),np.linspace(0,80)*0+500,linestyle='-',alpha=1,linewidth=0.8,color=cmap(1))
ax11.set_yscale('log')
ax11.set_xlim(0,40)
ax11.set_xticks([10,20,30])
ax11.set_yticks([10**2,10**3,10**4])
ax11.set_xticklabels(['10','20','30'])
ax11.set_yticklabels(['2','3','4'])
ax11.set_xlabel('Area: ratio [%]')
ax11.legend(fontsize=8)


ax12.plot(np.linspace(1e24,1e36),np.linspace(1e24,1e36)*0+1000,linestyle='--',alpha=1,linewidth=0.8,color=cmap(1))
ax12.plot(np.linspace(1e24,1e36),np.linspace(1e24,1e36)*0+500,linestyle='-',alpha=1,linewidth=0.8,color=cmap(1))
ax12.set_xlim(10**28.5,10**31.5)
ax12.set_xscale('log')
ax12.set_yscale('log')
ax12.set_xticks([10**29.0,10**30.0,10**31.0])
ax12.set_yticks([10**2,10**3,10**4])
ax12.set_xticklabels(['29','30','31'])
ax12.set_yticklabels(['2','3','4'])
ax12.set_xlabel('Thermal energe [ergs]')
ax12.legend(fontsize=8)
plt.show()
