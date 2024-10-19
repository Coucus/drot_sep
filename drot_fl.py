import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr
import matplotlib as mpl
cmap = mpl.cm.get_cmap("viridis", 10)
smap = mpl.cm.get_cmap("spring",10)
amap=mpl.cm.get_cmap("plasma",15)
tmap=mpl.cm.get_cmap("terrain",10)
# 读取Excel文件

df1 = pd.read_excel('/Users/duan/Desktop/sep/SolarErupDB_Kazachenko2023_V1.xlsx')

# 定义速度的范围和标签
bins = [0, 359, float('inf')]
labels = [1, 2]
binsfl =['E','C']


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
eruptivity=df1['Eruptivity']
kb=1.380649e-16
thermal=3/np.power(8,0.25)*np.sqrt(EM)*np.power(Rbarea,0.75)*T*kb*10**30
thermal = pd.to_numeric(thermal,errors='coerce')
mask_cme =  ~np.isnan(cmespeed)
mask_sep = sep !=1
mask_sep1 = sep == 1
mask_fl= eruptivity =='E'
print(eruptivity.unique())
widthtype = pd.cut(width[mask_cme], bins=bins, labels=labels, right=False)
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

for type in eruptivity.unique():
    mask = eruptivity == type
    if type=='E':
      ax1.scatter(T[mask],flux[mask],color=amap(7),marker='o',facecolor='none',s=30,label='{}'.format(type))
      ax1.scatter(T[mask&mask_sep1],flux[mask_sep1],color=amap(1),marker='o',s=30,label='SEP')
      coefficients = np.polyfit(T[mask],np.log10(flux[mask]),  1)
      corr_coefficient,p_value=pearsonr(np.log10(flux[mask]),T[mask])
      polynomial = np.poly1d(coefficients)
      coefficients_sep = np.polyfit(T[mask&mask_sep1], np.log10(flux[mask&mask_sep1]), 1)
      corr_coefficient_sep, p_value_sep = pearsonr(T[mask&mask_sep1], np.log10(flux[mask&mask_sep1]))
      polynomial_sep = np.poly1d(coefficients_sep)
      x_fit = np.linspace(T[mask].min(), T[mask].max(), 100)
      y_fit = polynomial(x_fit)
      y_fit_sep= polynomial_sep(x_fit)
      ax1.plot(x_fit, 10 ** y_fit_sep, color=smap(1), linestyle='dashed', alpha=1, linewidth=1,
               label='r_sep={:.2f}'.format(corr_coefficient_sep))
      ax1.plot(x_fit, 10**y_fit, color=amap(1), linestyle='solid',alpha=0.6,linewidth=0.5,label='r_eur={:.2f}'.format(corr_coefficient))


for type in eruptivity.unique():
    mask = eruptivity == type
    if type=='E':
      ax2.scatter(EM[mask],flux[mask],color=amap(1),marker='o',facecolor='none',s=30)
      ax2.scatter(EM[mask&mask_sep1],flux[mask_sep1],color=amap(1),marker='o',s=30)
      coefficients = np.polyfit(np.log10(EM[mask]),np.log10(flux[mask]),  1)
      corr_coefficient,p_value=pearsonr(np.log10(flux[mask]),np.log10(EM[mask]))
      polynomial = np.poly1d(coefficients)
      coefficients_sep = np.polyfit(np.log10(EM[mask&mask_sep1]), np.log10(flux[mask&mask_sep1]), 1)
      corr_coefficient_sep, p_value_sep = pearsonr(np.log10(EM[mask&mask_sep1]), np.log10(flux[mask&mask_sep1]))
      polynomial_sep = np.poly1d(coefficients_sep)
      x_fit = np.linspace(np.log10(1), np.log10(EM[mask].max()), 100)
      y_fit = polynomial(x_fit)
      y_fit_sep= polynomial_sep(x_fit)
      ax2.plot(10**x_fit, 10 ** y_fit_sep, color=smap(1), linestyle='dashed', alpha=1, linewidth=1,
               label='r_sep={:.2f}'.format(corr_coefficient_sep))
      ax2.plot(10**x_fit, 10**y_fit, color=amap(1), linestyle='solid',alpha=0.6,linewidth=0.5,label='r_eur={:.2f}'.format(corr_coefficient))



for type in eruptivity.unique():
    mask = eruptivity == type
    if type=='E':
      ax3.scatter(dur[mask]/60,flux[mask],color=amap(1),marker='o',facecolor='none',s=30)
      ax3.scatter(dur[mask&mask_sep1]/60,flux[mask_sep1],color=amap(1),marker='o',s=30)
      coefficients = np.polyfit(dur[mask]/60,np.log10(flux[mask]),  1)
      corr_coefficient,p_value=pearsonr(np.log10(flux[mask]),dur[mask]/60)
      polynomial = np.poly1d(coefficients)
      coefficients_sep = np.polyfit(dur[mask&mask_sep1]/60, np.log10(flux[mask&mask_sep1]), 1)
      corr_coefficient_sep, p_value_sep = pearsonr(dur[mask&mask_sep1]/60, np.log10(flux[mask&mask_sep1]))
      polynomial_sep = np.poly1d(coefficients_sep)
      x_fit = np.linspace(dur[mask].min()/60, dur[mask].max()/60, 100)
      y_fit = polynomial(x_fit)
      y_fit_sep= polynomial_sep(x_fit)
      ax3.plot(x_fit, 10 ** y_fit_sep, color=smap(1), linestyle='dashed', alpha=1, linewidth=1,
               label='r_sep={:.2f}'.format(corr_coefficient_sep))
      ax3.plot(x_fit, 10**y_fit, color=amap(1), linestyle='solid',alpha=0.6,linewidth=0.5,label='r_eur={:.2f}'.format(corr_coefficient))




for type in eruptivity.unique():
    mask = eruptivity == type
    if type=='E':
      ax4.scatter(Unsigned[mask],flux[mask],color=amap(1),marker='o',facecolor='none',s=30)
      ax4.scatter(Unsigned[mask&mask_sep1],flux[mask_sep1],color=amap(1),marker='o',s=30)
      coefficients = np.polyfit(np.log10(Unsigned[mask]),np.log10(flux[mask]),  1)
      corr_coefficient,p_value=pearsonr(np.log10(flux[mask]),np.log10(Unsigned[mask]))
      polynomial = np.poly1d(coefficients)
      coefficients_sep = np.polyfit(np.log10(Unsigned[mask&mask_sep1]), np.log10(flux[mask&mask_sep1]), 1)
      corr_coefficient_sep, p_value_sep = pearsonr(np.log10(Unsigned[mask&mask_sep1]), np.log10(flux[mask&mask_sep1]))
      polynomial_sep = np.poly1d(coefficients_sep)
      x_fit = np.linspace(np.log10(Unsigned.min()), np.log10(Unsigned.max()), 100)
      y_fit = polynomial(x_fit)
      y_fit_sep= polynomial_sep(x_fit)
      ax4.plot(10**x_fit, 10 ** y_fit_sep, color=smap(1), linestyle='dashed', alpha=1, linewidth=1,
               label='r_sep={:.2f}'.format(corr_coefficient_sep))
      ax4.plot(10**x_fit, 10**y_fit, color=amap(1), linestyle='solid',alpha=0.6,linewidth=0.5,label='r_eur={:.2f}'.format(corr_coefficient))




for type in eruptivity.unique():
    mask = eruptivity == type
    if type=='E':
      ax5.scatter(recflux[mask],flux[mask],color=amap(1),marker='o',facecolor='none',s=30)
      ax5.scatter(recflux[mask&mask_sep1],flux[mask_sep1],color=amap(1),marker='o',s=30)
      coefficients = np.polyfit(np.log10(recflux[mask]),np.log10(flux[mask]),  1)
      corr_coefficient,p_value=pearsonr(np.log10(flux[mask]),np.log10(recflux[mask]))
      polynomial = np.poly1d(coefficients)
      coefficients_sep = np.polyfit(np.log10(recflux[mask&mask_sep1]), np.log10(flux[mask&mask_sep1]), 1)
      corr_coefficient_sep, p_value_sep = pearsonr(np.log10(recflux[mask&mask_sep1]), np.log10(flux[mask&mask_sep1]))
      polynomial_sep = np.poly1d(coefficients_sep)
      x_fit = np.linspace(np.log10(recflux.min()), np.log10(recflux.max()), 100)
      y_fit = polynomial(x_fit)
      y_fit_sep= polynomial_sep(x_fit)
      ax5.plot(10**x_fit, 10 ** y_fit_sep, color=smap(1), linestyle='dashed', alpha=1, linewidth=1,
               label='r_sep={:.2f}'.format(corr_coefficient_sep))
      ax5.plot(10**x_fit, 10**y_fit, color=amap(1), linestyle='solid',alpha=0.6,linewidth=0.5,label='r_eur={:.2f}'.format(corr_coefficient))




for type in eruptivity.unique():
    mask = eruptivity == type
    if type=='E':
      ax6.scatter(recrate[mask],flux[mask],color=amap(1),marker='o',facecolor='none',s=30)
      ax6.scatter(recrate[mask&mask_sep1],flux[mask_sep1],color=amap(1),marker='o',s=30)
      coefficients = np.polyfit(np.log10(recrate[mask]),np.log10(flux[mask]),  1)
      corr_coefficient,p_value=pearsonr(np.log10(flux[mask]),np.log10(recrate[mask]))
      polynomial = np.poly1d(coefficients)
      coefficients_sep = np.polyfit(np.log10(recrate[mask&mask_sep1]), np.log10(flux[mask&mask_sep1]), 1)
      corr_coefficient_sep, p_value_sep = pearsonr(np.log10(recrate[mask&mask_sep1]), np.log10(flux[mask&mask_sep1]))
      polynomial_sep = np.poly1d(coefficients_sep)
      x_fit = np.linspace(np.log10(recrate.min()), np.log10(recrate.max()), 100)
      y_fit = polynomial(x_fit)
      y_fit_sep= polynomial_sep(x_fit)
      ax6.plot(10**x_fit, 10 ** y_fit_sep, color=smap(1), linestyle='dashed', alpha=1, linewidth=1,
               label='r_sep={:.2f}'.format(corr_coefficient_sep))
      ax6.plot(10**x_fit, 10**y_fit, color=amap(1), linestyle='solid',alpha=0.6,linewidth=0.5,label='r_eur={:.2f}'.format(corr_coefficient))




for type in eruptivity.unique():
    mask = eruptivity == type
    if type=='E':
      ax7.scatter(ARarea[mask],flux[mask],color=amap(1),marker='o',facecolor='none',s=30)
      ax7.scatter(ARarea[mask&mask_sep1],flux[mask_sep1],color=amap(1),marker='o')
      coefficients = np.polyfit(np.log10(ARarea[mask]),np.log10(flux[mask]),  1)
      corr_coefficient,p_value=pearsonr(np.log10(flux[mask]),np.log10(ARarea[mask]))
      polynomial = np.poly1d(coefficients)
      coefficients_sep = np.polyfit(np.log10(ARarea[mask&mask_sep1]), np.log10(flux[mask&mask_sep1]), 1)
      corr_coefficient_sep, p_value_sep = pearsonr(np.log10(ARarea[mask&mask_sep1]), np.log10(flux[mask&mask_sep1]))
      polynomial_sep = np.poly1d(coefficients_sep)
      x_fit = np.linspace(np.log10(ARarea.min()), np.log10(ARarea.max()), 100)
      y_fit = polynomial(x_fit)
      y_fit_sep= polynomial_sep(x_fit)
      ax7.plot(10**x_fit, 10 ** y_fit_sep, color=smap(1), linestyle='dashed', alpha=1, linewidth=1,
               label='r_sep={:.2f}'.format(corr_coefficient_sep))
      ax7.plot(10**x_fit, 10**y_fit, color=amap(1), linestyle='solid',alpha=0.6,linewidth=0.5,label='r_eur={:.2f}'.format(corr_coefficient))



for type in eruptivity.unique():
    mask = eruptivity == type
    if type=='E':
      ax8.scatter(Rbarea[mask],flux[mask],color=amap(1),marker='o',facecolor='none',s=30)
      ax8.scatter(Rbarea[mask&mask_sep1],flux[mask_sep1],color=amap(1),marker='o')
      coefficients = np.polyfit(np.log10(Rbarea[mask]),np.log10(flux[mask]),  1)
      corr_coefficient,p_value=pearsonr(np.log10(flux[mask]),np.log10(Rbarea[mask]))
      polynomial = np.poly1d(coefficients)
      coefficients_sep = np.polyfit(np.log10(Rbarea[mask&mask_sep1]), np.log10(flux[mask&mask_sep1]), 1)
      corr_coefficient_sep, p_value_sep = pearsonr(np.log10(Rbarea[mask&mask_sep1]), np.log10(flux[mask&mask_sep1]))
      polynomial_sep = np.poly1d(coefficients_sep)
      x_fit = np.linspace(np.log10(Rbarea.min()), np.log10(Rbarea.max()), 100)
      y_fit = polynomial(x_fit)
      y_fit_sep= polynomial_sep(x_fit)
      ax8.plot(10**x_fit, 10 ** y_fit_sep, color=smap(1), linestyle='dashed', alpha=1, linewidth=1,
               label='r_sep={:.2f}'.format(corr_coefficient_sep))
      ax8.plot(10**x_fit, 10**y_fit, color=amap(1), linestyle='solid',alpha=0.6,linewidth=0.5,label='r_eur={:.2f}'.format(corr_coefficient))



for type in eruptivity.unique():
    mask = eruptivity == type
    if type=='E':
      ax9.scatter(fluxratio[mask],flux[mask],color=amap(1),marker='o',facecolor='none',s=30)
      ax9.scatter(fluxratio[mask&mask_sep1],flux[mask_sep1],color=amap(1),marker='o')
      coefficients = np.polyfit(fluxratio[mask],np.log10(flux[mask]),  1)
      corr_coefficient,p_value=pearsonr(np.log10(flux[mask]),fluxratio[mask])
      polynomial = np.poly1d(coefficients)
      coefficients_sep = np.polyfit(fluxratio[mask&mask_sep1], np.log10(flux[mask&mask_sep1]), 1)
      corr_coefficient_sep, p_value_sep = pearsonr(fluxratio[mask&mask_sep1], np.log10(flux[mask&mask_sep1]))
      polynomial_sep = np.poly1d(coefficients_sep)
      x_fit = np.linspace(fluxratio.min(), fluxratio.max(), 100)
      y_fit = polynomial(x_fit)
      y_fit_sep= polynomial_sep(x_fit)
      ax9.plot(x_fit, 10 ** y_fit_sep, color=smap(1), linestyle='dashed', alpha=1, linewidth=1,
               label='r_sep={:.2f}'.format(corr_coefficient_sep))
      ax9.plot(x_fit, 10**y_fit, color=amap(1), linestyle='solid',alpha=0.6,linewidth=0.5,label='r_eur={:.2f}'.format(corr_coefficient))




for type in eruptivity.unique():
    mask = eruptivity == type
    if type=='E':
      ax10.scatter(arearatio[mask],flux[mask],color=amap(1),marker='o',facecolor='none',s=30)
      ax10.scatter(arearatio[mask&mask_sep1],flux[mask_sep1],color=amap(1),marker='o',s=30)
      coefficients = np.polyfit(arearatio[mask],np.log10(flux[mask]),  1)
      corr_coefficient,p_value=pearsonr(np.log10(flux[mask]),arearatio[mask])
      polynomial = np.poly1d(coefficients)
      coefficients_sep = np.polyfit(arearatio[mask&mask_sep1], np.log10(flux[mask&mask_sep1]), 1)
      corr_coefficient_sep, p_value_sep = pearsonr(arearatio[mask&mask_sep1], np.log10(flux[mask&mask_sep1]))
      polynomial_sep = np.poly1d(coefficients_sep)
      x_fit = np.linspace(arearatio.min(), arearatio.max(), 100)
      y_fit = polynomial(x_fit)
      y_fit_sep= polynomial_sep(x_fit)
      ax10.plot(x_fit, 10 ** y_fit_sep, color=smap(1), linestyle='dashed', alpha=1, linewidth=1,
               label='r_sep={:.2f}'.format(corr_coefficient_sep))
      ax10.plot(x_fit, 10**y_fit, color=amap(1), linestyle='solid',alpha=0.6,linewidth=0.5,label='r_eur={:.2f}'.format(corr_coefficient))



for type in eruptivity.unique():
    mask = eruptivity == type
    if type=='E':
      ax11.scatter(thermal[mask],flux[mask],color=amap(1),marker='o',facecolor='none',s=30)
      ax11.scatter(thermal[mask&mask_sep1],flux[mask_sep1],color=amap(1),marker='o')
      coefficients = np.polyfit(np.log10(thermal[mask]),np.log10(flux[mask]),  1)
      corr_coefficient,p_value=pearsonr(np.log10(flux[mask]),np.log10(thermal[mask]))
      polynomial = np.poly1d(coefficients)
      coefficients_sep = np.polyfit(np.log10(thermal[mask&mask_sep1]), np.log10(flux[mask&mask_sep1]), 1)
      corr_coefficient_sep, p_value_sep = pearsonr(np.log10(thermal[mask&mask_sep1]), np.log10(flux[mask&mask_sep1]))
      polynomial_sep = np.poly1d(coefficients_sep)
      x_fit = np.linspace(np.log10(thermal.min()), np.log10(thermal.max()), 100)
      y_fit = polynomial(x_fit)
      y_fit_sep= polynomial_sep(x_fit)
      ax11.plot(10**x_fit, 10 ** y_fit_sep, color=smap(1), linestyle='dashed', alpha=1, linewidth=1,
               label='r_sep={:.2f}'.format(corr_coefficient_sep))
      ax11.plot(10**x_fit, 10**y_fit, color=amap(1), linestyle='solid',alpha=0.6,linewidth=0.5,label='r_eur={:.2f}'.format(corr_coefficient))

#ax1.set_xscale('log')
ax1.set_yscale('log')
ax1.set_yticks([10**(-5), 10**(-4), 10**(-3)])
ax1.set_xticks([10,15,20,25,30])
ax1.set_ylim(10**(-5.5),1e-3)
ax1.set_yticklabels(['-5', '-4', '-3'])
ax1.set_xticklabels(['10','15','20','25','30'])
ax1.set_xlabel('Peak Temperature [MK]')
ax1.set_ylabel('Flare Peak Flux [W/m^2]')
ax1.legend(fontsize=8)

ax2.set_xscale('log')
ax2.set_yscale('log')
ax2.set_ylim(10**(-5.5),1e-3)
ax2.set_yticks([10**(-5), 10**(-4), 10**(-3)])
ax2.set_xticks([10**0,10**1])
ax2.set_yticklabels(['-5', '-4', '-3'])
ax2.set_xticklabels(['0','1'])
ax2.set_xlabel('EM [10e48 cm-3]')
ax2.set_ylabel('Flare Peak Flux [W/m^2]')
ax2.legend(fontsize=8)


#ax3.set_xscale('log')
ax3.set_yscale('log')
ax3.set_ylim(10**(-5.5),1e-3)
ax3.set_yticks([10**(-5), 10**(-4), 10**(-3)])
ax3.set_xticks([0,200,400])
ax3.set_yticklabels(['-5', '-4', '-3'])
ax3.set_xticklabels(['0','200','400'])
ax3.set_xlabel('GOES duration [min]')
ax3.set_ylabel('Flare Peak Flux [W/m^2]')
ax3.legend(fontsize=8)


ax4.set_xscale('log')
ax4.set_yscale('log')
ax4.set_ylim(10**(-5.5),1e-3)
ax4.set_yticks([10**(-5), 10**(-4), 10**(-3)])
ax4.set_xticks([1e22,1e23])
ax4.set_yticklabels(['-5', '-4', '-3'])
ax4.set_xticklabels(['22','23'])
ax4.set_xlabel('Unsigned AR flux [Mx]')
ax4.set_ylabel('Flare Peak Flux [W/m^2]')
ax4.legend(fontsize=8)


ax5.set_xscale('log')
ax5.set_yscale('log')
ax5.set_ylim(10**(-5.5),1e-3)
ax5.set_yticks([10**(-5), 10**(-4), 10**(-3)])
ax5.set_xticks([1e21,1e22])
ax5.set_yticklabels(['-5', '-4', '-3'])
ax5.set_xticklabels(['22','23'])
ax5.set_xlabel('Reconnection flux [Mx]')
ax5.set_ylabel('Flare Peak Flux [W/m^2]')
ax5.legend(fontsize=8)



ax6.set_xscale('log')
ax6.set_yscale('log')
ax6.set_ylim(10**(-5.5),1e-3)
ax6.set_yticks([10**(-5), 10**(-4), 10**(-3)])
ax6.set_xticks([1e18,1e19,1e20])
ax6.set_yticklabels(['-5', '-4', '-3'])
ax6.set_xticklabels(['18','19','20'])
ax6.set_xlabel('Reconnection flux rate [Mx/s]')
ax6.set_ylabel('Flare Peak Flux [W/m^2]')
ax6.legend(fontsize=8)



ax7.set_xscale('log')
ax7.set_yscale('log')
ax7.set_ylim(10**(-5.5),1e-3)
ax7.set_yticks([10**(-5), 10**(-4), 10**(-3)])
ax7.set_xticks([10**19.5,1e20,10**20.5])
ax7.set_yticklabels(['-5', '-4', '-3'])
ax7.set_xticklabels(['19.5','20','20.5'])
ax7.set_xlabel('AR area [cm^2]')
ax7.set_ylabel('Flare Peak Flux [W/m^2]')
ax7.legend(fontsize=8)


ax8.set_xscale('log')
ax8.set_yscale('log')
ax8.set_ylim(10**(-5.5),1e-3)
ax8.set_yticks([10**(-5), 10**(-4), 10**(-3)])
ax8.set_xticks([10**18.5,1e19,10**19.5])
ax8.set_yticklabels(['-5', '-4', '-3'])
ax8.set_xticklabels(['18.5','19','19.5'])
ax8.set_xlabel('Ribbon area [cm^2]')
ax8.set_ylabel('Flare Peak Flux [W/m^2]')
ax8.legend(fontsize=8)

ax9.set_yscale('log')
ax9.set_ylim(10**(-5.5),1e-3)
ax9.set_yticks([10**(-5), 10**(-4), 10**(-3)])
ax9.set_xticks([0,10,20,30])
ax9.set_yticklabels(['-5', '-4', '-3'])
ax9.set_xticklabels(['0','10','20','30'])
ax9.set_xlabel('Flux ratio [%]')
ax9.set_ylabel('Flare Peak Flux [W/m^2]')
ax9.legend(fontsize=8)

ax10.set_yscale('log')
ax10.set_ylim(10**(-5.5),1e-3)
ax10.set_yticks([10**(-5), 10**(-4), 10**(-3)])
ax10.set_xticks([0,10,20,30])
ax10.set_yticklabels(['-5', '-4', '-3'])
ax10.set_xticklabels(['0','10','20','30'])
ax10.set_xlabel('Area ratio [%]')
ax10.set_ylabel('Flare Peak Flux [W/m^2]')
ax10.legend(fontsize=8)

ax11.set_yscale('log')
ax11.set_xscale('log')
ax11.set_ylim(10**(-5.5),1e-3)
ax11.set_yticks([10**(-5), 10**(-4), 10**(-3)])
ax11.set_xticks([1e29,1e30,1e31])
ax11.set_yticklabels(['-5', '-4', '-3'])
ax11.set_xticklabels(['29','30','31'])
ax11.set_xlabel('Thermal energe [ergs]')
ax11.set_ylabel('Flare Peak Flux [W/m^2]')
ax11.legend(fontsize=8)
plt.show()