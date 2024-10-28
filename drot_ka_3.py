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
lon= df1['Lon [deg]']
lat= df1['Lat [deg]']
kb=1.380649e-16
thermal=3/np.power(8,0.25)*np.sqrt(EM)*np.power(Rbarea,0.75)*T*kb*10**30
thermal = pd.to_numeric(thermal,errors='coerce')
mask_cme =  ~np.isnan(cmespeed)
mask_sep = sep !=1
mask_sep1 = sep == 1


widthtype = pd.cut(width[mask_cme], bins=bins, labels=labels, right=False)

fig3= plt.figure(figsize=(12,12))
axx1 = fig3.add_subplot(3,3,1)
axx2 = fig3.add_subplot(3,3,2)
axx3 = fig3.add_subplot(3,3,3)
axx4 = fig3.add_subplot(3,3,4)
axx5 = fig3.add_subplot(3,3,5)
axx6 = fig3.add_subplot(3,3,6)
axx7 = fig3.add_subplot(3,3,7)
axx8 = fig3.add_subplot(3,3,8)
axx9 = fig3.add_subplot(3,3,9)

fig2 = plt.figure(figsize=(15,5))
axa = fig2.add_subplot(1,3,1)
axb = fig2.add_subplot(1,3,2)
axc = fig2.add_subplot(1,3,3)


fig= plt.figure(figsize=(20,15))
ax1 = fig.add_subplot(3,4,1)
ax2 = fig.add_subplot(3,4,2)
ax4 = fig.add_subplot(3,4,3)
ax5 = fig.add_subplot(3,4,4)
ax6 = fig.add_subplot(3,4,5)
ax7 = fig.add_subplot(3,4,6)
ax8 = fig.add_subplot(3,4,7)
ax9 = fig.add_subplot(3,4,8)
ax10 = fig.add_subplot(3,4,9)
ax11 = fig.add_subplot(3,4,10)
ax12 = fig.add_subplot(3,4,11)

for type in widthtype.unique():
    mask = widthtype == type
    if type ==1:
       ax1.scatter(EM[mask_cme&mask],dur[mask_cme&mask]/60,color=cmap(5),marker='o',facecolor='none',s=30,label='None Halo')
       ax1.scatter(EM[mask & mask_sep1], dur[mask & mask_sep1]/60, color=cmap(1), marker='o', s=30)
       num_bins = 10
       bin_width = (np.log10(EM[mask_cme].max()) - np.log10(EM[mask_cme].min())) / num_bins
       bins0 = np.arange(np.log10(EM[mask_cme].min()), np.log10(EM[mask_cme].max()) + bin_width, bin_width)
       hist, bin_edges = np.histogram(np.log10(EM[mask_cme&mask]), bins=bins0)
       axx1.bar(bin_edges[:-1], hist/87, width=bin_width, align='edge', color=cmap(5), alpha=0.4, label='None Halo N=87')
       axx1.plot(np.linspace(0, 100) * 0 + np.log10(EM[mask_cme & mask].mean()), np.linspace(0, 100), color=cmap(5),
                linestyle='--')
       mean_none=np.log10(EM[mask_cme & mask].mean())
       mean_none1 = dur[mask_cme&mask].mean()/60
       bin_width1 = (dur[mask_cme&mask].max()/60 - dur[mask_cme&mask].min()/60) / num_bins
       bins1 = np.arange(dur[mask_cme&mask].min()/60, dur[mask_cme&mask].max()/60 + bin_width1, bin_width1)
       hist1, bin_edges1 = np.histogram(dur[mask_cme&mask]/60, bins=bins1)
       axx2.bar(bin_edges1[:-1], hist1/87, width=bin_width1, align='edge',color=cmap(5),alpha=0.4)
       axx2.plot(np.linspace(0,100)*0+dur[mask_cme&mask].mean()/60,np.linspace(0,100),color=cmap(5),linestyle='--')
       coefficients = np.polyfit(np.log10(EM[mask_cme & mask]), dur[mask_cme & mask]/60, 1)
       corr_coefficient_none,p_value_none=pearsonr(np.log10(EM[mask_cme&mask]),dur[mask_cme&mask]/60)
       polynomial = np.poly1d(coefficients)
       x_fit = np.linspace(np.log10(EM[mask_cme].min()), np.log10(EM[mask_cme].max()), 100)
       y_fit = polynomial(x_fit)
       ax1.plot(10**x_fit, y_fit, color=smap(1), linestyle='solid',alpha=1,linewidth=0.5,label='r={:.2f}-None Halo'.format(corr_coefficient_none))
    elif type ==2:
        ax1.scatter(EM[mask_cme & mask], dur[mask_cme & mask]/60, color=cmap(5), marker='^',s=30,
                    label='Halo')
        ax1.scatter(EM[mask&mask_sep1],dur[mask&mask_sep1]/60,color=cmap(1),marker='^',s=30,label='SEP')
        hist, bin_edges = np.histogram(np.log10(EM[mask_cme & mask]), bins=bins0)
        axx1.bar(bin_edges[:-1],hist/36,width=bin_width, align='edge',color=cmap(3), alpha=0.4, label='Halo N=36')
        axx1.plot(np.linspace(0, 100) * 0 + np.log10(EM[mask_cme & mask].mean()), np.linspace(0, 100), color=cmap(3),
                 linestyle='--')
        mean_halo = np.log10(EM[mask_cme & mask].mean())
        hist1, bin_edges1 = np.histogram(dur[mask_cme & mask] / 60, bins=bins1)
        axx2.bar(bin_edges1[:-1], hist1/36, width=bin_width1, align='edge', color=cmap(3), alpha=0.4)
        axx2.plot(np.linspace(0, 100) * 0 + dur[mask_cme & mask].mean() / 60, np.linspace(0, 100), color=cmap(3),
                  linestyle='--')
        mean_halo1 = dur[mask_cme & mask].mean() / 60
        coefficients = np.polyfit(np.log10(EM[mask_cme & mask]), dur[mask_cme & mask]/60, 1)
        corr_coefficient_halo, p_value_halo = pearsonr(dur[mask_cme & mask]/60, np.log10(EM[mask_cme & mask]))
        polynomial = np.poly1d(coefficients)
        y_fit = polynomial(x_fit)
        ax1.plot(10**x_fit, y_fit, color=smap(5), linestyle='solid', alpha=1, linewidth=0.5,
                 label='r={:.2f}-Halo'.format(corr_coefficient_halo))
        axx1.plot([], [], label='Mean difference={:.2f}'.format(mean_halo - mean_none))


for type in widthtype.unique():
    mask = widthtype == type
    if type ==1:
       ax2.scatter(EM[mask_cme&mask],thermal[mask_cme&mask],color=cmap(5),marker='o',facecolor='none',s=30)
       ax2.scatter(EM[mask & mask_sep1], thermal[mask & mask_sep1], color=cmap(1), marker='o', s=30)
       num_bins = 10
       bin_width = (np.log10(thermal[mask_cme].max()) - np.log10(thermal[mask_cme].min())) / num_bins
       bins = np.arange(np.log10(thermal[mask_cme].min()), np.log10(thermal[mask_cme].max()) + bin_width, bin_width)
       hist, bin_edges = np.histogram(np.log10(thermal[mask_cme&mask]), bins=bins)
       mean_none = np.log10(thermal[mask_cme & mask].mean())
       axx3.bar(bin_edges[:-1], hist/87, width=bin_width, align='edge',color=cmap(5),alpha=0.4)
       axx3.plot(np.linspace(0,100)*0+np.log10(thermal[mask_cme&mask].mean()),np.linspace(0,100),color=cmap(5),linestyle='--')
       coefficients = np.polyfit(np.log10(EM[mask_cme & mask]),np.log10(thermal[mask_cme & mask]), 1)
       corr_coefficient_none,p_value_none=pearsonr(np.log10(EM[mask_cme&mask]),np.log10(thermal[mask_cme & mask]))
       polynomial = np.poly1d(coefficients)
       x_fit = np.linspace(np.log10(EM[mask_cme].min()), np.log10(EM[mask_cme].max()), 100)
       y_fit = polynomial(x_fit)
       ax2.plot(10**x_fit,  10**y_fit, color=smap(1), linestyle='solid',alpha=1,linewidth=0.5,label='r={:.2f}-None Halo'.format(corr_coefficient_none))
    elif type ==2:
        ax2.scatter(EM[mask_cme & mask], thermal[mask_cme&mask], color=cmap(5), marker='^',s=30)
        ax2.scatter(EM[mask&mask_sep1],thermal[mask_sep1&mask],color=cmap(1),marker='^',s=30)
        hist, bin_edges = np.histogram(np.log10(thermal[mask_cme&mask]), bins=bins)
        mean_halo = np.log10(thermal[mask_cme & mask].mean())
        axx3.bar(bin_edges[:-1],hist/36,width=bin_width, align='edge',color=cmap(3), alpha=0.4)
        axx3.plot(np.linspace(0, 100) * 0 + np.log10(thermal[mask_cme&mask].mean()), np.linspace(0, 100), color=cmap(3),
                 linestyle='--')
        coefficients = np.polyfit(np.log10(EM[mask_cme & mask]), np.log10(thermal[mask_cme&mask]), 1)
        corr_coefficient_halo, p_value_halo = pearsonr(np.log10(thermal[mask_cme&mask]), np.log10(EM[mask_cme & mask]))
        polynomial = np.poly1d(coefficients)
        y_fit = polynomial(x_fit)
        ax2.plot(10**x_fit, 10**y_fit, color=smap(5), linestyle='solid', alpha=1, linewidth=0.5,
                 label='r={:.2f}-Halo'.format(corr_coefficient_halo))
        axx3.plot([], [], label='Mean difference={:.2f}'.format(mean_halo - mean_none))


for type in widthtype.unique():
    mask = widthtype == type
    if type ==1:
       num_bins = 10
       bin_width = (np.log10(flux[mask_cme].max()) - np.log10(flux[mask_cme].min())) / num_bins
       bins = np.arange(np.log10(flux[mask_cme].min()), np.log10(flux[mask_cme].max()) + bin_width, bin_width)
       hist, bin_edges = np.histogram(np.log10(flux[mask_cme&mask]), bins=bins)
       mean_none = np.log10(flux[mask_cme&mask].mean())
       axx4.bar(bin_edges[:-1], hist/87, width=bin_width, align='edge',color=cmap(5),alpha=0.4)
       axx4.plot(np.linspace(0,100)*0+np.log10(flux[mask_cme&mask].mean()),np.linspace(0,100),color=cmap(5),linestyle='--')
       coefficients = np.polyfit(np.log10(EM[mask_cme & mask]),np.log10(flux[mask_cme & mask]), 1)
       corr_coefficient_none,p_value_none=pearsonr(np.log10(EM[mask_cme&mask]),np.log10(flux[mask_cme & mask]))
       polynomial = np.poly1d(coefficients)
       x_fit = np.linspace(np.log10(EM[mask_cme].min()), np.log10(EM[mask_cme].max()), 100)
       y_fit = polynomial(x_fit)

    elif type ==2:

        hist, bin_edges = np.histogram(np.log10(flux[mask_cme&mask]), bins=bins)
        axx4.bar(bin_edges[:-1],hist/36,width=bin_width, align='edge',color=cmap(3), alpha=0.4)
        axx4.plot(np.linspace(0, 100) * 0 + np.log10(flux[mask_cme&mask].mean()), np.linspace(0, 100), color=cmap(3),
                 linestyle='--')
        coefficients = np.polyfit(np.log10(EM[mask_cme & mask]), np.log10(flux[mask_cme&mask]), 1)
        corr_coefficient_halo, p_value_halo = pearsonr(np.log10(flux[mask_cme&mask]), np.log10(EM[mask_cme & mask]))
        polynomial = np.poly1d(coefficients)
        y_fit = polynomial(x_fit)
        mean_halo = np.log10(flux[mask_cme & mask].mean())
        axx4.plot([], [], label='Mean difference={:.2f}'.format(mean_halo - mean_none))



for type in widthtype.unique():
    mask = widthtype == type
    if type ==1:
       ax4.scatter(EM[mask_cme&mask],Rbarea[mask_cme&mask],color=cmap(5),marker='o',facecolor='none',s=30)
       ax4.scatter(EM[mask & mask_sep1], Rbarea[mask & mask_sep1], color=cmap(1), marker='o', s=30)
       num_bins = 10
       bin_width = (np.log10(Rbarea[mask_cme].max()) - np.log10(Rbarea[mask_cme].min())) / num_bins
       bins = np.arange(np.log10(Rbarea[mask_cme].min()), np.log10(Rbarea[mask_cme].max()) + bin_width, bin_width)
       hist, bin_edges = np.histogram(np.log10(Rbarea[mask_cme&mask]), bins=bins)
       mean_none=np.log10(Rbarea[mask_cme&mask].mean())
       axx5.bar(bin_edges[:-1], hist/87, width=bin_width, align='edge',color=cmap(5),alpha=0.4)
       axx5.plot(np.linspace(0,100)*0+np.log10(Rbarea[mask_cme&mask].mean()),np.linspace(0,100),color=cmap(5),linestyle='--')
       coefficients = np.polyfit(np.log10(EM[mask_cme & mask]),np.log10(Rbarea[mask_cme & mask]), 1)
       corr_coefficient_none,p_value_none=pearsonr(np.log10(EM[mask_cme&mask]),np.log10(Rbarea[mask_cme & mask]))
       polynomial = np.poly1d(coefficients)
       x_fit = np.linspace(np.log10(EM[mask_cme].min()), np.log10(EM[mask_cme].max()), 100)
       y_fit = polynomial(x_fit)
       ax4.plot(10**x_fit,  10**y_fit, color=smap(1), linestyle='solid',alpha=1,linewidth=0.5,label='r={:.2f}-None Halo'.format(corr_coefficient_none))
    elif type ==2:
        ax4.scatter(EM[mask_cme & mask], Rbarea[mask_cme&mask], color=cmap(5), marker='^',s=30)
        ax4.scatter(EM[mask&mask_sep1],Rbarea[mask_sep1&mask],color=cmap(1),marker='^',s=30)
        hist, bin_edges = np.histogram(np.log10(Rbarea[mask_cme&mask]), bins=bins)
        axx5.bar(bin_edges[:-1],hist/36,width=bin_width, align='edge',color=cmap(3), alpha=0.4)
        axx5.plot(np.linspace(0, 100) * 0 + np.log10(Rbarea[mask_cme&mask].mean()), np.linspace(0, 100), color=cmap(3),
                 linestyle='--')
        mean_halo=np.log10(Rbarea[mask_cme&mask].mean())
        coefficients = np.polyfit(np.log10(EM[mask_cme & mask]), np.log10(Rbarea[mask_cme&mask]), 1)
        corr_coefficient_halo, p_value_halo = pearsonr(np.log10(Rbarea[mask_cme&mask]), np.log10(EM[mask_cme & mask]))
        polynomial = np.poly1d(coefficients)
        y_fit = polynomial(x_fit)
        ax4.plot(10**x_fit, 10**y_fit, color=smap(5), linestyle='solid', alpha=1, linewidth=0.5,
                 label='r={:.2f}-Halo'.format(corr_coefficient_halo))
        axx5.plot([], [], label='Mean difference={:.2f}'.format(mean_halo - mean_none))

for type in widthtype.unique():
    mask = widthtype == type
    if type ==1:
       ax5.scatter(EM[mask_cme&mask],recflux[mask_cme&mask],color=cmap(5),marker='o',facecolor='none',s=30)
       ax5.scatter(EM[mask & mask_sep1], recflux[mask & mask_sep1], color=cmap(1), marker='o', s=30)
       num_bins = 10
       bin_width = (np.log10(recflux[mask_cme].max()) - np.log10(recflux[mask_cme].min())) / num_bins
       bins = np.arange(np.log10(recflux[mask_cme].min()), np.log10(recflux[mask_cme].max()) + bin_width, bin_width)
       hist, bin_edges = np.histogram(np.log10(recflux[mask_cme&mask]), bins=bins)
       axx6.bar(bin_edges[:-1], hist/87, width=bin_width, align='edge',color=cmap(5),alpha=0.4)
       axx6.plot(np.linspace(0,100)*0+np.log10(recflux[mask_cme&mask].mean()),np.linspace(0,100),color=cmap(5),linestyle='--')
       mean_none=np.log10(recflux[mask_cme&mask].mean())
       coefficients = np.polyfit(np.log10(EM[mask_cme & mask]),np.log10(recflux[mask_cme & mask]), 1)
       corr_coefficient_none,p_value_none=pearsonr(np.log10(EM[mask_cme&mask]),np.log10(recflux[mask_cme & mask]))
       polynomial = np.poly1d(coefficients)
       x_fit = np.linspace(np.log10(EM[mask_cme].min()), np.log10(EM[mask_cme].max()), 100)
       y_fit = polynomial(x_fit)
       ax5.plot(10**x_fit,  10**y_fit, color=smap(1), linestyle='solid',alpha=1,linewidth=0.5,label='r={:.2f}-None Halo'.format(corr_coefficient_none))
    elif type ==2:
        ax5.scatter(EM[mask_cme & mask], recflux[mask_cme&mask], color=cmap(5), marker='^',s=30)
        ax5.scatter(EM[mask&mask_sep1],recflux[mask_sep1&mask],color=cmap(1),marker='^',s=30)
        hist, bin_edges = np.histogram(np.log10(recflux[mask_cme&mask]), bins=bins)
        axx6.bar(bin_edges[:-1],hist/36,width=bin_width, align='edge',color=cmap(3), alpha=0.4)
        axx6.plot(np.linspace(0, 100) * 0 + np.log10(recflux[mask_cme&mask].mean()), np.linspace(0, 100), color=cmap(3),
                 linestyle='--')
        mean_halo=np.log10(recflux[mask_cme&mask].mean())
        coefficients = np.polyfit(np.log10(EM[mask_cme & mask]), np.log10(recflux[mask_cme&mask]), 1)
        corr_coefficient_halo, p_value_halo = pearsonr(np.log10(recflux[mask_cme&mask]), np.log10(EM[mask_cme & mask]))
        polynomial = np.poly1d(coefficients)
        y_fit = polynomial(x_fit)
        ax5.plot(10**x_fit, 10**y_fit, color=smap(5), linestyle='solid', alpha=1, linewidth=0.5,
                 label='r={:.2f}-Halo'.format(corr_coefficient_halo))
        axx6.plot([], [], label='Mean difference={:.2f}'.format(mean_halo - mean_none))

for type in widthtype.unique():
    mask = widthtype == type
    if type ==1:
       ax6.scatter(EM[mask_cme&mask],fluxratio[mask_cme&mask],color=cmap(5),marker='o',facecolor='none',s=30)
       ax6.scatter(EM[mask & mask_sep1], fluxratio[mask & mask_sep1], color=cmap(1), marker='o', s=30)
       num_bins = 10
       bin_width = (fluxratio[mask_cme].max() - fluxratio[mask_cme].min()) / num_bins
       bins = np.arange(fluxratio[mask_cme].min(), fluxratio[mask_cme].max() + bin_width, bin_width)
       hist, bin_edges = np.histogram(fluxratio[mask_cme&mask], bins=bins)
       mean_none=fluxratio[mask_cme&mask].mean()
       axx7.bar(bin_edges[:-1], hist/87, width=bin_width, align='edge',color=cmap(5),alpha=0.4)
       axx7.plot(np.linspace(0,100)*0+fluxratio[mask_cme&mask].mean(),np.linspace(0,100),color=cmap(5),linestyle='--')
       coefficients = np.polyfit(np.log10(EM[mask_cme & mask]),fluxratio[mask_cme & mask], 1)
       corr_coefficient_none,p_value_none=pearsonr(np.log10(EM[mask_cme&mask]),fluxratio[mask_cme & mask])
       polynomial = np.poly1d(coefficients)
       x_fit = np.linspace(np.log10(EM[mask_cme].min()), np.log10(EM[mask_cme].max()), 100)
       y_fit = polynomial(x_fit)
       ax6.plot(10**x_fit,  y_fit, color=smap(1), linestyle='solid',alpha=1,linewidth=0.5,label='r={:.2f}-None Halo'.format(corr_coefficient_none))
    elif type ==2:
        ax6.scatter(EM[mask_cme & mask], fluxratio[mask_cme&mask], color=cmap(5), marker='^',s=30)
        ax6.scatter(EM[mask&mask_sep1],fluxratio[mask_sep1&mask],color=cmap(1),marker='^',s=30)
        hist, bin_edges = np.histogram(fluxratio[mask_cme&mask], bins=bins)
        axx7.bar(bin_edges[:-1],hist/36,width=bin_width, align='edge',color=cmap(3), alpha=0.4)
        axx7.plot(np.linspace(0, 100) * 0 + fluxratio[mask_cme&mask].mean(), np.linspace(0, 100), color=cmap(3),
                 linestyle='--')
        mean_halo = fluxratio[mask_cme & mask].mean()
        coefficients = np.polyfit(np.log10(EM[mask_cme & mask]), fluxratio[mask_cme&mask], 1)
        corr_coefficient_halo, p_value_halo = pearsonr(fluxratio[mask_cme&mask], np.log10(EM[mask_cme & mask]))
        polynomial = np.poly1d(coefficients)
        y_fit = polynomial(x_fit)
        ax6.plot(10**x_fit, y_fit, color=smap(5), linestyle='solid', alpha=1, linewidth=0.5,
                 label='r={:.2f}-Halo'.format(corr_coefficient_halo))
        axx7.plot([],[],label='Mean difference={:.2f}'.format(mean_halo - mean_none))


for type in widthtype.unique():
    mask = widthtype == type
    if type ==1:
       ax7.scatter(dur[mask_cme & mask]/60,fluxratio[mask_cme&mask],color=cmap(5),marker='o',facecolor='none',s=30)
       ax7.scatter(dur[mask_sep1 & mask]/60, fluxratio[mask & mask_sep1], color=cmap(1), marker='o', s=30)
       num_bins = 10
       bin_width = (T[mask_cme].max() - T[mask_cme].min()) / num_bins
       bins = np.arange(T[mask_cme].min(), T[mask_cme].max() + bin_width, bin_width)
       hist, bin_edges = np.histogram(T[mask_cme&mask], bins=bins)
       axx8.bar(bin_edges[:-1], hist/87, width=bin_width, align='edge',color=cmap(5),alpha=0.4)
       axx8.plot(np.linspace(0,100)*0+T[mask_cme&mask].mean(),np.linspace(0,100),color=cmap(5),linestyle='--')
       mean_none=T[mask_cme&mask].mean()
       coefficients = np.polyfit(dur[mask_cme & mask]/60,fluxratio[mask_cme & mask], 1)
       corr_coefficient_none,p_value_none=pearsonr(dur[mask_cme & mask]/60,fluxratio[mask_cme & mask])
       polynomial = np.poly1d(coefficients)
       x_fit = np.linspace(dur[mask_cme & mask].min()/60, dur[mask_cme & mask].max()/60, 100)
       y_fit = polynomial(x_fit)
       ax7.plot(x_fit,  y_fit, color=smap(1), linestyle='solid',alpha=1,linewidth=0.5,label='r={:.2f}-None Halo'.format(corr_coefficient_none))
    elif type ==2:
        ax7.scatter(dur[mask_cme & mask]/60, fluxratio[mask_cme&mask], color=cmap(5), marker='^',s=30)
        ax7.scatter(dur[mask_sep1 & mask]/60,fluxratio[mask_sep1&mask],color=cmap(1),marker='^',s=30)
        hist, bin_edges = np.histogram(T[mask_cme&mask], bins=bins)
        axx8.bar(bin_edges[:-1], hist/36, width=bin_width, align='edge', color=cmap(3), alpha=0.4)
        axx8.plot(np.linspace(0, 100) * 0 + T[mask_cme & mask].mean(), np.linspace(0, 100), color=cmap(3),
                  linestyle='--')
        mean_halo = T[mask_cme & mask].mean()
        coefficients = np.polyfit(dur[mask_cme & mask]/60, fluxratio[mask_cme&mask], 1)
        corr_coefficient_halo, p_value_halo = pearsonr(fluxratio[mask_cme&mask], dur[mask_cme & mask]/60)
        polynomial = np.poly1d(coefficients)
        y_fit = polynomial(x_fit)
        ax7.plot(x_fit, y_fit, color=smap(5), linestyle='solid', alpha=1, linewidth=0.5,
                 label='r={:.2f}-Halo'.format(corr_coefficient_halo))
        axx8.plot([], [], label='Mean difference={:.2f}'.format(mean_halo - mean_none))


for type in widthtype.unique():
    mask = widthtype == type
    if type ==1:
       ax8.scatter(dur[mask_cme & mask]/60,Rbarea[mask_cme&mask],color=cmap(5),marker='o',facecolor='none',s=30)
       ax8.scatter(dur[mask_sep1 & mask]/60, Rbarea[mask_sep1&mask], color=cmap(1), marker='o', s=30)
       num_bins = 10
       bin_width = (np.log10(Unsigned[mask_cme].max()) - np.log10(Unsigned[mask_cme].min())) / num_bins
       bins = np.arange(np.log10(Unsigned[mask_cme].min()), np.log10(Unsigned[mask_cme].max()) + bin_width, bin_width)
       hist, bin_edges = np.histogram(np.log10(Unsigned[mask_cme&mask]), bins=bins)
       mean_none = np.log10(Unsigned[mask_cme & mask].mean())
       axx9.bar(bin_edges[:-1], hist/87, width=bin_width, align='edge',color=cmap(5),alpha=0.4)
       axx9.plot(np.linspace(0,100)*0+np.log10(Unsigned[mask_cme&mask].mean()),np.linspace(0,100),color=cmap(5),linestyle='--')
       coefficients = np.polyfit(dur[mask_cme & mask]/60,np.log10(Rbarea[mask_cme&mask]), 1)
       corr_coefficient_none,p_value_none=pearsonr(dur[mask_cme & mask]/60,np.log10(Rbarea[mask_cme&mask]))
       polynomial = np.poly1d(coefficients)
       x_fit = np.linspace(dur[mask_cme & mask].min()/60, dur[mask_cme & mask].max()/60, 100)
       y_fit = polynomial(x_fit)
       ax8.plot(x_fit,  10**y_fit, color=smap(1), linestyle='solid',alpha=1,linewidth=0.5,label='r={:.2f}-None Halo'.format(corr_coefficient_none))
    elif type ==2:
        ax8.scatter(dur[mask_cme & mask]/60, Rbarea[mask_cme&mask], color=cmap(5), marker='^',s=30)
        ax8.scatter(dur[mask_sep1 & mask]/60,Rbarea[mask_sep1&mask],color=cmap(1),marker='^',s=30)
        hist, bin_edges = np.histogram(np.log10(Unsigned[mask_cme&mask]), bins=bins)
        mean_halo=np.log10(Unsigned[mask_cme&mask].mean())
        axx9.bar(bin_edges[:-1], hist / 36, width=bin_width, align='edge', color=cmap(3), alpha=0.4)
        axx9.plot(np.linspace(0, 100) * 0 + np.log10(Unsigned[mask_cme&mask].mean()), np.linspace(0, 100), color=cmap(3),
                  linestyle='--')
        coefficients = np.polyfit(dur[mask_cme & mask]/60, np.log10(Rbarea[mask_cme&mask]), 1)
        corr_coefficient_halo, p_value_halo = pearsonr(np.log10(Rbarea[mask_cme&mask]), dur[mask_cme & mask]/60)
        polynomial = np.poly1d(coefficients)
        y_fit = polynomial(x_fit)
        ax8.plot(x_fit, 10**y_fit, color=smap(5), linestyle='solid', alpha=1, linewidth=0.5,
                 label='r={:.2f}-Halo'.format(corr_coefficient_halo))
        axx9.plot([], [], label='Mean difference={:.2f}'.format(mean_halo - mean_none))

for type in widthtype.unique():
    mask = widthtype == type
    if type ==1:
       ax9.scatter(dur[mask_cme & mask]/60,thermal[mask_cme&mask],color=cmap(5),marker='o',facecolor='none',s=30)
       ax9.scatter(dur[mask_sep1 & mask]/60, thermal[mask_sep1&mask], color=cmap(1), marker='o', s=30)
       num_bins = 10
       bin_width = (fluxratio[mask_cme].max() - fluxratio[mask_cme].min()) / num_bins
       bins = np.arange(fluxratio[mask_cme].min(), fluxratio[mask_cme].max() + bin_width, bin_width)
       hist, bin_edges = np.histogram(fluxratio[mask_cme&mask], bins=bins)
       #ax14.bar(bin_edges[:-1], hist, width=bin_width, align='edge',color=cmap(5),alpha=0.4)
       #ax14.plot(np.linspace(0,100)*0+fluxratio[mask_cme&mask].mean(),np.linspace(0,100),color=cmap(5),linestyle='--')
       coefficients = np.polyfit(dur[mask_cme & mask]/60,np.log10(thermal[mask_cme&mask]), 1)
       corr_coefficient_none,p_value_none=pearsonr(dur[mask_cme & mask]/60,np.log10(thermal[mask_cme&mask]))
       polynomial = np.poly1d(coefficients)
       x_fit = np.linspace(dur[mask_cme & mask].min()/60, dur[mask_cme & mask].max()/60, 100)
       y_fit = polynomial(x_fit)
       ax9.plot(x_fit,  10**y_fit, color=smap(1), linestyle='solid',alpha=1,linewidth=0.5,label='r={:.2f}-None Halo'.format(corr_coefficient_none))
    elif type ==2:
        ax9.scatter(dur[mask_cme & mask]/60, thermal[mask_cme&mask], color=cmap(5), marker='^',s=30)
        ax9.scatter(dur[mask_sep1 & mask]/60,thermal[mask_sep1&mask],color=cmap(1),marker='^',s=30)
        hist, bin_edges = np.histogram(fluxratio[mask_cme&mask], bins=bins)
        #ax14.bar(bin_edges[:-1],hist,width=bin_width, align='edge',color=cmap(3), alpha=0.4)
        #ax14.plot(np.linspace(0, 100) * 0 + fluxratio[mask_cme&mask].mean(), np.linspace(0, 100), color=cmap(3),
                 #linestyle='--')
        coefficients = np.polyfit(dur[mask_cme & mask]/60, np.log10(thermal[mask_cme&mask]), 1)
        corr_coefficient_halo, p_value_halo = pearsonr(np.log10(thermal[mask_cme&mask]), dur[mask_cme & mask]/60)
        polynomial = np.poly1d(coefficients)
        y_fit = polynomial(x_fit)
        ax9.plot(x_fit, 10**y_fit, color=smap(5), linestyle='solid', alpha=1, linewidth=0.5,
                 label='r={:.2f}-Halo'.format(corr_coefficient_halo))


for type in widthtype.unique():
    mask = widthtype == type
    if type ==1:
       ax10.scatter(Rbarea[mask_cme&mask],fluxratio[mask_cme&mask],color=cmap(5),marker='o',facecolor='none',s=30)
       ax10.scatter(Rbarea[mask_sep1&mask], fluxratio[mask_sep1&mask],color=cmap(1), marker='o', s=30)
       num_bins = 10
       bin_width = (fluxratio[mask_cme].max() - fluxratio[mask_cme].min()) / num_bins
       bins = np.arange(fluxratio[mask_cme].min(), fluxratio[mask_cme].max() + bin_width, bin_width)
       hist, bin_edges = np.histogram(fluxratio[mask_cme&mask], bins=bins)
       #ax14.bar(bin_edges[:-1], hist, width=bin_width, align='edge',color=cmap(5),alpha=0.4)
       #ax14.plot(np.linspace(0,100)*0+fluxratio[mask_cme&mask].mean(),np.linspace(0,100),color=cmap(5),linestyle='--')
       coefficients = np.polyfit(np.log10(Rbarea[mask_cme&mask]),fluxratio[mask_cme&mask], 1)
       corr_coefficient_none,p_value_none=pearsonr(fluxratio[mask_cme&mask],np.log10(Rbarea[mask_cme&mask]))
       polynomial = np.poly1d(coefficients)
       x_fit = np.linspace(np.log10(Rbarea[mask_cme&mask].min()), np.log10(Rbarea[mask_cme&mask].max()), 100)
       y_fit = polynomial(x_fit)
       ax10.plot(10**x_fit,  y_fit, color=smap(1), linestyle='solid',alpha=1,linewidth=0.5,label='r={:.2f}-None Halo'.format(corr_coefficient_none))
    elif type ==2:
        ax10.scatter(Rbarea[mask_cme&mask],fluxratio[mask_cme&mask],color=cmap(5),marker='^',s=30)
        ax10.scatter(Rbarea[mask_sep1&mask], fluxratio[mask_sep1&mask],color=cmap(1), marker='^', s=30)
        hist, bin_edges = np.histogram(fluxratio[mask_cme&mask], bins=bins)
        #ax14.bar(bin_edges[:-1],hist,width=bin_width, align='edge',color=cmap(3), alpha=0.4)
        #ax14.plot(np.linspace(0, 100) * 0 + fluxratio[mask_cme&mask].mean(), np.linspace(0, 100), color=cmap(3),
                 #linestyle='--')
        coefficients = np.polyfit(np.log10(Rbarea[mask_cme&mask]),fluxratio[mask_cme&mask], 1)
        corr_coefficient_halo,p_value_halo=pearsonr(fluxratio[mask_cme&mask],np.log10(Rbarea[mask_cme&mask]))
        polynomial = np.poly1d(coefficients)
        y_fit = polynomial(x_fit)
        ax10.plot(10**x_fit, y_fit, color=smap(5), linestyle='solid', alpha=1, linewidth=0.5,
                 label='r={:.2f}-Halo'.format(corr_coefficient_halo))


for type in widthtype.unique():
    mask = widthtype == type
    if type ==1:
       ax11.scatter(Rbarea[mask_cme&mask],thermal[mask_cme&mask],color=cmap(5),marker='o',facecolor='none',s=30)
       ax11.scatter(Rbarea[mask_sep1&mask], thermal[mask_sep1&mask],color=cmap(1), marker='o', s=30)
       num_bins = 10
       bin_width = (fluxratio[mask_cme].max() - fluxratio[mask_cme].min()) / num_bins
       bins = np.arange(fluxratio[mask_cme].min(), fluxratio[mask_cme].max() + bin_width, bin_width)
       hist, bin_edges = np.histogram(fluxratio[mask_cme&mask], bins=bins)
       #ax14.bar(bin_edges[:-1], hist, width=bin_width, align='edge',color=cmap(5),alpha=0.4)
       #ax14.plot(np.linspace(0,100)*0+fluxratio[mask_cme&mask].mean(),np.linspace(0,100),color=cmap(5),linestyle='--')
       coefficients = np.polyfit(np.log10(Rbarea[mask_cme&mask]),np.log10(thermal[mask_cme&mask]), 1)
       corr_coefficient_none,p_value_none=pearsonr(np.log10(thermal[mask_cme&mask]),np.log10(Rbarea[mask_cme&mask]))
       polynomial = np.poly1d(coefficients)
       x_fit = np.linspace(np.log10(Rbarea[mask_cme&mask].min()), np.log10(Rbarea[mask_cme&mask].max()), 100)
       y_fit = polynomial(x_fit)
       ax11.plot(10**x_fit,  10**y_fit, color=smap(1), linestyle='solid',alpha=1,linewidth=0.5,label='r={:.2f}-None Halo'.format(corr_coefficient_none))
    elif type ==2:
        ax11.scatter(Rbarea[mask_cme&mask],thermal[mask_cme&mask],color=cmap(5),marker='^',s=30)
        ax11.scatter(Rbarea[mask_sep1&mask], thermal[mask_sep1&mask],color=cmap(1), marker='^', s=30)
        hist, bin_edges = np.histogram(fluxratio[mask_cme&mask], bins=bins)
        #ax14.bar(bin_edges[:-1],hist,width=bin_width, align='edge',color=cmap(3), alpha=0.4)
        #ax14.plot(np.linspace(0, 100) * 0 + fluxratio[mask_cme&mask].mean(), np.linspace(0, 100), color=cmap(3),
                 #linestyle='--')
        coefficients = np.polyfit(np.log10(Rbarea[mask_cme&mask]),np.log10(thermal[mask_cme&mask]), 1)
        corr_coefficient_halo,p_value_halo=pearsonr(np.log10(thermal[mask_cme&mask]),np.log10(Rbarea[mask_cme&mask]))
        polynomial = np.poly1d(coefficients)
        y_fit = polynomial(x_fit)
        ax11.plot(10**x_fit, 10**y_fit, color=smap(5), linestyle='solid', alpha=1, linewidth=0.5,
                 label='r={:.2f}-Halo'.format(corr_coefficient_halo))


for type in widthtype.unique():
    mask = widthtype == type
    if type ==1:
       ax11.scatter(Rbarea[mask_cme&mask],thermal[mask_cme&mask],color=cmap(5),marker='o',facecolor='none',s=30)
       ax11.scatter(Rbarea[mask_sep1&mask], thermal[mask_sep1&mask],color=cmap(1), marker='o', s=30)
       num_bins = 10
       bin_width = (fluxratio[mask_cme].max() - fluxratio[mask_cme].min()) / num_bins
       bins = np.arange(fluxratio[mask_cme].min(), fluxratio[mask_cme].max() + bin_width, bin_width)
       hist, bin_edges = np.histogram(fluxratio[mask_cme&mask], bins=bins)
       #ax14.bar(bin_edges[:-1], hist, width=bin_width, align='edge',color=cmap(5),alpha=0.4)
       #ax14.plot(np.linspace(0,100)*0+fluxratio[mask_cme&mask].mean(),np.linspace(0,100),color=cmap(5),linestyle='--')
       coefficients = np.polyfit(np.log10(Rbarea[mask_cme&mask]),np.log10(thermal[mask_cme&mask]), 1)
       corr_coefficient_none,p_value_none=pearsonr(np.log10(thermal[mask_cme&mask]),np.log10(Rbarea[mask_cme&mask]))
       polynomial = np.poly1d(coefficients)
       x_fit = np.linspace(np.log10(Rbarea[mask_cme&mask].min()), np.log10(Rbarea[mask_cme&mask].max()), 100)
       y_fit = polynomial(x_fit)
       ax11.plot(10**x_fit,  10**y_fit, color=smap(1), linestyle='solid',alpha=1,linewidth=0.5,label='r={:.2f}-None Halo'.format(corr_coefficient_none))
    elif type ==2:
        ax11.scatter(Rbarea[mask_cme&mask],thermal[mask_cme&mask],color=cmap(5),marker='^',s=30)
        ax11.scatter(Rbarea[mask_sep1&mask], thermal[mask_sep1&mask],color=cmap(1), marker='^', s=30)
        hist, bin_edges = np.histogram(fluxratio[mask_cme&mask], bins=bins)
        #ax14.bar(bin_edges[:-1],hist,width=bin_width, align='edge',color=cmap(3), alpha=0.4)
        #ax14.plot(np.linspace(0, 100) * 0 + fluxratio[mask_cme&mask].mean(), np.linspace(0, 100), color=cmap(3),
                 #linestyle='--')
        coefficients = np.polyfit(np.log10(Rbarea[mask_cme&mask]),np.log10(thermal[mask_cme&mask]), 1)
        corr_coefficient_halo,p_value_halo=pearsonr(np.log10(thermal[mask_cme&mask]),np.log10(Rbarea[mask_cme&mask]))
        polynomial = np.poly1d(coefficients)
        y_fit = polynomial(x_fit)
        ax11.plot(10**x_fit, 10**y_fit, color=smap(5), linestyle='solid', alpha=1, linewidth=0.5,
                 label='r={:.2f}-Halo'.format(corr_coefficient_halo))



for type in widthtype.unique():
    mask = widthtype == type
    if type ==1:
       ax12.scatter(thermal[mask_cme&mask],fluxratio[mask_cme&mask],color=cmap(5),marker='o',facecolor='none',s=30)
       ax12.scatter(thermal[mask_sep1&mask], fluxratio[mask_sep1&mask],color=cmap(1), marker='o', s=30)
       num_bins = 10
       bin_width = (fluxratio[mask_cme].max() - fluxratio[mask_cme].min()) / num_bins
       bins = np.arange(fluxratio[mask_cme].min(), fluxratio[mask_cme].max() + bin_width, bin_width)
       hist, bin_edges = np.histogram(fluxratio[mask_cme&mask], bins=bins)
       #ax14.bar(bin_edges[:-1], hist, width=bin_width, align='edge',color=cmap(5),alpha=0.4)
       #ax14.plot(np.linspace(0,100)*0+fluxratio[mask_cme&mask].mean(),np.linspace(0,100),color=cmap(5),linestyle='--')
       coefficients = np.polyfit(np.log10(thermal[mask_cme&mask]),fluxratio[mask_cme&mask], 1)
       corr_coefficient_none,p_value_none=pearsonr(np.log10(thermal[mask_cme&mask]),np.log10(Rbarea[mask_cme&mask]))
       polynomial = np.poly1d(coefficients)
       x_fit = np.linspace(np.log10(thermal[mask_cme&mask].min()), np.log10(thermal[mask_cme&mask].max()), 100)
       y_fit = polynomial(x_fit)
       ax12.plot(10**x_fit,  y_fit, color=smap(1), linestyle='solid',alpha=1,linewidth=0.5,label='r={:.2f}-None Halo'.format(corr_coefficient_none))
    elif type ==2:
        ax12.scatter(thermal[mask_cme&mask],fluxratio[mask_cme&mask],color=cmap(5),marker='^',s=30)
        ax12.scatter(thermal[mask_sep1&mask], fluxratio[mask_sep1&mask],color=cmap(1), marker='^', s=30)
        hist, bin_edges = np.histogram(fluxratio[mask_cme&mask], bins=bins)
        #ax14.bar(bin_edges[:-1],hist,width=bin_width, align='edge',color=cmap(3), alpha=0.4)
        #ax14.plot(np.linspace(0, 100) * 0 + fluxratio[mask_cme&mask].mean(), np.linspace(0, 100), color=cmap(3),
                 #linestyle='--')
        coefficients = np.polyfit(np.log10(thermal[mask_cme&mask]),fluxratio[mask_cme&mask], 1)
        corr_coefficient_halo,p_value_halo=pearsonr(np.log10(thermal[mask_cme&mask]),fluxratio[mask_cme&mask])
        polynomial = np.poly1d(coefficients)
        y_fit = polynomial(x_fit)
        ax12.plot(10**x_fit, y_fit, color=smap(5), linestyle='solid', alpha=1, linewidth=0.5,
                 label='r={:.2f}-Halo'.format(corr_coefficient_halo))


for type in widthtype.unique():
    mask = widthtype == type
    if type ==1:
        axa.scatter(flux[mask&mask_cme],lat[mask&mask_cme],color=cmap(5),marker='o',facecolor='none',s=30,label='None Halo')
        axb.scatter(flux[mask&mask_cme],lon[mask&mask_cme],color=cmap(5),marker='o',facecolor='none',s=30)
        axc.scatter(lon[mask&mask_cme],lat[mask&mask_cme],color=cmap(5),marker='o',facecolor='none',s=30)
    if type ==2:
        axa.scatter(flux[mask&mask_cme],lat[mask&mask_cme],color=cmap(5),marker='^',s=30,label='Halo')
        axb.scatter(flux[mask&mask_cme],lon[mask&mask_cme],color=cmap(5),marker='^',s=30)
        axc.scatter(lon[mask&mask_cme],lat[mask&mask_cme],color=cmap(5),marker='^',s=30)

axa.set_xscale('log')
axa.set_xlabel('Peak Flare Flux [W/m^2]')
axa.set_ylabel('Lat [deg]')
axa.legend(fontsize=8)
axb.set_xlabel('Peak Flare Flux [W/m^2]')
axb.set_ylabel('Lon [deg]')
axb.set_xscale('log')
axc.set_ylabel('Lat [deg]')
axc.set_xlabel('Lon [deg]')

ax1.set_xscale('log')
#ax1.set_xticks([0,1])
ax1.set_yticks([0,200,400])
#ax1.set_xticklabels(['0', '1'])
ax1.set_yticklabels(['0','200','400'])
ax1.set_xlabel('Peak EM [1e48 cm-3]')
ax1.set_ylabel('GOES duration [min]')
ax1.legend(fontsize=8)


ax2.set_xscale('log')
ax2.set_yscale('log')
ax2.set_yticks([1e29,1e30,1e31])
#ax1.set_xticklabels(['0', '1'])
ax2.set_xlabel('Peak EM [1e48 cm-3]')
ax2.set_ylabel('Thermal Energe [ergs]')
ax2.legend(fontsize=8)






ax4.set_xscale('log')
ax4.set_yscale('log')
#ax3.set_yticks([1e29,1e30,1e31])
#ax1.set_xticklabels(['0', '1'])
ax4.set_xlabel('Peak EM [1e48 cm-3]')
ax4.set_ylabel('Ribbon area [cm^2]')
ax4.legend(fontsize=8)




ax5.set_xscale('log')
ax5.set_yscale('log')
#ax1.set_xticks([0,1])
#ax1.set_xticklabels(['0', '1'])
#ax5.set_yticklabels(['0','200','400'])
ax5.set_xlabel('Peak EM [1e48 cm-3]')
ax5.set_ylabel('Reconnection flux [Mx]')
ax5.legend(fontsize=8)








ax6.set_xscale('log')
ax6.set_ylim(0,50)
#ax1.set_xticks([0,1])
#ax1.set_xticklabels(['0', '1'])
#ax5.set_yticklabels(['0','200','400'])
ax6.set_xlabel('Peak EM [1e48 cm-3]')
ax6.set_ylabel('Flux ratio [%]')
ax6.legend(fontsize=8)






ax7.set_ylim(0,50)
#ax1.set_xticks([0,1])
#ax1.set_xticklabels(['0', '1'])
#ax5.set_yticklabels(['0','200','400'])
ax7.set_xlabel('GOES duration [min]')
ax7.set_ylabel('Flux ratio [%]')
ax7.legend(fontsize=8)




ax8.set_yscale('log')
#ax1.set_xticks([0,1])
#ax1.set_xticklabels(['0', '1'])
ax8.set_yticks([1e18,1e19])
ax8.set_ylabel('Ribbon area [cm^2]')
ax8.set_xlabel('GOES duration [min]')
ax8.legend(fontsize=8)


ax9.set_yscale('log')
#ax3.set_yticks([1e29,1e30,1e31])
#ax1.set_xticklabels(['0', '1'])
ax9.set_xlabel('GOES duration [min]')
ax9.set_ylabel('Thermal Energe [ergs]')
ax9.legend(fontsize=8)



ax10.set_xscale('log')
ax10.set_ylim(0,50)
#ax3.set_yticks([1e29,1e30,1e31])
#ax1.set_xticklabels(['0', '1'])
ax10.set_xlabel('Ribbon area [cm^2]')
ax10.set_ylabel('Flux ratio [%]')
ax10.set_xticks([1e18,1e19])
ax10.legend(fontsize=8)


ax11.set_xscale('log')
ax11.set_yscale('log')
ax11.set_xticks([1e18,1e19])
ax11.set_xlabel('Ribbon area [cm^2]')
ax11.set_ylabel('Thermal Energe [ergs]')
ax11.legend(fontsize=8)


ax12.set_xscale('log')
ax12.set_ylim(0,50)
ax12.set_xticks([1e29,1e30,1e31])
ax12.set_ylabel('Flux ratio [%]')
ax12.set_xlabel('Thermal Energe [ergs]')
ax12.legend(fontsize=8)

axx1.set_ylim(0,1)

#ax1.set_xticklabels(['0', '1'])
#ax5.set_yticklabels(['0','200','400'])
axx1.set_xlabel('Peak EM [1e48 cm-3]')
axx1.set_ylabel('Count')
axx1.legend(fontsize=8)



axx2.plot([],[],label='Mean difference={:.2f}'.format(mean_halo1-mean_none1))
axx2.set_ylim(0,1)
axx2.set_ylabel('Count')
axx2.set_xlabel('GOES duration [min]')
axx2.legend(fontsize=8)


axx3.set_ylim(0,1)
axx3.set_ylabel('Count')
axx3.set_xlabel('Thermal Energe [ergs]')
axx3.legend(fontsize=8)



axx4.set_ylim(0,1)
axx4.set_ylabel('Count')
axx4.set_xlabel('Flare peak X-ray flux[W/m^2]')
axx4.legend(fontsize=8)

axx5.set_ylim(0,1)
axx5.set_ylabel('Count')
axx5.set_xlabel('Ribbon area [cm^2]')
axx5.legend(fontsize=8)


axx6.set_ylim(0,1)
axx6.set_ylabel('Count')
axx6.set_xlabel('Reconnection flux [Mx]')
axx6.legend(fontsize=8)

axx7.set_ylim(0,1)
axx7.set_ylabel('Count')
axx7.set_xlabel('Flux ratio [%]')
axx7.legend(fontsize=8)

axx8.set_ylim(0,1)
axx8.set_ylabel('Count')
axx8.set_xlabel('Peak Temperature [MK]')
axx8.legend(fontsize=8)


axx9.set_ylim(0,1)
axx9.set_ylabel('Count')
axx9.set_xlabel('Unsigned AR flux [Mx]')
axx9.legend(fontsize=8)
plt.show()