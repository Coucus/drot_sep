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
df1 = pd.read_excel('/Users/duan/Desktop/sep/class_x.xlsx', sheet_name='Sheet1')
df2 = pd.read_excel('/Users/duan/Desktop/sep/class_m.xlsx', sheet_name='Sheet1')
df3 = pd.read_excel('/Users/duan/Desktop/sep/class_c.xlsx', sheet_name='Sheet1')
df4 = pd.read_csv('/Users/duan/Desktop/sep/MagParDB.csv')
flux1 = df1['Flux']
flux2 = df2['Flux']
flux3 = df3['Flux']
flux4 = df4[' IXPEAK [W/m^2]']
speed1 = df1['CMEspeed(km/s)']
speed2 = df2['CMEspeed(km/s)']
speed3 = df3['CMEspeed(km/s)']
cmespeed = df4['Vcme(km/s)']
sep= df4['SEP']
flux_data_array = np.concatenate([flux1, flux2, flux3])
cme_data_array = np.concatenate([speed1, speed2, speed3])
mask_cme =  ~np.isnan(cmespeed)
mask_sep = sep !=1

flux_cof = np.concatenate([flux4[mask_sep&mask_cme], flux_data_array])
cme_cof = np.concatenate([cmespeed[mask_sep&mask_cme], cme_data_array])
corr_coefficient1, p_value1 = pearsonr(np.log10(flux_cof), cme_cof)
corr_coefficient2, p_value2 = pearsonr(np.log10(flux_data_array), cme_data_array)
coefficients1 = np.polyfit(np.log10(flux_cof),cme_cof,1)
coefficients2 = np.polyfit(np.log10(flux_data_array), cme_data_array, 1)
polynomial1 = np.poly1d(coefficients1)
print(polynomial1)
polynomial2 = np.poly1d(coefficients2)
print(polynomial2)
print(corr_coefficient1,corr_coefficient2)
x_fit = np.linspace(np.log10(flux_data_array).min(), np.log10(flux_data_array).max(), 100)
y_fit1 = polynomial1(x_fit)
y_fit2 = polynomial2(x_fit)

fig= plt.figure(figsize=(20,15))
ax1 = fig.add_subplot(5,3,1)
ax1.scatter(flux4[mask_cme&mask_sep], cmespeed[mask_cme&mask_sep],color=cmap(1),label='Flare only relate CME')
ax1.scatter(flux_data_array,cme_data_array,color=cmap(8),label='Flare relate SEP')
ax1.plot(10**x_fit,y_fit1, color=smap(1),linestyle='--')
ax1.plot(10**x_fit,y_fit2, color=cmap(8),linestyle='--')
ax1.set_title('CME-Flare')
ax1.set_xscale('log')

ax1.set_xticks([10**(-6), 10**(-5), 10**(-4), 10**(-3)])
ax1.set_yticks([0,1600,3200])
ax1.set_xticklabels(['-6', '-5', '-4', '-3'])
ax1.set_yticklabels(['0','2000','4000'])
ax1.set_xlabel('Peak SXR-Flux(log)')
ax1.set_ylabel('Vcme(km/s)')
ax1.legend(fontsize=8)




print(len(flux4[mask_cme&mask_sep]))
print(len(flux_data_array))
plt.show()
