import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl


smap = mpl.cm.get_cmap("Set1", 10)

# 读取Excel文件
df1 = pd.read_excel('/Users/duan/Desktop/sep/class_c.xlsx', sheet_name='Sheet1')
df2 = pd.read_excel('/Users/duan/Desktop/sep/class_m.xlsx', sheet_name='Sheet1')
df3 = pd.read_excel('/Users/duan/Desktop/sep/class_x.xlsx', sheet_name='Sheet1')

flux1 = df1['Flux']
type1 = df1['ARtype']
d1ch1 = df1['FisrtPeak13-16Mev/(cm^2 sr MeV s)']
d1ch2 = df1['FisrtPeak16 - 20MeV/(cm^2 sr MeV s)']
d1ch3 = df1['FisrtPeak20 - 25 MeV/(cm^2 sr MeV s)']
d1ch4 = df1['FisrtPeak25 - 32 MeV/(cm^2 sr MeV s)']
d1ch5 = df1['FisrtPeak32 - 40 MeV/(cm^2 sr MeV s)']
d1ch6 = df1['FisrtPeak40 - 50 MeV/(cm^2 sr MeV s)']
d1ch7 = df1['FisrtPeak50 - 64 MeV/(cm^2 sr MeV s)']
d1ch8 = df1['FisrtPeak64 - 80 MeV/(cm^2 sr MeV s)']
d1ch9 = df1['FisrtPeak80 - 100 MeV/(cm^2 sr MeV s)']
d1ch10 = df1['FisrtPeak100 - 130 MeV/(cm^2 sr MeV s)']
flux2 = df2['Flux']
type2 = df2['ARtype']
d2ch1 = df2['FisrtPeak13-16Mev/(cm^2 sr MeV s)']
d2ch2 = df2['FisrtPeak16 - 20MeV/(cm^2 sr MeV s)']
d2ch3 = df2['FisrtPeak20 - 25 MeV/(cm^2 sr MeV s)']
d2ch4 = df2['FisrtPeak25 - 32 MeV/(cm^2 sr MeV s)']
d2ch5 = df2['FisrtPeak32 - 40 MeV/(cm^2 sr MeV s)']
d2ch6 = df2['FisrtPeak40 - 50 MeV/(cm^2 sr MeV s)']
d2ch7 = df2['FisrtPeak50 - 64 MeV/(cm^2 sr MeV s)']
d2ch8 = df2['FisrtPeak64 - 80 MeV/(cm^2 sr MeV s)']
d2ch9 = df2['FisrtPeak80 - 100 MeV/(cm^2 sr MeV s)']
d2ch10 = df2['FisrtPeak100 - 130 MeV/(cm^2 sr MeV s)']
flux3 = df3['Flux']
type3 = df3['ARtype']
d3ch1 = df3['FisrtPeak13-16Mev/(cm^2 sr MeV s)']
d3ch2 = df3['FisrtPeak16 - 20MeV/(cm^2 sr MeV s)']
d3ch3 = df3['FisrtPeak20 - 25 MeV/(cm^2 sr MeV s)']
d3ch4 = df3['FisrtPeak25 - 32 MeV/(cm^2 sr MeV s)']
d3ch5 = df3['FisrtPeak32 - 40 MeV/(cm^2 sr MeV s)']
d3ch6 = df3['FisrtPeak40 - 50 MeV/(cm^2 sr MeV s)']
d3ch7 = df3['FisrtPeak50 - 64 MeV/(cm^2 sr MeV s)']
d3ch8 = df3['FisrtPeak64 - 80 MeV/(cm^2 sr MeV s)']
d3ch9 = df3['FisrtPeak80 - 100 MeV/(cm^2 sr MeV s)']
d3ch10 = df3['FisrtPeak100 - 130 MeV/(cm^2 sr MeV s)']

# 假设你要绘制的两列数据的列名分别是'Column1'和'Column2'
flux_data_array = pd.concat([flux1, flux2, flux3])
type_data_array = pd.concat([type1, type2, type3])
ch1_data_array = pd.concat([d1ch1, d2ch1, d3ch1])
ch2_data_array = pd.concat([d1ch2, d2ch2, d3ch2])
ch3_data_array = pd.concat([d1ch3, d2ch3, d3ch3])
ch4_data_array = pd.concat([d1ch4, d2ch4, d3ch4])
ch5_data_array = pd.concat([d1ch5, d2ch5, d3ch5])
ch6_data_array = pd.concat([d1ch6, d2ch6, d3ch6])
ch7_data_array = pd.concat([d1ch7, d2ch7, d3ch7])
ch8_data_array = pd.concat([d1ch8, d2ch8, d3ch8])
ch9_data_array = pd.concat([d1ch9, d2ch9, d3ch9])
ch10_data_array = pd.concat([d1ch10, d2ch10, d3ch10])
# 绘制散点图
fig = plt.figure(figsize=(20, 10))
ax1 = fig.add_subplot(2, 5, 1)
ax2 = fig.add_subplot(2, 5, 2)
ax3 = fig.add_subplot(2, 5, 3)
ax4 = fig.add_subplot(2, 5, 4)
ax5 = fig.add_subplot(2, 5, 5)
ax6 = fig.add_subplot(2, 5, 6)
ax7 = fig.add_subplot(2, 5, 7)
ax8 = fig.add_subplot(2, 5, 8)
ax9 = fig.add_subplot(2, 5, 9)
ax10 = fig.add_subplot(2, 5, 10)
# 设置图表标题和坐标轴标签

mask1 =  ~np.isnan(ch1_data_array)
mask2 =  ~np.isnan(ch2_data_array)
mask3 =  ~np.isnan(ch3_data_array)
mask4 =  ~np.isnan(ch4_data_array)
mask5 =  ~np.isnan(ch5_data_array)
mask6 =  ~np.isnan(ch6_data_array)
mask7 =  ~np.isnan(ch7_data_array)
mask8 =  ~np.isnan(ch8_data_array)
mask9 =  ~np.isnan(ch9_data_array)
mask10 =  ~np.isnan(ch10_data_array)
for type in type_data_array.unique():
    mask = type_data_array == type
    ax1.scatter(flux_data_array[mask&mask1], ch1_data_array[mask&mask1],color=smap(type),label='ARtype{}'.format(type))
    ax2.scatter(flux_data_array[mask&mask2], ch2_data_array[mask&mask2],color=smap(type))
    ax3.scatter(flux_data_array[mask&mask3], ch3_data_array[mask&mask3],color=smap(type))
    ax4.scatter(flux_data_array[mask&mask4], ch4_data_array[mask&mask4],color=smap(type))
    ax5.scatter(flux_data_array[mask&mask5], ch5_data_array[mask&mask5],color=smap(type))
    ax6.scatter(flux_data_array[mask&mask6], ch6_data_array[mask&mask6],color=smap(type))
    ax7.scatter(flux_data_array[mask&mask7], ch7_data_array[mask&mask7],color=smap(type))
    ax8.scatter(flux_data_array[mask&mask8], ch8_data_array[mask&mask8],color=smap(type))
    ax9.scatter(flux_data_array[mask&mask9], ch9_data_array[mask&mask9],color=smap(type))
    ax10.scatter(flux_data_array[mask&mask10], ch10_data_array[mask&mask10],color=smap(type))

ax1.set_title('Peak Protons Flux of 13-16Mev')
ax1.set_xscale('log')
ax1.set_yscale('log')
ax1.set_xticks([10 ** (-6), 10 ** (-5), 10 ** (-4), 10 ** (-3)])
ax1.set_yticks([10 ** (-4), 10 ** (-2.5), 10 ** (-1), 10 ** (0.5), 10 ** (1.5)])
ax1.set_xticklabels(['-6', '-5', '-4', '-3'])
ax1.set_yticklabels(['-4', '-2.5', '-1', '0.5', '1.5'])
ax1.set_xlabel('X-ray Flux(log)')
ax1.set_ylabel('Flux(log)/(cm^2 sr MeV s)')
ax1.legend(fontsize=9)

ax2.set_title('Peak Protons Flux of 16-20Mev')
ax2.set_xscale('log')
ax2.set_yscale('log')
ax2.set_xticks([10 ** (-6), 10 ** (-5), 10 ** (-4), 10 ** (-3)])
ax2.set_yticks([10 ** (-4), 10 ** (-2.5), 10 ** (-1), 10 ** (0.5), 10 ** (1.5)])
ax2.set_xticklabels(['-6', '-5', '-4', '-3'])
ax2.set_yticklabels(['-4', '-2.5', '-1', '0.5', '1.5'])
ax2.set_xlabel('X-ray Flux(log)')
ax2.set_ylabel('Flux(log)/(cm^2 sr MeV s)')


ax3.set_title('Peak Protons Flux of 20-25Mev')
ax3.set_xscale('log')
ax3.set_yscale('log')
ax3.set_xticks([10 ** (-6), 10 ** (-5), 10 ** (-4), 10 ** (-3)])
ax3.set_yticks([10 ** (-4), 10 ** (-2.5), 10 ** (-1), 10 ** (0.5), 10 ** (1.5)])
ax3.set_xticklabels(['-6', '-5', '-4', '-3'])
ax3.set_yticklabels(['-4', '-2.5', '-1', '0.5', '1.5'])
ax3.set_xlabel('X-ray Flux(log)')
ax3.set_ylabel('Flux(log)/(cm^2 sr MeV s)')


ax4.set_title('Peak Protons Flux of 25-32Mev')
ax4.set_xscale('log')
ax4.set_yscale('log')
ax4.set_xticks([10 ** (-6), 10 ** (-5), 10 ** (-4), 10 ** (-3)])
ax4.set_yticks([10 ** (-4), 10 ** (-2.5), 10 ** (-1), 10 ** (0.5), 10 ** (1.5)])
ax4.set_xticklabels(['-6', '-5', '-4', '-3'])
ax4.set_yticklabels(['-4', '-2.5', '-1', '0.5', '1.5'])
ax4.set_xlabel('X-ray Flux(log)')
ax4.set_ylabel('Flux(log)/(cm^2 sr MeV s)')


ax5.set_title('Peak Protons Flux of 32-40Mev')
ax5.set_xscale('log')
ax5.set_yscale('log')
ax5.set_xticks([10 ** (-6), 10 ** (-5), 10 ** (-4), 10 ** (-3)])
ax5.set_yticks([10 ** (-4), 10 ** (-2.5), 10 ** (-1), 10 ** (0.5), 10 ** (1.5)])
ax5.set_xticklabels(['-6', '-5', '-4', '-3'])
ax5.set_yticklabels(['-4', '-2.5', '-1', '0.5', '1.5'])
ax5.set_xlabel('X-ray Flux(log)')
ax5.set_ylabel('Flux(log)/(cm^2 sr MeV s)')


ax6.set_title('Peak Protons Flux of 40-50Mev')
ax6.set_xscale('log')
ax6.set_yscale('log')
ax6.set_xticks([10 ** (-6), 10 ** (-5), 10 ** (-4), 10 ** (-3)])
ax6.set_yticks([10 ** (-4), 10 ** (-2.5), 10 ** (-1), 10 ** (0.5), 10 ** (1.5)])
ax6.set_xticklabels(['-6', '-5', '-4', '-3'])
ax6.set_yticklabels(['-4', '-2.5', '-1', '0.5', '1.5'])
ax6.set_xlabel('X-ray Flux(log)')
ax6.set_ylabel('Flux(log)/(cm^2 sr MeV s)')

ax7.set_title('Peak Protons Flux of 50-64Mev')
ax7.set_xscale('log')
ax7.set_yscale('log')
ax7.set_xticks([10 ** (-6), 10 ** (-5), 10 ** (-4), 10 ** (-3)])
ax7.set_yticks([10 ** (-4), 10 ** (-2.5), 10 ** (-1), 10 ** (0.5), 10 ** (1.5)])
ax7.set_xticklabels(['-6', '-5', '-4', '-3'])
ax7.set_yticklabels(['-4', '-2.5', '-1', '0.5', '1.5'])
ax7.set_xlabel('X-ray Flux(log)')
ax7.set_ylabel('Flux(log)/(cm^2 sr MeV s)')


ax8.set_title('Peak Protons Flux of 64-80Mev')
ax8.set_xscale('log')
ax8.set_yscale('log')
ax8.set_xticks([10 ** (-6), 10 ** (-5), 10 ** (-4), 10 ** (-3)])
ax8.set_yticks([10 ** (-4), 10 ** (-2.5), 10 ** (-1), 10 ** (0.5), 10 ** (1.5)])
ax8.set_xticklabels(['-6', '-5', '-4', '-3'])
ax8.set_yticklabels(['-4', '-2.5', '-1', '0.5', '1.5'])
ax8.set_xlabel('X-ray Flux(log)')
ax8.set_ylabel('Flux(log)/(cm^2 sr MeV s)')


ax9.set_title('Peak Protons Flux of 80-100Mev')
ax9.set_xscale('log')
ax9.set_yscale('log')
ax9.set_xticks([10 ** (-6), 10 ** (-5), 10 ** (-4), 10 ** (-3)])
ax9.set_yticks([10 ** (-4), 10 ** (-2.5), 10 ** (-1), 10 ** (0.5), 10 ** (1.5)])
ax9.set_xticklabels(['-6', '-5', '-4', '-3'])
ax9.set_yticklabels(['-4', '-2.5', '-1', '0.5', '1.5'])
ax9.set_xlabel('X-ray Flux(log)')
ax9.set_ylabel('Flux(log)/(cm^2 sr MeV s)')


ax10.set_title('Peak Protons Flux of 100-130Mev')
ax10.set_xscale('log')
ax10.set_yscale('log')
ax10.set_xticks([10 ** (-6), 10 ** (-5), 10 ** (-4), 10 ** (-3)])
ax10.set_yticks([10 ** (-4), 10 ** (-2.5), 10 ** (-1), 10 ** (0.5), 10 ** (1.5)])
ax10.set_xticklabels(['-6', '-5', '-4', '-3'])
ax10.set_yticklabels(['-4', '-2.5', '-1', '0.5', '1.5'])
ax10.set_xlabel('X-ray Flux(log)')
ax10.set_ylabel('Flux(log)/(cm^2 sr MeV s)')

plt.subplots_adjust(wspace=0.4, hspace=0.3)
# 显示图表
plt.show()