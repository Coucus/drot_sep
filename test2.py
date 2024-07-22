import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
file_path = r"/Users/duan/Desktop/数据/c5.0.xlsx"
custom_bins = [3e-6,4e-6,6e-6,8e-6,1e-5,2e-5,3e-5,5e-5,7e-5,1e-4,1e-3,5e-3, np.inf]
df = pd.read_excel(file_path)
df_sorted = df.sort_values(by=' PEAKFLUX')
df_sorted['group'] = pd.cut(df_sorted[' PEAKFLUX'], bins=custom_bins)
# 统计每个分组中 class_value 等于 0 和 1 的总数
result = df_sorted.groupby('group')['class'].value_counts().unstack().fillna(0)
result['total'] = result.sum(axis=1)  # 新增一列，表示总数

x = np.arange(len(result.index))  # the label locations
fig = plt.figure(figsize=(16, 10))
ax1 = fig.add_subplot(1, 1, 1)
#fig1 = plt.figure(figsize=(16, 10))
#ax2=fig1.add_subplot(1, 1, 1)
x_ticks = [interval.mid for interval in result.index]
#size_multiplier2 = 200
#size_multiplier1= 10
result['total'] = result['total'].replace(0, -1)
print(result)
#ax2.scatter(np.log10(x_ticks), result[1], label='Eruptive',s=result[1] * size_multiplier1,edgecolor='blue',facecolor='white')
#ax2.scatter(np.log10(x_ticks), result[0], label='Confined',s=result[1] * size_multiplier1,edgecolor='red',facecolor='white')
ax1.plot(np.log10(x_ticks), result[1]/result['total'], label='Eruptive ratio', marker='o',linestyle='--',linewidth=1.2,markersize=5)
#ax1.scatter(np.log10(x_ticks), result[0]/result['total'], label='Confined ratio', marker='o',s= size_multiplier2,edgecolor='red',facecolor='red')
ax1.set_ylabel('Numbers')
ax1.grid(axis='y', linestyle='--', linewidth=0.5)
ax1.tick_params(labelsize=8)
#sci_categories = [f'({interval.left*1000:.2}, {interval.right*1000:.2}]' for interval in result.index]
#ax1.set_xticklabels(sci_categories)
#ax1.set_xticks(np.arange(-6, -1, 1), [f'10^{i}' for i in range(-6, -1)])
#ax1.legend(loc=[0.85, 0.80], fontsize=7,prop={'size': 11})

#ax1.set_xticks(np.arange(-6, -1, 1), [f'10^{i}' for i in range(-6, -1)])
# 设置需要标记的刻度值对应的标签
#ax2.legend(loc=[0.85, 0.80], fontsize=7,prop={'size': 11})
custom_xticks1 = np.linspace(1,10,10)*10e-6
custom_xticks2 = np.linspace(1,10,10)*10e-5
custom_xticks3 = np.linspace(1,10,10)*10e-4
result_ticks=np.concatenate((custom_xticks1,custom_xticks2,custom_xticks3),axis=0)  # 自定义刻度位置
ax1.set_xticks(np.log10(result_ticks))

# 使用 FormatStrFormatter 设置刻度标签格式为一位小数
#ax1.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
ax1.xaxis.set_major_formatter(FormatStrFormatter("$10^{%.1f}$"))
ax1.set_xlabel('10^[W/m^2]')
ax1.set_xticklabels([f'$10^{i:.0f}$' if i in [-5,-4,-3] else '' for i in np.log10(result_ticks)])
print(result_ticks)
plt.show()


