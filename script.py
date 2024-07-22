import numpy as np
from scipy.fft import fft
import pandas as pd
import matplotlib.pyplot as plt
file_path = r"C:\Users\13989\Desktop\c5.0.xlsx"
df = pd.read_excel(file_path)
df_sorted = df.sort_values(by=' PEAKFLUX')
df_sorted['group'] = pd.cut(df_sorted[' PEAKFLUX'], bins=100)
# 使用groupby和value_counts函数统计每组中class列中0和1的出现次数
result = df_sorted.groupby('group')['class'].value_counts()
df_result = result.reset_index()
df_result.columns = ['group', 'class', 'count']  # 重命名列
x = np.arange(len(df_result['group'].unique()))  # the label locations
width = 0.25  # the width of the bars
fig = plt.figure(figsize=(16, 10))
ax1 = fig.add_subplot(1, 1, 1)
for class_value in df_result['class'].unique():
    data = df_result[df_result['class'] == class_value]
    '''if class_value == 0:
        plt.bar(x - width/2, data['count'], width, label='Confined')
    else:
        plt.bar(x + width/2, data['count'], width, label='Eruptive')'''
    if class_value == 0:
        ax1.plot(np.log10(x) , data['count'],  label='Confined',marker='o',markersize=4)
    elif class_value ==1:
        ax1.plot(np.log10(x) , data['count'],  label='Eruptive',marker='o',markersize=4)
ax1.set_ylabel('Numbers')
x_ticks = [interval.mid for interval in df_result['group'].unique()]
ax1.grid(axis='y', linestyle='--', linewidth=0.5)
float_list = [float(i) for i in x_ticks]
for i in range(len(float_list)):
    float_list[i]="{:.2e}".format(float_list[i])
ax1.set_xticks(np.log10(x))
#sci_categories = [f'({interval.left*1000:.2}, {interval.right*1000:.2}]' for interval in df_result['group'].unique()]
#ax1.set_xticklabels(sci_categories)
ax1.legend(loc=[0.8,0.90],fontsize=8)
ax1.tick_params(labelsize=8)
ax1.set_xlabel('E-3[W/m^2]')
plt.show()
