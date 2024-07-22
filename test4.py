import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

file_path = r"C:\Users\13989\Desktop\XFlare Catalogue edit.xls"
df = pd.read_excel(file_path)
df_sorted = df.sort_values(by='X-ray class')
df_sorted['group'] = pd.cut(df_sorted['Flux'], bins=20)
# 使用groupby和value_counts函数统计每组中class列中0和1的出现次数
result = df_sorted.groupby('group')['class'].value_counts()
df_result = result.reset_index()
df_result.columns = ['group', 'class', 'count']  # 重命名列
x = np.arange(len(df_result['group'].unique()))  # the label locations
width = 0.3  # the width of the bars
fig, ax = plt.subplots(figsize=(10,8))
datamatrix=np.zeros((20,2))

#ax.bar(x, data1/(data1+data0), width, label='Eruptive')
for class_value in df_result['class'].unique():
    data = df_result[df_result['class'] == class_value]
    if class_value == 0:
        datamatrix[:,0]=data['count']
        #ax.bar(x - width/2, data['count'], width, label='Confined')
    else:
        datamatrix[:,1]=data['count']
        #ax.bar(x + width/2, data['count'], width, label='Eruptive')
ax.bar(x, datamatrix[:,1]/(datamatrix[:,0]+datamatrix[:,1]), width=0.65, label='Eruptive Flares')
ax.set_ylabel('Ratio')
ax.set_title('The type ratio with X-ray flux(Data from 2011/02/15-2023/08/07)')
x_ticks = [interval.mid*10000 for interval in df_result['group'].unique()]
for i in range(len(x_ticks)):
    x_ticks[i]="{:.2}".format(x_ticks[i])
ax.legend()
ax.grid(axis='y', linestyle='--', linewidth=0.5)
float_list = [float(i) for i in x_ticks]
for i in range(len(float_list)):
    float_list[i]="{:.2e}".format(float_list[i])
sci_categories = [f'({interval.left*10000:.2}, {interval.right*10000:.2}]' for interval in df_result['group'].unique()]
ax.set_xticks(x)
#ax.set_xticklabels(sci_categories)
ax.set_xticklabels(x_ticks)
ax.legend(loc=[0.82,0.96],fontsize=8)
ax.set_xlabel('X-ray peak flux[E-4 W/m^2]')
plt.show()
