import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
file_path = r"C:\Users\13989\Desktop\1996-2011.xlsx"
file_path1 = r"C:\Users\13989\Desktop\97-24s.txt"
file_path2 = r"C:\Users\13989\Desktop\97-24.txt"
df = pd.read_excel(file_path, sheet_name="Sheet1")
df1 = pd.read_csv(file_path1, sep='\\s+', on_bad_lines='skip')
df2= pd.read_csv(file_path2, sep='\\s+', on_bad_lines='skip')
time,time1= df1.iloc[:, 2],df2.iloc[:, 2]
number,number1=df1.iloc[:, 3],df2.iloc[:, 3]
timescale = np.arange(1997, 2025)
flare_matrix = np.zeros((28, 6), dtype=float)
flare_matrix[:, 0] = timescale
for i, j in zip(range(1997, 2025), range(29)):
    filtered_df = df[df['start'].dt.year.between(i, i)]
    class_counts = filtered_df['class'].value_counts()
    flare_matrix[j, 1] = class_counts.get(1, 0)
    flare_matrix[j, 2] = class_counts.get(0, 0)
    flare_matrix[j, 5] = class_counts.get(-1, 0)
    flare_matrix[j,4]=flare_matrix[j, 2]+flare_matrix[j, 1]+flare_matrix[j, 5]
    if flare_matrix[j, 1] == 0:
        flare_matrix[j, 3] = 0 
    elif flare_matrix[j, 2] == 0:
        flare_matrix[j, 3] = 1
    else:
        flare_matrix[j, 3] = flare_matrix[j, 1] / (flare_matrix[j, 1] + flare_matrix[j, 2])
fig = plt.figure(figsize=(16, 10))
ax1 = fig.add_subplot(2, 1, 1)
ax2 = fig.add_subplot(2, 1, 2)
ax1.plot(timescale, flare_matrix[:, 1], label='Eruptive', linestyle='-.', marker='o',markersize=4)
ax1.plot(timescale, flare_matrix[:, 2], label='Confined', linestyle='--', marker='o',markersize=4)
ax1.plot(timescale, flare_matrix[:, 4], label='Total numbers', linestyle='-', marker='o',markersize=4)
ax3 = ax1.twinx()
ax3.plot(timescale, flare_matrix[:, 3],color='r',linewidth=1.2,linestyle=':',label='Eruptive ratio',marker='o',markersize=2)
ax3.set_ylabel('Eruptive ratio')
ax1.grid(axis='y', linestyle='--', linewidth=0.3)
ax1.set_ylabel('Numbers')
ax1.set_xlabel('Years')
ax1.legend(loc=[0.67,0.82],fontsize=7)
ax3.legend(loc=[0.76,0.90],fontsize=7)
ax1.set_title('Data from 1997/11/04-2024/02/22')
ax1.set_xticks(np.arange(1997,2025))
ax1.set_yticks(np.arange(0,22))
ax2.plot(time,number,label='Smoothed monthly sunspot numbers')
ax2.plot(time1,number1,label='monthly sunspot numbers',linestyle='--',linewidth=1.2)
ax2.grid(axis='y', linestyle='--', linewidth=0.5)
ax2.set_ylabel('Numbers')
ax2.set_xlabel('Years')
ax2.legend(loc=[0.67,0.85],fontsize=7)
ax2.set_xticks(np.arange(1997,2025))
ax2.fill_between(time,number, color='grey', alpha=0.4)
fig.subplots_adjust(hspace=0)

'''df_sorted = df.sort_values(by='X-ray class')
df_sorted['group'] = pd.cut(df_sorted['Flux'], bins=8)
# 使用groupby和value_counts函数统计每组中class列中0和1的出现次数
result = df_sorted.groupby('group')['class'].value_counts()
df_result = result.reset_index()
df_result.columns = ['group', 'class', 'count']  # 重命名列
print(df_result)
x = np.arange(len(df_result['group'].unique()))  # the label locations
width = 0.25  # the width of the bars
for class_value in df_result['class'].unique():
    data = df_result[df_result['class'] == class_value]
    if class_value == 0:
        ax3.bar(x - width/2, data['count'], width, label='Confined')
    else:
        ax3.bar(x + width/2, data['count'], width, label='Eruptive')
ax3.set_ylabel('Numbers')
x_ticks = [interval.mid for interval in df_result['group'].unique()]
ax3.grid(axis='y', linestyle='--', linewidth=0.5)
float_list = [float(i) for i in x_ticks]
for i in range(len(float_list)):
    float_list[i]="{:.2e}".format(float_list[i])
ax3.set_xticks(x)
sci_categories = [f'({interval.left*10000:.2}, {interval.right*10000:.2}]' for interval in df_result['group'].unique()]
ax3.set_xticklabels(sci_categories)
ax3.legend(loc=[0.65,0.80],fontsize=8)
ax3.set_xlabel('E-4[W/m^2]')'''
plt.show()
# 显示图形




