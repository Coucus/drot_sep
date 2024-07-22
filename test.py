'''from bunch import *
book = Bunch()
book.data = [[16],[23],[10],[16],[11]]
book.target =[0,1,0,0,1]
book.target_name = ['肉类零食','乳制品']
book.data_name = ['能量']
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
knn = KNeighborsClassifier(n_neighbors = 1)
X_train ,X_test , y_train ,y_test = train_test_split(book['data'],book['target'],random_state=0)
knn.fit(X_train,y_train)
print('测试得分:{:.2f}\n'.format(knn.score(X_test,y_test)))
'''
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

file_path = r"C:\Users\13989\Desktop\XFlare Catalogue edit.xls"
df = pd.read_excel(file_path)

# 创建一个新的列，表示每个日期所在的季度
df['quarter'] = df['start'].dt.to_period('Q')

# 获取所有的季度
quarters = pd.Series(df['quarter'].unique()).sort_values()

flare_matrix = np.zeros((len(quarters), 5), dtype=float)

for i, quarter in enumerate(quarters):
    filtered_df = df[df['quarter'] == quarter]
    class_counts = filtered_df['class'].value_counts()
    flare_matrix[i, 0] = quarter.year + quarter.quarter / 10.0  # 使用年和季度的组合作为x轴的值
    flare_matrix[i, 1] = class_counts.get(1, 0)
    flare_matrix[i, 2] = class_counts.get(0, 0)
    flare_matrix[i,4]=flare_matrix[i, 2]+flare_matrix[i, 1]
    if flare_matrix[i, 1] == 0:
        flare_matrix[i, 3] = 0 
    elif flare_matrix[i, 2] == 0:
        flare_matrix[i, 3] = 1
    else:
        flare_matrix[i, 3] = flare_matrix[i, 1] / (flare_matrix[i, 1] + flare_matrix[i, 2])
file_path1 = r"C:\Users\13989\Desktop\sunspot_smooth.txt"
file_path2 = r"C:\Users\13989\Desktop\sunspot.txt"
df1 = pd.read_csv(file_path1, sep='\\s+', on_bad_lines='skip')
df2= pd.read_csv(file_path2, sep='\\s+', on_bad_lines='skip')
time,time1= df1.iloc[:, 2],df2.iloc[:, 2]
number,number1=df1.iloc[:, 3],df2.iloc[:, 3]
fig = plt.figure(figsize=(8, 6))
ax1 = fig.add_subplot(2, 1, 1)
ax2 = fig.add_subplot(2, 1, 2)
ax1.plot(time,number,label='Smoothed monthly sunspot numbers')
ax1.plot(time1,number1,label='Monthly sunspot numbers',linestyle='--',linewidth=1)
ax1.fill_between(time,number, color='grey', alpha=0.4)
ax2.plot(flare_matrix[:, 0], flare_matrix[:, 1], label='Eruptive numbers', linestyle='--', marker='o',linewidth=1,markersize=3)
ax2.plot(flare_matrix[:, 0], flare_matrix[:, 2], label='Confined numbers', linestyle=':', marker='^',c='green',linewidth=1,markersize=3)
ax1.grid(axis='y', linestyle='--', linewidth=0.5)
ax2.grid(axis='y', linestyle='--', linewidth=0.5)
ax1.set_xticks([])
ax2.set_xticks(np.linspace(2011,2023,7))
ax1.set_ylabel('Numbers')
ax1.set_xlabel('Years')
ax2.set_ylabel('Numbers')
ax2.set_xlabel('Years')
ax1.legend(loc=[0.6,0.85],fontsize=7)
ax1.set_title('Data from 2011/02/15-2023/08/07')
ax2.legend(loc=[0.6,0.85],fontsize=7)
fig.subplots_adjust(hspace=0)
plt.show()





