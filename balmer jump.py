
import numpy as np
from scipy.fft import fft
import pandas as pd
import matplotlib.pyplot as plt

file_path = r"C:\Users\13989\Desktop\c5.0.xlsx"
df = pd.read_excel(file_path)
df_sorted = df.sort_values(by=' PEAKFLUX')
df_sorted['group'] = pd.cut(df_sorted[' PEAKFLUX'], bins=100)

result = df_sorted.groupby('group')['class'].value_counts()
df_result = result.reset_index()
df_result.columns = ['group', 'class', 'count']

x = np.arange(len(df_result['group'].unique()))
width = 0.25
fig = plt.figure(figsize=(16, 10))
ax1 = fig.add_subplot(1, 1, 1)

for class_value in df_result['class'].unique():
    data = df_result[df_result['class'] == class_value]
    x_log = np.log10([interval.mid for interval in data['group'].unique()])
    ax1.plot(x_log, data['count'], label='Confined' if class_value == 0 else 'Eruptive', marker='o', markersize=4)

ax1.set_ylabel('Numbers')
ax1.grid(axis='y', linestyle='--', linewidth=0.5)
ax1.set_xticks(np.log10(x))
ax1.set_xticklabels(["{:.2e}".format(interval.mid) for interval in data['group'].unique()])
ax1.legend(loc=[0.8, 0.90], fontsize=8)
ax1.tick_params(labelsize=8)
ax1.set_xlabel('E-3[W/m^2]')
plt.show()
