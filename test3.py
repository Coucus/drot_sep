import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
file_path1 = r"C:\Users\13989\Desktop\sunspot_smooth.txt"
file_path2 = r"C:\Users\13989\Desktop\sunspot.txt"
df1 = pd.read_csv(file_path1, sep='\\s+', on_bad_lines='skip')
df2= pd.read_csv(file_path2, sep='\\s+', on_bad_lines='skip')
time,time1= df1.iloc[:, 2],df2.iloc[:, 2]
number,number1=df1.iloc[:, 3],df2.iloc[:, 3]
fig=plt.figure(figsize=(20,6))
plt.plot(time,number,label='Smoothed monthly sunspot numbers')
plt.plot(time1,number1,label='monthly sunspot numbers',linestyle='--',linewidth=1)
plt.grid(axis='y', linestyle='--', linewidth=0.5)
plt.ylabel('Numbers')
plt.xlabel('Years')
plt.legend(loc=[0.89,1.03])
plt.title('Data from 2011/02/01-2023/08/23')
plt.xticks(np.arange(2011,2024))
plt.show()