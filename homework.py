import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
'''random=np.random.randn(5)
steps=5
walk=[]
walk=np.where(random>0,1,-1)
postition=np.cumsum(walk)
print(postition[:5])
plt.plot(postition[:5])
plt.show()
data=pd.DataFrame({'a':[1,2,3],'b':[2,3,4]})
result=data.apply(pd.value_counts).fillna(0)
print(result)'''


'''def binarySearch(arr,test):
    if test>arr[-1] or test<arr[0]:
        print('invalid')
        return -1
    else:
     i=int(len(arr)/2)
     if test==arr[i]:
        print('the number is valid')
        return arr[i]    
     elif test>arr[i]:
        arr=arr[i+1:]    
        return binarySearch(arr,test)
     elif test<arr[i]:
        arr=arr[0:i-1]
        return binarySearch(arr,test)
     else:
        print('number is invalid')
        return -1

test1=[1,2,3,4,5]
num=2
binarySearch(test1,num)'''

'''def insertionSort(arr): 
  for a  in range(1,len(arr)):
    for b in range(0,a):
        if arr[a]<arr[b]:
            arr.insert(b,arr[a])
            del arr[a+1]
  print(arr)
insertionSort([2,6,1,8,2,5,3])'''

'''def quickSort(arr):
    if(len(arr)<2): #不用进行排序
        return arr
    else:
        pivot=arr[0]
        less=[i for i in arr[1:] if(i<=pivot)]
        great=[i for i in arr[1:] if(i>pivot)]
        return quickSort(less)+[pivot]+quickSort(great)
arr=[1,4,5,2,41,4,24,5,78,5,67,89,65]
print("原始数据：",arr)
print("排序后的数据：",quickSort(arr))'''
'''new_arr=[]
def select(arr):
      if len(arr)>1:
       for i in range(len(arr)-1):
         if arr[i]>arr[i+1]:
            arr[i],arr[i+1]=arr[i+1],arr[i]
       new_arr.append(arr[len(arr)-1])
       del arr[len(arr)-1]
       select(arr)
      else: new_arr.append(arr[len(arr)-1]),print(new_arr)
      return -1
'''
'''data=pd.DataFrame(np.arange(12).reshape(3,4))
transform= lambda x:x[:4].upper()
print(type(data))'''
'''#定义成本函数
def computeCost(X, Y, theta):
    inner = np.power((X * theta.T) - Y, 2)
    #用矩阵很方便录入不同的参数，行数为个数,列为变量数
    return np.sum(inner) / (2 * len(X))
#len取行数
    
#定义梯度下降
def gradientDescent(X, Y, theta, alpha, iters):#X,Y为列矩阵，theta为行矩阵
    temp = np.matrix(np.zeros(theta.shape))
    parameters = int(theta.shape[1])#[1]取列的个数
    cost = np.zeros(iters)#iters为执行次数,cost是数组
    for i in range(iters):
        error = (X * theta.T) - Y
        for j in range(parameters):
            term = np.multiply(error,X[:, j])#取某列 ,multply做直积
            temp[0, j] = theta[0, j] - alpha / len(X) * np.sum(term)
        theta = temp
        cost[i] = computeCost(X, Y, theta)
    return theta, cost
path =r"C:\Users\13989\Desktop\ex1data2.txt"
data = pd.read_csv(path, header=None,names=['Price', 'Bedrooms', 'Years'])
data.head()
means = data.mean().values
stds = data.std().values
mins = data.min().values
maxs = data.max().values
data_ = data.values
data.describe()
data = (data - data.mean()) / data.std()
data.insert(3,'C',1)#c常数组
X_test=np.matrix(data[['Bedrooms','Years','C']])
Y_test=np.matrix(data['Price'])
theta_test=np.matrix([2,2,1])
iters=100
alpha=[0.01,0.03]
g,cost=gradientDescent(X_test,Y_test,theta_test,alpha[0],iters)
g1,cost1=gradientDescent(X_test,Y_test,theta_test,alpha[1],iters)
fig=plt.figure(figsize=(15,8))
ax1=fig.add_subplot(1,2,1)
ax2=fig.add_subplot(1,2,2)
ax1.plot(np.arange(iters),cost, 'r')
ax1.set_xlabel('Iterations')
ax1.set_ylabel('Cost')
ax1.set_title('learning curve')
ax2.plot(np.arange(iters),cost1,'b')
plt.show()
from scipy.fft import fft

# 定义信号
Fs = 10e3  # 采样频率
f1 = 390  # 信号频率1
f2 = 2e3  # 信号频率2
t = np.linspace(0, 1, Fs)  # 生成 1s 的时间序列
y = 2 * np.sin(2 * np.pi * f1 * t) + 5 * np.sin(2 * np.pi * f2 * t)  # 信号

# 傅里叶变换
fft_y = fft(y)

# 计算频率
N = len(y)
freq = np.fft.fftfreq(N, 1/Fs)

# 绘制频谱
import matplotlib.pyplot as plt
plt.plot(freq[:N//2], np.abs(fft_y)[:N//2]*2/N)  # 只绘制正频率部分
plt.xlabel('Frequency [Hz]')
plt.ylabel('Amplitude')
plt.show()
'''
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


    
    



            
        
