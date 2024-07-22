'''import numpy as np
import matplotlib.pyplot as plt
# 常数定义
transition_start_wavelength1=91.2e-9
transition_start_wavelength2=364.7e-9
absorption_constant1=0.87
absorption_constant2=0.1
h = 6.626e-34  # 普朗克常数
c = 3e8  # 光速
k = 1.38e-23  # 玻尔兹曼常数
def piecewise_radiation_modified(wavelength):
    if wavelength < transition_start_wavelength1:
        temperature = 6800
        intensity =(8 * np.pi * h * c) / (wavelength**5 * (np.exp((h * c) / (wavelength * k * temperature)) - 1))
        return intensity
    elif wavelength >= transition_start_wavelength1 and wavelength <transition_start_wavelength2:
        temperature = 6800
        intensity = (8 * np.pi * h * c) / (wavelength**5 * (np.exp((h * c) / (wavelength * k * temperature)) - 1))
        return intensity-absorption_constant1*intensity
    elif wavelength >= transition_start_wavelength2:
        temperature = 6800
        intensity = (8 * np.pi * h * c) / (wavelength**5 * (np.exp((h * c) / (wavelength * k * temperature)) - 1))
        return intensity-absorption_constant1*intensity-absorption_constant2*intensity
# 温度（K）


# 波长范围（m）
wavelength = np.linspace(1e-10, 3e-6, 2000)
# 计算辐射强度
intensity = np.array([piecewise_radiation_modified(w) for w in wavelength])
# 绘制图形
plt.plot(wavelength * 1e9,intensity)
plt.xlabel('wavelength (nm)')
plt.ylabel('intensity')
plt.title('6800K')
plt.grid(True)
plt.show()'''
import torch
print(torch.__version__)
print(torch.cuda.is_available())