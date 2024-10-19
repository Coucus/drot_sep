import netCDF4 as nc
import os
import cftime

import matplotlib.pyplot as plt

dir='/Users/duan/Desktop/Pythonwork/sateliite/goes_data/20130929'
filename='g15_hepad_s15_1m_20130901_20130930.nc'


file_path = os.path.join(dir, filename)
# 打开NC文件
ds = nc.Dataset(file_path)
# 从GOES-16的SGPS仪器获取数据





# %%
# Open netcdf file for reading data
ds = nc.Dataset(file_path)




# %%
# Print all variable names
print("\nAll variable names: ")
print(list(ds.variables.keys()), "\n")
print(ds['S1_NUM_PTS'])

