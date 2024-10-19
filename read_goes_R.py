import netCDF4 as nc
import os
import cftime
import matplotlib.pyplot as plt

dir='/Users/duan/Desktop'
filename='sci_sgps-l2-avg1m_g16_d20230101_v3-0-0.nc'


file_path = os.path.join(dir, filename)
# 打开NC文件
ds = nc.Dataset(file_path)
# 从GOES-16的SGPS仪器获取数据





# %%
# Open netcdf file for reading data
ds = nc.Dataset(file_path)


# %%
# Time conversion
times = cftime.num2pydate(ds.variables["time"][:], ds["time"].units)
print("start and end times:", times[0], times[-1])


# %%
# Print all variable names
print("\nAll variable names: ")
print(list(ds.variables.keys()), "\n")
print(ds['DiffProtonLowerEnergy'])
print(ds['DiffProtonUpperEnergy'])
print(ds['DiffProtonLowerEnergy'][:])
print(ds['DiffProtonUpperEnergy'][:])
print(ds['AvgDiffProtonFlux'])
print(ds['IntegralProtonEffectiveEnergy'])
# %%
# Plot 1 day of MPSH 1-minute AvgDiffProtonFlux in Telescope 0, Band 0
'''var = 'AvgDiffProtonFlux'
tel = 0
band = 12
data = []
for i in range(0, len(times)):
    data.append(ds.variables[var][i][tel][band])
plt.figure(0, figsize=[10, 7])
plt.plot(
    times[:],
    data,
    linewidth=1,
    color="green",
    label=f"tel{tel}-band{band}",
)
plt.yscale("log")
plt.legend(loc="upper right", prop={"size": 12})
plt.xlabel("Time [UT]")
plt.ylabel(f"{var} [{ds[var].units}]")
plt.show()
print(ds['AvgIntProtonFlux'])
print("Done.\n")'''