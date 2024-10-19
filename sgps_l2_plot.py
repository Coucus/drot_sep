import pyspedas.projects.goes


trange = ['2023-01-01', '2023-01-02']
vars = pyspedas.projects.goes.epead(trange=trange,time_clip=True)
print(vars)