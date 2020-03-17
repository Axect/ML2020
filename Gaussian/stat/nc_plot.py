from netCDF4 import Dataset
import matplotlib.pyplot as plt

# Import netCDF file
ncfile11 = './mahal1.nc'
data11 = Dataset(ncfile11)
var11 = data11.variables

ncfile12 = './mahal2.nc'
data12 = Dataset(ncfile12)
var12 = data12.variables

ncfile2 = './data.nc'
data2 = Dataset(ncfile2)
var2 = data2.variables

# Use latex
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

# Prepare Plot
plt.figure(figsize=(10,6), dpi=300)
plt.title(r"Title", fontsize=16)
plt.xlabel(r'$x$', fontsize=14)
plt.ylabel(r'$y$', fontsize=14)

# Prepare Data to Plot
x11 = var11['x'][:]
y11 = var11['y'][:]
x12 = var12['x'][:]
y12 = var12['y'][:]
x2 = var2['x'][:]
y2 = var2['y'][:]

# Plot with Legends
plt.plot(x11, y11, label=r'$d=1$')
plt.plot(x12, y12, label=r'$d=2$')
plt.scatter(x2, y2, label='data')

# Other options
plt.legend(fontsize=12)
plt.grid()
plt.savefig("plot.png", dpi=300)
