from netCDF4 import Dataset
import matplotlib.pyplot as plt

# Import netCDF file
ncfile = './data/data.nc'
data = Dataset(ncfile)
var = data.variables
ncfile2 = './data/reg.nc'
data2 = Dataset(ncfile2)
var2 = data2.variables

# Use latex
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

# Prepare Plot
plt.figure(figsize=(10,6), dpi=300)
plt.title(r"Linear Regression", fontsize=16)
plt.xlabel(r'$x$', fontsize=14)
plt.ylabel(r'$y$', fontsize=14)

# Prepare Data to Plot
x = var['x'][:]
t = var['t'][:]  
x2 = var2['x'][:]
y2 = var2['y'][:]

# Plot with Legends
plt.scatter(x, t, label=r'Data')
plt.plot(x2, y2, label=r'Reg')

# Other options
plt.legend(fontsize=12)
plt.grid()
plt.savefig("plot.png", dpi=300)
