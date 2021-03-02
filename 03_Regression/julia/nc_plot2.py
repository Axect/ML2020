from netCDF4 import Dataset
import matplotlib.pyplot as plt

# Import netCDF file
ncfile1 = './data/gauss.nc'
data1 = Dataset(ncfile1)
var1 = data1.variables

#ncfile2 = './reg.nc'
ncfile2 = './data/gauss_single_reg.nc'
data2 = Dataset(ncfile2)
var2 = data2.variables

# Use latex
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

# Prepare Plot
plt.figure(figsize=(10,6), dpi=300)
plt.title(r"Gaussian linear regression", fontsize=16)
plt.xlabel(r'$x$', fontsize=14)
plt.ylabel(r'$y$', fontsize=14)

# Prepare Data to Plot
x1 = var1['x'][:]
y1 = var1['y'][:]

x2 = var2['x'][:]
y2 = var2['y'][:]

# Plot with Legends
plt.scatter(x1, y1, label='data', alpha=0.7)
plt.plot(x2, y2, label=r'$\lambda = 0.5$', color='r', alpha=0.7)

# Other options
plt.legend(fontsize=12)
plt.grid()
plt.savefig("plot_lam_01_5.png", dpi=300)
