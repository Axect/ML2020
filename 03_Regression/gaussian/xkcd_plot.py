from netCDF4 import Dataset
import matplotlib.pyplot as plt

# Import netCDF file
ncfile = './data/data.nc'
data = Dataset(ncfile)
var = data.variables

ncfile2 = './data/single/reg.nc'
data2 = Dataset(ncfile2)
var2 = data2.variables

# Use latex
#plt.rc('text', usetex=True)
#plt.rc('font', family='serif')

with plt.xkcd():
    # Prepare Plot
    plt.figure(figsize=(10,6), dpi=300)
    plt.title(r"Non Linear Regression?", fontsize=16)
    plt.xlabel(r'x', fontsize=14)
    plt.ylabel(r'y', fontsize=14)
    
    # Prepare Data to Plot
    x1 = var['x'][:]
    y1 = var['y'][:]  

    x2 = var2['x'][:]
    y2 = var2['y'][:]  
    
    # Plot with Legends
    plt.scatter(x1, y1, label='data')
    plt.plot(x2, y2, label=r'fit', color='r')
    
    # Other options
    plt.legend(fontsize=12)

plt.savefig("plot/xkcd_lam_10_5_plot.png", dpi=300)
