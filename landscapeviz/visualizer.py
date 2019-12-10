import h5py
import os
import matplotlib.pyplot as plt
import numpy as np

from mpl_toolkits.mplot3d import Axes3D


def plot_contour(key=None, vmin=0.1, vmax=10, vlevel=0.5, trajectory=None):
    
    with h5py.File("meshfile.hdf5", "r") as f:
        X, Y = np.array(f["X"]), np.array(f["Y"])
        Z = np.array(f[get_key(f.keys(), key)]) 

    fig, ax = plt.subplots()
    CS = ax.contour(X, Y, Z,  cmap='summer', levels=np.arange(vmin, vmax, vlevel))
    ax.clabel(CS, inline=1, fontsize=8)
    if trajectory:
        with h5py.File(os.path.join(trajectory, ".trajectory", "model_weights.hdf5"), "r") as f:
            ax.plot(np.array(f["X"]), np.array(f["Y"]), marker='.')
    fig.savefig("./countour.svg")
    plt.show()

def plot_grid(key=None):

    f = h5py.File("meshfile.hdf5", "r")
    X, Y = f["X"], f["Y"]
    Z = np.array(f[get_key(f.keys(), key)]) 

    fig, ax = plt.subplots()

    cmap = plt.cm.coolwarm
    cmap.set_bad(color='black')
    plt.imshow(Z, interpolation='none', cmap=cmap, extent=[X.min(), X.max(), Y.min(), Y.max()])
    fig.savefig("./grid.svg")
    plt.show()

def plot_3d(log=False, key=None):
    
    f = h5py.File("meshfile.hdf5", "r")
    X = np.array(f["X"])
    Y = np.array(f["Y"])
    Z = np.array(f[get_key(f.keys(), key)]).copy()

    if log:
        Z = np.log(Z + 0.1)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot the surface.
    surf = ax.plot_surface(X, Y, Z, cmap=plt.cm.coolwarm, linewidth=0, antialiased=False)
    fig.colorbar(surf, shrink=0.5, aspect=5)
    fig.savefig("./surface.svg")
    plt.show()

def get_key(fields, key):
    
    if key is not None:
        key = [s for s in fields if key in s][0]
    else:
        key = [s for s in fields if "Z_" in s][0]
    return key



  