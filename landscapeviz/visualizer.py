import h5py
import os
import matplotlib.pyplot as plt
import numpy as np

from mpl_toolkits.mplot3d import Axes3D

FILENAME = "./files/meshfile.hdf5"


def _fetch_data(key, filename):

    if filename[-5:] != ".hdf5":
        filename += ".hdf5"

    with h5py.File(filename, "r") as f:
        space = np.asarray(f["space"])
        Z = np.array(f[key])

    X, Y = np.meshgrid(space, space)
    return X, Y, Z


def plot_contour(key, vmin=0.1, vmax=10, vlevel=0.5, trajectory=None, filename=FILENAME, save=False):

    X, Y, Z = _fetch_data(key, filename)

    fig, ax = plt.subplots()
    CS = ax.contour(X, Y, Z,  cmap='summer',
                    levels=np.arange(vmin, vmax, vlevel))
    ax.clabel(CS, inline=1, fontsize=8)

    if trajectory:
        with h5py.File(os.path.join(trajectory, ".trajectory", "model_weights.hdf5"), "r") as f:
            ax.plot(np.array(f["X"]), np.array(f["Y"]), marker='.')
    if save:
        fig.savefig("./countour.svg")

    plt.show()


def plot_grid(key, filename=FILENAME, save=False):

    X, Y, Z = _fetch_data(key, filename)
    fig, _ = plt.subplots()

    cmap = plt.cm.coolwarm
    cmap.set_bad(color='black')
    plt.imshow(Z, interpolation='none', cmap=cmap, extent=[
               X.min(), X.max(), Y.min(), Y.max()])
    if save:
        fig.savefig("./grid.svg")

    plt.show()


def plot_3d(key, filename=FILENAME, log=False, save=False):

    X, Y, Z = _fetch_data(key, filename)

    if log:
        Z = np.log(Z + 0.1)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot the surface.
    surf = ax.plot_surface(X, Y, Z, cmap=plt.cm.coolwarm,
                           linewidth=0, antialiased=False)
    fig.colorbar(surf, shrink=0.5, aspect=5)

    if save:
        fig.savefig("./surface.svg")

    plt.show()

