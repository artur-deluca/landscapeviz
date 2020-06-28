import h5py
import logging
import numpy as np
import os
import gc
import resource
import tensorflow as tf

from tempfile import mkdtemp
from sklearn.decomposition import PCA

from .trajectory import load_weights, weight_encoder


def get_vectors(model, seed=None, trajectory=None):

    np.random.seed(seed)
    vector_x, vector_y = list(), list()
    weights = model.get_weights()

    if trajectory:
        # this has to be re-written
        load_weights(model, trajectory)
        file_path = os.path.join(
            trajectory, ".trajectory", "model_weights.hdf5")

        with h5py.File(file_path, "r+") as f:
            differences = list()
            trajectory = np.array(f["weights"])
            for i in range(0, len(trajectory)-1):
                differences.append(trajectory[i]-trajectory[-1])

            pca = PCA(n_components=2)
            pca.fit(np.array(differences))
            f["X"], f["Y"] = pca.transform(np.array(differences)).T

        vector_x = weight_encoder(model, pca.components_[0])
        vector_y = weight_encoder(model, pca.components_[1])

        return weights, vector_x, vector_y

    else:
        cast = np.array([1]).T
        for layer in weights:
            # set standard normal parameters
            # filter-wise normalization
            k = len(layer.shape) - 1
            d = np.random.multivariate_normal(
                [0], np.eye(1), layer.shape).reshape(layer.shape)
            dist_x = (
                d/(1e-10 + cast*np.linalg.norm(d, axis=k))[:, np.newaxis]).reshape(d.shape)

            vector_x.append(
                (dist_x * (cast*np.linalg.norm(layer, axis=k))
                 [:, np.newaxis]).reshape(d.shape)
            )

            d = np.random.multivariate_normal(
                [0], np.eye(1), layer.shape).reshape(layer.shape)
            dist_y = (
                d/(1e-10 + cast*np.linalg.norm(d, axis=k))[:, np.newaxis]).reshape(d.shape)

            vector_y.append(
                (dist_y * (cast*np.linalg.norm(layer, axis=k))
                 [:, np.newaxis]).reshape(d.shape)
            )

        return weights, vector_x, vector_y


def _obj_fn(model, data, solution):

    old_weights = model.get_weights()
    model.set_weights(solution)
    value = model.evaluate(data[0], data[1], verbose=0)
    model.set_weights(old_weights)

    return value


def build_mesh(model, data, grid_length, extension=1, filename="meshfile", verbose=True, seed=None, trajectory=None):

    logging.basicConfig(level=logging.INFO)

    z_keys = model.metrics_names
    z_keys[0] = model.loss
    Z = list()

    # get vectors and set spacing
    origin, vector_x, vector_y = get_vectors(
        model, seed=seed, trajectory=trajectory)
    space = np.linspace(-extension, extension, grid_length)

    X, Y = np.meshgrid(space, space)

    for i in range(grid_length):
        if verbose:
            logging.info("line {} out of {}".format(i, grid_length))

        for j in range(grid_length):
            solution = [
                origin[x] + X[i][j] * vector_x[x] +
                Y[i][j] * vector_y[x]
                for x in range(len(origin))
            ]

            Z.append(_obj_fn(model, data, solution))

    Z = np.array(Z)
    os.makedirs('./files', exist_ok=True)

    with h5py.File("./files/{}.hdf5".format(filename), "w") as f:

        f["space"] = space
        original_results = _obj_fn(model, data, origin)

        for i, metric in enumerate(z_keys):
            f["original_" + metric] = original_results[i]
            f[metric] = Z[:, i].reshape(X.shape)
        f.close()

    del Z
    gc.collect()
