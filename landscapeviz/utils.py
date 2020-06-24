import h5py
import logging
import numpy as np
import os
import tensorflow as tf

from memory_profiler import profile
from functools import reduce, partial
from tempfile import mkdtemp
from sklearn.decomposition import PCA

from .trajectory import load_weights, weight_encoder

def get_vectors(model, seed=None, trajectory=None):

    np.random.seed(seed)
    vector_x, vector_y = list(), list()

    if trajectory:
        load_weights(model, trajectory)
        file_path = os.path.join(trajectory, ".trajectory", "model_weights.hdf5")
        with h5py.File(file_path, "r+") as f:
            differences = list()
            trajectory = np.array(f["weights"])
            for i in range(1, len(trajectory)):
                differences.append(trajectory[i]-trajectory[i-1])
            pca = PCA(n_components=2)
            pca.fit(np.array(differences))
            #print(differences)
            f["X"], f["Y"] = pca.transform(np.array(trajectory)).T
            #print(pca.transform(np.array(differences)).T)
        vector_x = weight_encoder(model, pca.components_[0])
        vector_y = weight_encoder(model, pca.components_[1])
        return model.get_weights(), vector_x, vector_y
    
    else:
        for layer in model.get_weights():
            # set standard normal parameters
            dist_x = np.random.multivariate_normal([1], np.eye(1), layer.shape).reshape(layer.shape)
            dist_y = np.random.multivariate_normal([1], np.eye(1), layer.shape).reshape(layer.shape)
            vector_x.append(dist_x*layer / np.linalg.norm(dist_x))
            vector_y.append(dist_y*layer / np.linalg.norm(dist_y))
    
    return model.get_weights(), vector_x, vector_y

def build_mesh(model, data, grid_lenght, extension=1, verbose=True, seed=None, trajectory=None):

    logging.basicConfig(level=logging.INFO)

    def obj_fn(model, data, solution):

        old_weights = model.get_weights()
        model.set_weights(solution)
        value = model.evaluate(data[0], data[1], verbose=0)
        model.set_weights(old_weights)

        return value
     
    with h5py.File("meshfile.hdf5", "w") as f:

        z_keys = ["Z_" + model.loss]
        z_keys += [metric.name for metric in model.metrics]
        z_dict = {}

        for metric in z_keys:
            filename = os.path.join(mkdtemp(), metric + '.dat')
            z_dict[metric] = np.memmap(filename, dtype='float32', mode='w+', shape=(grid_lenght, grid_lenght))

        # get vectors and set spacing
        origin, vector_x, vector_y  = get_vectors(model, seed=seed, trajectory=trajectory)
        spacing = np.linspace(-extension, extension, grid_lenght)

        f["X"], f["Y"] = np.meshgrid(spacing, spacing)

        for i in range(grid_lenght):
            if verbose:
                logging.info("line {} out of {}".format(i, grid_lenght))

            for j in range(grid_lenght):
                solution = [
                    origin[x] + f["X"][i][j] * vector_x[x] + f["Y"][i][j] * vector_y[x]
                    for x in range(len(origin))
                ]
                obj_value = obj_fn(model, data, solution)

                for index, metric in enumerate(z_keys):
                    z_dict[metric][i][j] = obj_value[index]

        for metric in z_keys:
            f[metric] = z_dict[metric]
