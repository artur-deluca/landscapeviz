import h5py
import numpy as np
import os
import tensorflow as tf


def load_weights(model, folder_path):

    sgd_weights = list()

    file_path = os.path.join(folder_path, ".trajectory")
    if not os.path.exists(file_path):
        os.makedirs(file_path)
    file_path = os.path.join(file_path, "model_weights.hdf5")

    for weights in sorted(os.listdir(folder_path)):
        if weights.endswith(".hdf5"):
            model.load_weights(os.path.join(folder_path, weights))
            solution = weight_decoder(model)
            sgd_weights.append(solution)
    with h5py.File(file_path, "w") as f:
        f["weights"] = np.array(sgd_weights)


def weight_decoder(model):
    solution = np.array([])
    weights = model.get_weights()
    for i in range(len(weights)):
        solution = np.append(solution, weights[i].flatten())
    return solution


def weight_encoder(model, solution):
    start = 0
    weights = model.get_weights()
    for i in range(len(weights)):
        weight_shape = weights[i].shape
        finish = np.prod(weight_shape)
        weights[i] = np.reshape(solution[start:start + finish], weight_shape)
        start += finish
    return weights
