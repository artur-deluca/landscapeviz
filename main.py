
import tensorflow as tf
import os

from functools import partial
from sklearn import datasets, preprocessing, model_selection

import landscapeviz


def build_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(10, activation=tf.nn.relu, input_shape=(4,)),  # input shape required
        tf.keras.layers.Dense(10, activation=tf.nn.relu),
        tf.keras.layers.Dense(3, activation=tf.nn.softmax)
    ])
    
    model.compile("sgd", loss="sparse_categorical_crossentropy", metrics=['sparse_categorical_accuracy', 'categorical_hinge'])
    return model


def get_data(seed):
    data = datasets.load_iris()
    X_train, X_test, y_train, y_test = model_selection.train_test_split(data["data"], data["target"], test_size=0.25, random_state=seed)
    
    scaler_x = preprocessing.MinMaxScaler(feature_range=(-1,+1)).fit(X_train)
    X_train = scaler_x.transform(X_train)
    X_test = scaler_x.transform(X_test)

    return  X_train, X_test, y_train, y_test


if __name__ == "__main__":
    
    seed = 42
    tf.random.set_seed(seed)
    X_train, X_test, y_train, y_test = get_data(seed)

    checkpoint = tf.keras.callbacks.ModelCheckpoint("./weights/sgd.{epoch:02d}.hdf5", verbose=0, save_weights_only=True, period=1)

    model = build_model()
    model.fit(X_train, y_train, batch_size=32, epochs=60, verbose=0, callbacks=[checkpoint])
    landscapeviz.build_mesh(model, (X_train, y_train), grid_lenght=40, extension=10, verbose=1, seed=seed)
    landscapeviz.plot_contour(key=None, trajectory="./weights")
    landscapeviz.plot_3d(log=False)



