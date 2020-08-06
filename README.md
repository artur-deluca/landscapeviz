# Visualizing the Loss Landscape of Neural Networks
> <b>Application showcasing landscapeviz [here](https://landscapeviz.herokuapp.com/)</b>

<p align="center">
  <img width="350" height="350" src="/docs/img/countour.svg">
  <img width="450" height="350" src="/docs/img/surface_hinge.svg">
</p>

This repository is an implementation of the paper

> Hao Li, Zheng Xu, Gavin Taylor, Christoph Studer and Tom Goldstein. [*Visualizing the Loss Landscape of Neural Nets*](https://arxiv.org/abs/1712.09913). NIPS, 2018.

This code was implemented in tensorflow 2.0. The authors also have an [implementation](https://github.com/tomgoldstein/loss-landscape) using pytorch.

## How to use
```python
# 1. define model

model = tf.keras.Sequential([
	tf.keras.layers.Dense(10, activation=tf.nn.relu, input_shape=(4,)),  # input shape required
	tf.keras.layers.Dense(10, activation=tf.nn.relu),
	tf.keras.layers.Dense(3, activation=tf.nn.softmax)
])

model.compile("sgd", loss="sparse_categorical_crossentropy", metrics=['sparse_categorical_accuracy', 'categorical_hinge'])

# 2. get data
data = sklearn.datasets.load_iris()
X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(data["data"], data["target"], test_size=0.25, random_state=seed)

scaler_x = sklearn.preprocessing.MinMaxScaler(feature_range=(-1,+1)).fit(X_train)
X_train = scaler_x.transform(X_train)
X_test = scaler_x.transform(X_test)

# 3. train model
model.fit(X_train, y_train, batch_size=32, epochs=60, verbose=0)


# 4. build mesh and plot
landscapeviz.build_mesh(model, (X_train, y_train), grid_length=40, verbose=0)
landscapeviz.plot_contour(key="sparse_categorical_crossentropy")
landscapeviz.plot_3d(key="sparse_categorical_crossentropy")
```


