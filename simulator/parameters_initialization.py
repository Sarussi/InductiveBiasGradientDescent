import numpy as np


def initialize_weights_single_layer(layer, weights_initialization_values_arr):
    layer.set_weights(weights_initialization_values_arr)


def zeros_array(shape):
    return np.zeros(shape)


def random_normal(shape, mean=0, std=1):
    return np.random.normal(loc=mean, scale=std, size=shape)


def initialize_all_weights(model, initialization_method=zeros_array):
    for layer in model.layers:
        layer_initialization = [initialization_method(layer.get_weights()[0].shape)]
        layer.set_weights(layer_initialization)
