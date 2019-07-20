import numpy as np


def initialize_weights_single_layer(layer, weights_initialization_values_arr):
    layer.set_weights(weights_initialization_values_arr)


def zeros_array(shape):
    return np.zeros(shape)


def random_normal(shape, mean=0, std=1):
    return np.random.normal(loc=mean, scale=std, size=shape)


def set_layer_weights_to_anti_symmetric_constant(layer, constant=1):
    layer_weights_shape = layer.get_weights()[0].shape
    constant_anti_symmetric_weight_vec = [np.concatenate((constant * np.ones(int(layer_weights_shape[0] / 2)),
                                                          -constant * np.ones(
                                                              int(layer_weights_shape[0] / 2)))).reshape(
        layer_weights_shape)]
    layer.set_weights(constant_anti_symmetric_weight_vec)


def initialize_all_weights_from_method(model, initialization_method=zeros_array):
    for layer in model.layers:
        layer_initialization = [initialization_method(layer.get_weights()[0].shape)]
        layer.set_weights(layer_initialization)
