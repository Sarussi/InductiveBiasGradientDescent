import numpy as np
import keras.backend as K


def initialize_weights_single_layer(layer, weights_initialization_values_arr):
    layer.set_weights(weights_initialization_values_arr)


def zeros_array(shape):
    return np.zeros(shape)


def random_normal(shape, mean=0, std=0.005):
    return np.random.normal(loc=mean, scale=std, size=shape)


def set_layer_weights_to_anti_symmetric_constant(layer, constant=1):
    layer_weights_shape = layer.get_weights()[0].shape
    constant_anti_symmetric_weight_vec = np.concatenate((constant * np.ones(int(layer_weights_shape[0] / 2)),
                                                         -constant * np.ones(
                                                             int(layer_weights_shape[0] / 2)))).reshape(
        layer_weights_shape)
    K.set_value(layer.weights[0], constant_anti_symmetric_weight_vec)
    # layer.set_weights(constant_anti_symmetric_weight_vec)


def set_weights_to_constant_vector(layer, vector):
    initialization_matrix = vector.reshape(layer.get_weights()[0].shape)
    layer.set_weights([initialization_matrix])


def set_weights_to_discontinious_prediction_function(layer):
    matrix = layer.get_weights()[0]
    bias = layer.get_weights()[1]
    W = np.zeros(shape=matrix.shape)
    W[:, 1] = np.array([1, -1])
    W[:, 2] = np.array([-1, 1])
    bias_init = np.zeros(shape=bias.shape)
    bias_init[1] = -0.5
    bias_init[2] = 0
    K.set_value(layer.weights[0], W)
    K.set_value(layer.weights[1], bias_init)
    layer.set_weights([W, bias_init])


def initialize_all_weights_from_method(model, initialization_method=zeros_array):
    for layer in model.layers:
        layer_weights_shapes = [weight_matrix.shape for weight_matrix in layer.get_weights()]
        layer_initialization = [initialization_method(weight_shape) for weight_shape in layer_weights_shapes]
        layer.set_weights(layer_initialization)
