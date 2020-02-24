from keras.layers import Dense
from keras.models import Sequential
from keras import losses
import keras.backend as K
import keras.backend as K
import time
from keras.activations import linear
import numpy as np


def exponent_loss(y_true, y_pred):
    return K.mean(K.exp(-y_true * y_pred))


def logistic_loss(y_true, y_pred):
    return K.mean(K.log(1 + K.exp(-y_true * y_pred)))


def cross_entropy(y_true, y_pred):
    return K.mean(K.binary_crossentropy(y_true, y_pred, from_logits=True), axis=-1)


def mse_loss(y_true, y_pred):
    return losses.mean_squared_error(y_true, y_pred)


def prediction_accuracy(y_true, y_pred):
    return K.not_equal(K.sign(y_pred), y_true)


def regression_point_wise_loss(training_predictions_array, training_labels_array):
    return (training_predictions_array - training_labels_array) ** 2


def hinge_point_wise_loss(training_predictions_array, training_labels_array):
    return np.maximum(1 - training_labels_array * training_predictions_array, 0)


def sigmoid_point_wise(training_predictions_array):
    return 1/(1 + np.exp(-training_predictions_array))


def cross_entropy_point_wise_loss(training_predictions_array, training_labels_array):
    return -(training_labels_array * np.log(sigmoid_point_wise(training_predictions_array)) + (
        1 - training_labels_array) * np.log(1 - sigmoid_point_wise(training_predictions_array)))


def create_model_skeleton(network_architecture_by_layers_dict, activation, input_dimension):
    model = Sequential()
    try:
        activation_name_str = activation.__name__
    except:
        activation_name_str = activation.__class__.__name__
    layers_keys = list(network_architecture_by_layers_dict.keys())
    model.add(Dense(network_architecture_by_layers_dict[layers_keys[0]]['number_of_neruons'], input_dim=input_dimension,
                    bias=network_architecture_by_layers_dict[layers_keys[0]]['use_bias'], activation=activation,
                    name='{activation_name}_0_trainable_{trainable_status}'.format(activation_name=activation_name_str,
                                                                                   trainable_status=str(
                                                                                       network_architecture_by_layers_dict[
                                                                                           layers_keys[0]][
                                                                                           'is_trainable'])),
                    trainable=network_architecture_by_layers_dict[layers_keys[0]][
                        'is_trainable']))
    for layer_index in range(1, len(layers_keys) - 1):
        model.add(
            Dense(network_architecture_by_layers_dict[layers_keys[layer_index]]['number_of_neruons'],
                  bias=network_architecture_by_layers_dict[layers_keys[0]]['use_bias'],
                  activation=activation,
                  name='{activation_name}_{index}_trainable_{trainable_status}'.format(
                      activation_name=activation_name_str,
                      index=layer_index, trainable_status=
                      str(network_architecture_by_layers_dict[
                              layers_keys[layer_index]][
                              'is_trainable'])),
                  trainable=
                  network_architecture_by_layers_dict[layers_keys[layer_index]][
                      'is_trainable']))
    if len(layers_keys) > 1:
        model.add(
            Dense(network_architecture_by_layers_dict[layers_keys[-1]]['number_of_neruons'],
                  bias=network_architecture_by_layers_dict[layers_keys[0]]['use_bias'],
                  activation=linear, trainable=network_architecture_by_layers_dict[
                    layers_keys[-1]][
                    'is_trainable'],
                  name='linear_trainable_{trainable_status}'.format(trainable_status=
                                                                    str(network_architecture_by_layers_dict[
                                                                            layers_keys[-1]][
                                                                            'is_trainable']))))
    return model


def create_model(number_of_neurons_in_each_layer, activation, input_dimension, optimizer, loss_function,
                 learning_rate=0.01, evaluation_metrics=prediction_accuracy):
    model = create_model_skeleton(number_of_neurons_in_each_layer, activation, input_dimension)
    K.set_value(optimizer.lr, learning_rate)
    if evaluation_metrics:
        model.compile(optimizer=optimizer,
                      loss=loss_function, metrics=[evaluation_metrics])
    else:
        model.compile(optimizer=optimizer,
                      loss=loss_function)
    return model
