from keras.layers import Dense
from keras.models import Sequential

import keras.backend as K


def exponent_loss(y_true, y_pred):
    return K.mean(K.exp(-y_true * y_pred))


def logistic_loss(y_true, y_pred):
    return K.mean(K.log(1 + K.exp(-y_true * y_pred)))


def prediction_accuracy(y_true, y_pred):
    return K.not_equal(K.sign(y_pred), y_true)


def create_model_skeleton(number_of_neurons_in_each_layer, activation, input_dimension):
    model = Sequential()
    model.add(Dense(number_of_neurons_in_each_layer[0], input_dim=input_dimension, bias=False, activation=activation))
    for index in range(1, len(number_of_neurons_in_each_layer)):
        model.add(Dense(number_of_neurons_in_each_layer[index], bias=False, activation=activation))
    return model


def create_model(number_of_neurons_in_each_layer, activation, input_dimension, optimizer, loss_function):
    model = create_model_skeleton(number_of_neurons_in_each_layer, activation, input_dimension)
    model.compile(optimizer=optimizer,
                  loss=loss_function, metrics=[prediction_accuracy])
    return model
