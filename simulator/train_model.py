import keras
import keras.backend as K
import numpy as np


class WeightsHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.weights = []
        self.losses = []
        self.weights.append(self.model.weights)

    def on_batch_end(self, epoch, logs={}):
        modelWeights = dict()
        for layer_index in range(0, len(self.model.layers)):
            current_layer = self.model.layers[layer_index]
            modelWeights['layer_{layer_index}'.format(layer_index=layer_index)] = current_layer.get_weights()[0]
        self.weights.append(modelWeights)
        self.losses.append(logs.get('loss'))


def train_model(model, x_train, y_train, x_test, y_test, number_of_epochs):
    weights_history = WeightsHistory()
    weights_history.set_model(model)
    callbacks = [weights_history]
    error_history = model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=number_of_epochs, batch_size=1,
                              callbacks=callbacks)
    weights_temp = weights_history.weights
    aa = 1
    weights_temp.remove(weights_temp[0])
    loss_diff = np.diff(np.array(weights_history.losses))
    non_zero_updates = np.array(np.where(np.abs(loss_diff) > 0.000000001))
    aa=1
    weights_layers_epochs = {}
    for layer_index in range(len(model.layers)):
        key = 'layer_{layer_index}'.format(layer_index=layer_index)
        weights_layers_epochs[key] = []
    for index in range(0, len(weights_temp)):
        current_weights = weights_temp[index]
        for key in current_weights.keys():
            weights_layers_epochs[key].append(current_weights[key])
    aa=1
    for key in weights_layers_epochs.keys():
        weights_layers_epochs[key] = list(np.array(weights_layers_epochs[key])[list(non_zero_updates+1)])
    aa=1
    return weights_layers_epochs, error_history.history
