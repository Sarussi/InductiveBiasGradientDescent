import keras
import keras.backend as K
import numpy as np
from simulator import create_model


class WeightsHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.weights = []
        self.losses = []
        self.weights.append(self.model.weights)

    def on_batch_end(self, epoch, logs={}):
        modelWeights = dict()
        modelErrors = dict()
        for layer_index in range(0, len(self.model.layers)):
            current_layer = self.model.layers[layer_index]
            modelWeights['layer_{layer_index}'.format(layer_index=layer_index)] = current_layer.get_weights()[0]
        self.weights.append(modelWeights)
        # modelErrors['test_mse'] = K.eval(create_model.mse_loss(self.model.predict(self.validation_data[0]),
        #                                                 self.validation_data[1]))
        # modelErrors['test_accuracy'] = K.eval(create_model.prediction_accuracy(self.model.predict(self.validation_data[0]),
        #                                                                 self.validation_data[1]))
        self.losses.append(logs.get('loss'))


def train_model(model, x_train, y_train, x_test, y_test, number_of_epochs):
    weights_history = WeightsHistory()
    weights_history.set_model(model)
    callbacks = [weights_history]
    errors_at_epochs = model.fit(np.array(x_train), np.array(y_train),
                                 validation_data=(np.array(x_test), np.array(y_test)),
                                 epochs=number_of_epochs, batch_size=1,
                                 callbacks=callbacks)
    weights_temp = weights_history.weights
    train_error_at_batch = weights_history.losses
    errors_between_batch_and_epochs_dict = {}
    errors_between_batch_and_epochs_dict['batches'] = {'train_loss': train_error_at_batch}
    errors_between_batch_and_epochs_dict['epochs'] = errors_at_epochs.history
    weights_temp.remove(weights_temp[0])
    loss_diff = np.diff(np.array(weights_history.losses))
    non_zero_updates = np.array(np.where(np.abs(loss_diff) > 0.000000001))
    weights_layers_between_batches = {}
    for layer_index in range(len(model.layers)):
        key = 'layer_{layer_index}'.format(layer_index=layer_index)
        weights_layers_between_batches[key] = []
    for index in range(0, len(weights_temp)):
        current_weights = weights_temp[index]
        for key in current_weights.keys():
            weights_layers_between_batches[key].append(current_weights[key])
    for key in weights_layers_between_batches.keys():
        weights_layers_between_batches[key] = list(
            np.array(weights_layers_between_batches[key])[list(non_zero_updates + 1)])
    return weights_layers_between_batches, errors_between_batch_and_epochs_dict
