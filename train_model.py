import keras
import keras.backend as K
import numpy as np
import create_model


class WeightsHistory(keras.callbacks.Callback):
    def __init__(self, history_flags):
        keras.callbacks.Callback.__init__(self)
        self.history_flags = history_flags

    def on_train_begin(self, logs={}):
        self.weights_batch = []
        self.weights_epoch = []
        self.losses = []
        modelWeights = dict()
        for layer_index in range(0, len(self.model.layers)):
            current_layer = self.model.layers[layer_index]
            if len(current_layer.weights) > 1:
                modelWeights['layer_{layer_index}'.format(layer_index=layer_index)] = {
                    'weights': current_layer.get_weights()[0], 'bias': current_layer.get_weights()[1]}
            else:
                modelWeights['layer_{layer_index}'.format(layer_index=layer_index)] = current_layer.get_weights()[0]
        self.weights_batch.append(modelWeights)
        self.weights_epoch.append(modelWeights)

    def on_batch_end(self, epoch, logs={}):
        if self.history_flags['batches']:
            modelWeights = dict()
            modelErrors = dict()
            for layer_index in range(0, len(self.model.layers)):
                current_layer = self.model.layers[layer_index]
                if len(current_layer.weights) > 1:
                    modelWeights['layer_{layer_index}'.format(layer_index=layer_index)] = {
                        'weights': current_layer.get_weights()[0], 'bias': current_layer.get_weights()[1]}
                else:
                    modelWeights['layer_{layer_index}'.format(layer_index=layer_index)] = current_layer.get_weights()[0]
            self.weights_batch.append(modelWeights)
        self.losses.append(logs.get('loss'))

    def on_epoch_end(self, epoch, logs={}):
        if self.history_flags['epochs']:
            modelWeights = dict()
            modelErrors = dict()
            for layer_index in range(0, len(self.model.layers)):
                current_layer = self.model.layers[layer_index]
                if len(current_layer.weights) > 1:
                    modelWeights['layer_{layer_index}'.format(layer_index=layer_index)] = {
                        'weights': current_layer.get_weights()[0], 'bias': current_layer.get_weights()[1]}
                else:
                    modelWeights['layer_{layer_index}'.format(layer_index=layer_index)] = current_layer.get_weights()[0]
            self.weights_epoch.append(modelWeights)


def train_index_to_layer_dict_weights_transform(weights_through_training, non_zero_updates_indices, model_layers):
    weights_layers_between_training_steps = {}
    for layer_index in range(len(model_layers)):
        key = 'layer_{layer_index}'.format(layer_index=layer_index)
        weights_layers_between_training_steps[key] = []
    for index in range(0, len(weights_through_training)):
        current_weights = weights_through_training[index]
        for key in current_weights.keys():
            weights_layers_between_training_steps[key].append(current_weights[key])
    for key in weights_layers_between_training_steps.keys():
        weights_layers_between_training_steps[key] = list(
            np.array(weights_layers_between_training_steps[key])[[0] + (non_zero_updates_indices + 1).tolist()[0]])
    return weights_layers_between_training_steps


def get_non_zero_updates_indices(loss_through_training, NON_ZERO_THRESHOLD=0.000000000000001):
    non_zero_indices = np.array(np.where(np.abs(np.diff(np.array(loss_through_training))) > NON_ZERO_THRESHOLD))
    return non_zero_indices


def train_model(model, x_train, y_train, x_test, y_test, number_of_epochs, batch_size,
                history_flags={'batches': True, 'epochs': False}):
    weights_history = WeightsHistory(history_flags)
    weights_history.set_model(model)
    callbacks = [weights_history]
    errors_at_epochs = model.fit(np.array(x_train), np.array(y_train),
                                 validation_data=(np.array(x_test), np.array(y_test)),
                                 epochs=number_of_epochs, batch_size=batch_size,
                                 callbacks=callbacks)
    train_error_at_batch = weights_history.losses
    errors_between_batch_and_epochs_dict = {}
    errors_between_batch_and_epochs_dict['batches'] = {'train_loss': train_error_at_batch}
    errors_between_batch_and_epochs_dict['epochs'] = errors_at_epochs.history
    weights_between_batch_and_epochs_dict = {}
    if history_flags['batches']:
        weights_temp_batch = weights_history.weights_batch
        batch_non_zero_updates = get_non_zero_updates_indices(train_error_at_batch)
        weights_layers_between_batches = train_index_to_layer_dict_weights_transform(weights_temp_batch,
                                                                                     batch_non_zero_updates,
                                                                                     model.layers)
        if type(weights_layers_between_batches.get(list(weights_layers_between_batches.keys())[0])[0]) is dict:
            weights_between_batch_and_epochs_dict['batches'] = dict()
            for layer in weights_layers_between_batches.keys():
                weights_between_batch_and_epochs_dict['batches'][layer] = dict()
                weights_between_batch_and_epochs_dict['batches'][layer]['weights'] = [weights['weights'] for weights in
                                                                                      weights_layers_between_batches[layer]]
                weights_between_batch_and_epochs_dict['batches'][layer]['bias'] = [weights['bias'] for weights in
                                                                                   weights_layers_between_batches[layer]]
        else:

            weights_between_batch_and_epochs_dict['batches'] = weights_layers_between_batches
    if history_flags['epochs']:
        weights_temp_epochs = weights_history.weights_epoch
        epoch_non_zero_updates = get_non_zero_updates_indices(errors_between_batch_and_epochs_dict['epochs']['loss'])
        weights_layers_between_epochs = train_index_to_layer_dict_weights_transform(weights_temp_epochs,
                                                                                    epoch_non_zero_updates,
                                                                                    model.layers)
        weights_between_batch_and_epochs_dict['epochs'] = dict()
        if type(weights_layers_between_epochs.get(list(weights_layers_between_epochs.keys())[0])[0]) is dict:
            for layer in weights_layers_between_epochs.keys():
                weights_between_batch_and_epochs_dict['epochs'][layer] = dict()
                weights_between_batch_and_epochs_dict['epochs'][layer]['weights'] = [weights['weights'] for weights in
                                                                                     weights_layers_between_epochs[layer]]
                weights_between_batch_and_epochs_dict['epochs'][layer]['bias'] = [weights['bias'] for weights in
                                                                                  weights_layers_between_epochs[layer]]
        else:
            weights_between_batch_and_epochs_dict['epochs'] = weights_layers_between_epochs
    return weights_between_batch_and_epochs_dict, errors_between_batch_and_epochs_dict
