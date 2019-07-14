import keras
import keras.backend as K

class WeightsHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.weights = []
        self.losses = []
        self.weights.append(self.model.weights)

    def on_epoch_end(self, epoch, logs={}):
        modelWeights = dict()
        for layer_index in range(0, len(self.model.layers)):
            current_layer = self.model.layers[layer_index]
            modelWeights['layer_{layer_index}'.format(layer_index=layer_index)] = current_layer.get_weights()[0]
        self.weights.append(modelWeights)




def train_model(model, x_train, y_train, x_test, y_test, number_of_epochs):
    weights_history = WeightsHistory()
    weights_history.set_model(model)
    callbacks = [weights_history]
    error_history = model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=number_of_epochs,
                              callbacks=callbacks)
    weights_temp = weights_history.weights
    weights_temp.remove(weights_temp[0])
    weights_layers_epochs = {}
    for layer_index in range(len(model.layers)):
        key = 'layer_{layer_index}'.format(layer_index=layer_index)
        weights_layers_epochs[key] = []
    for index in range(0, len(weights_temp)):
        current_weights = weights_temp[index]
        for key in current_weights.keys():
            weights_layers_epochs[key].append(current_weights[key])
    return weights_layers_epochs, error_history.history
