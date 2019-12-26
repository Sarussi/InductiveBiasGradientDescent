import numpy as np


def set_network_architecture_linearly(max_neurons_in_layer, number_of_layers):
    neurons_in_layers_array = np.linspace(start=max_neurons_in_layer, stop=1, num=number_of_layers,
                                          dtype=int)
    number_of_neurons_in_layers = dict()
    for layer_index in range(0, len(neurons_in_layers_array)):
        current_layer_key = 'layer_{index}'.format(index=layer_index + 1)
        number_of_neurons_in_layers[current_layer_key] = neurons_in_layers_array[layer_index]
    return number_of_neurons_in_layers

def set_network_architecture_linearly(max_neurons_in_layer, number_of_layers):
    neurons_in_layers_array = np.linspace(start=max_neurons_in_layer, stop=1, num=number_of_layers,
                                          dtype=int)
    number_of_neurons_in_layers = dict()
    for layer_index in range(0, len(neurons_in_layers_array)):
        current_layer_key = 'layer_{index}'.format(index=layer_index + 1)
        number_of_neurons_in_layers[current_layer_key] = neurons_in_layers_array[layer_index]
    return number_of_neurons_in_layers
