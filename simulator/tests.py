import functools

from keras import activations
from keras.layers import LeakyReLU
from keras import optimizers
from simulator import create_model
from simulator import simulator_configuration
from simulator import parameters_initialization
from simulator import input_generator
from simulator import train_model
from simulator import visualize
import matplotlib.pyplot as plt
import os
import numpy as np
from simulator import simulator_configuration


def array_of_dictionaries_mean(dict_list):
    number_of_dict = float(len(dict_list))
    mean_dict ={}
    for key in list(dict_list[0].keys()):
        mean_dict[key] = np.zeros(len(dict_list[0][key]))
        for dictionary in dict_list:
            mean_dict[key] += np.array(dictionary[key])
        mean_dict[key] /= number_of_dict
    return mean_dict


network_architecture = simulator_configuration.configuration_parameters['model']['network_architecture']
model = create_model.create_model(list(network_architecture.values()),
                                  simulator_configuration.configuration_parameters['model']['activation_type'],
                                  simulator_configuration.configuration_parameters['model']['input_dimension'],
                                  simulator_configuration.configuration_parameters['model']['optimizer'],
                                  simulator_configuration.configuration_parameters['model']['loss_type'])

# parameters_initialization.initialize_all_weights(model, initialization_method=functools.partial(
#     parameters_initialization.random_normal, mean=0, std=0.1))
x_data, y_data = simulator_configuration.configuration_parameters['data']['data_provider'](
    simulator_configuration.configuration_parameters['data']['sample_size'],
    simulator_configuration.configuration_parameters['data']['sample_dimension'],
    simulator_configuration.configuration_parameters['data']['margin'])
x_train, y_train, x_test, y_test = input_generator.train_and_test_shuffle_split(x_data, y_data,
                                                                                train_test_split_ratio=
                                                                                simulator_configuration.configuration_parameters[
                                                                                    'data']['train_test_split_ratio'])
weights_max_min_ratios_list = []
weights_norm_ratios_list = []
errors_dict_list = []
for _ in range(simulator_configuration.configuration_parameters['model']['number_of_runs']):
    model = create_model.create_model(list(network_architecture.values()),
                                      simulator_configuration.configuration_parameters['model']['activation_type'],
                                      simulator_configuration.configuration_parameters['model']['input_dimension'],
                                      simulator_configuration.configuration_parameters['model']['optimizer'],
                                      simulator_configuration.configuration_parameters['model']['loss_type'])
    weights, errors = train_model.train_model(model, x_train, y_train, x_test, y_test,
                                              simulator_configuration.configuration_parameters['model'][
                                                  'number_of_epochs'])
    _, weights_max_min_ratios = visualize.extract_weights_values_ratio(weights)
    weights_max_min_ratios_list.append(weights_max_min_ratios)
    weights_2_vs_fro_norm_ratio = visualize.extract_weights_norms_ratio(weights)
    weights_norm_ratios_list.append(weights_2_vs_fro_norm_ratio)
    errors_dict_list.append(errors)
weights_max_min_ratios_avg = array_of_dictionaries_mean(weights_max_min_ratios_list)
weights_norm_ratios_avg = array_of_dictionaries_mean(weights_norm_ratios_list)
errors_avg = array_of_dictionaries_mean(errors_dict_list)
visualize.visualize_weights_max_min_values_ratio(weights_max_min_ratios_avg,
                                                 simulator_configuration.configuration_parameters['results'][
                                                     'results_path'])
visualize.visualize_errors(errors_avg, simulator_configuration.configuration_parameters['results']['results_path'])
visualize.visualize_weights_norms_ratio(weights_norm_ratios_avg, simulator_configuration.configuration_parameters['results']['results_path'])
