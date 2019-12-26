import numpy as np
import os
from keras.layers import LeakyReLU
from keras import optimizers
from keras import losses
from simulator import input_generator, parameters_initialization
from simulator.configurations.utils import set_network_architecture_linearly
from functools import partial

N = 3000
DIMENSION = 50
NUMBER_OF_NEURONS = 2*10
NUMBER_OF_LAYERS = 2
LEARNING_RATE = 0.001
SGD_OPTIMIZER = optimizers.SGD(lr=LEARNING_RATE)
NUMBER_OF_EPOCHS = 2500
RESULTS_PATH = os.path.join("results", "alons_architecture_tests")
TRAIN_TEST_SPLIT_RATIO = 0.75
MARGIN = 0.05
NUMBER_OF_RUNS = 1
LEAKY_RELU_ALPHA=0.3
# Weights initialization is a dictionary of the weights initializations,
# it support both initializations for all layers and for single layers,
# for single layers the the value expected is a list of tuples of (layer_index,initialization_functions)
alons_paper_configuration = {
    'data': {
        'data_provider': partial(input_generator.generate_linearly_separable_samples, margin=MARGIN,
                                 dimension=DIMENSION),
        # "data_provider": partial(input_generator.get_linearly_separable_mnist, demean_flag=False),
        'sample_size': N,
        'sample_dimension': DIMENSION,
        # 'margin': MARGIN,
        'train_test_split_ratio': TRAIN_TEST_SPLIT_RATIO
    },
    'model': {
        'learning_rate': LEARNING_RATE,
        'number_of_layers': NUMBER_OF_LAYERS,
        'network_architecture': set_network_architecture_linearly(NUMBER_OF_NEURONS, NUMBER_OF_LAYERS),
        'activation_type': LeakyReLU(LEAKY_RELU_ALPHA),
        'loss_type': losses.hinge,

        'weights_initialization':
            {'all_layers': partial(parameters_initialization.initialize_all_weights_from_method,
                                   initialization_method=partial(parameters_initialization.random_normal, std=0.025)),
             'single_layers': [(-1, partial(parameters_initialization.set_layer_weights_to_anti_symmetric_constant,
                                            constant=1 / np.sqrt(NUMBER_OF_NEURONS)))]},
        'optimizer': SGD_OPTIMIZER,
        'input_dimension': DIMENSION,
        'number_of_epochs': NUMBER_OF_EPOCHS,
        'number_of_runs': NUMBER_OF_RUNS
    },
    'results': {
        'results_path': RESULTS_PATH,
        'weights':
            {'max_min_ratios_desired_graph_indices': [2, 3]}
    }
}


def get_non_zero_updates_bound(decision_boundary_norm, leaky_alpha, number_of_neurons, initialization_of_second_layer,
                               learning_rate, norm_bound_of_first_layer_weights_initialization):
    M = (decision_boundary_norm ** 2) / (leaky_alpha ** 2) + (decision_boundary_norm ** 2) / (
        number_of_neurons * learning_rate * (initialization_of_second_layer ** 2) * (leaky_alpha ** 2)) + np.sqrt(
        norm_bound_of_first_layer_weights_initialization * (
            8 * (number_of_neurons ** 2) * (learning_rate ** 2) * (initialization_of_second_layer ** 2) + 8 * (
                learning_rate * number_of_neurons))) * (decision_boundary_norm ** 1.5) / (2 * number_of_neurons * (
        learning_rate * initialization_of_second_layer * leaky_alpha) ** 1.5) + 2 * norm_bound_of_first_layer_weights_initialization * decision_boundary_norm / (
        learning_rate * initialization_of_second_layer * leaky_alpha)
    # M = (decision_boundary_norm ** 2) / (leaky_alpha ** 2) + 50*((decision_boundary_norm ** 2) / (min(leaky_alpha,np.sqrt(learning_rate))))
    return M
