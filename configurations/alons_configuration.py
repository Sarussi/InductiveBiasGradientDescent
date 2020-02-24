import numpy as np
import os
import arrow
from keras.layers import LeakyReLU
from keras import optimizers
from keras import activations
from keras import losses
import input_generator, parameters_initialization, create_model
from  configurations.utils import set_network_architecture_linearly
from functools import partial

from  input_generator import generate_separation_boundary

print("Using alons paper configuration(classification), updates imports in metrics_visualizer")

CURRENT_TIME_STR = str(arrow.now().format('YYYY-MM-DD_HH_mm'))
np.random.seed(int(arrow.now().format('MMDDHHmm')))
DIMENSION = 2
# N = 2 * (2 ** (DIMENSION))
N = 500
NUMBER_OF_NEURONS = 50
LEARNING_RATE = 0.5
SGD_OPTIMIZER = optimizers.SGD(lr=LEARNING_RATE)
# SGD_OPTIMIZER = optimizers.Adagrad(lr=LEARNING_RATE)
NUMBER_OF_EPOCHS = 500
RESULTS_PATH = os.path.join("results", "alons_architecture_tests")
TRAIN_TEST_SPLIT_RATIO = 0.1
MARGIN = 1
NUMBER_OF_RUNS = 1
LEAKY_RELU_ALPHA = 0.3
W_STAR = generate_separation_boundary(DIMENSION, MARGIN)
# SYMMETRY_INDEX = 1
# W_STAR = np.zeros((1, DIMENSION))
# W_STAR[:, SYMMETRY_INDEX] = -2 / MARGIN
# HISTORY_FLAGS = {'batch': False, 'epoch': True}
# Weights initialization is a dictionary of the weights initializations,
# it support both initializations for all layers and for single layers,
# for single layers the the value expected is a list of tuples of (layer_index,initialization_functions)
alons_paper_configuration = {
    'data': {
        'data_provider': partial(input_generator.generate_two_seperable_balls, first_label=1,
                                 second_label=0, dimension=DIMENSION,
                                 positive_center_loc=np.array([0.5, 0]), radius=0.2),
        # 'data_provider': partial(input_generator.generate_linearly_separable_samples, first_label=1,
        #                          second_label=0,
        #                          dimension=DIMENSION, separating_boundary=W_STAR,
        #                          samples_generator=partial(
        #                              input_generator.generate_two_simplex,
        #                              symmetry_index=SYMMETRY_INDEX)),
        # "data_provider": partial(input_generator.get_linearly_separable_mnist, first_label=1,
        #                          second_label=-1, demean_flag=False),
        # 'data_provider': partial(input_generator.generate_linearly_separable_samples, first_label=1,
        #                          second_label=-1,
        #                          dimension=DIMENSION, separating_boundary=W_STAR,
        #                          samples_generator=input_generator.radamacher_cube_vertices),
        'sample_size': N,
        'sample_dimension': DIMENSION,
        'w_star': W_STAR,
        'first_label': 1,
        'second_label': 0,
        'train_test_split_ratio': TRAIN_TEST_SPLIT_RATIO
    },
    'model': {
        'learning_rate': LEARNING_RATE,
        'number_of_neurons': 1,
        'network_architecture': {
            'layer_0': {'number_of_neruons': 1, 'use_bias': False, 'is_trainable': True}},
        # 'layer_0': {'number_of_neruons': 2 * NUMBER_OF_NEURONS, 'use_bias': False, 'is_trainable': True},
        # 'layer_1': {'number_of_neruons': 1, 'use_bias': False, 'is_trainable': False}},
        'activation_type': activations.linear,
        'loss_type': create_model.cross_entropy,
        # 'loss_type': losses.hinge,
        # 'point_wise_loss': create_model.hinge_point_wise_loss,
        'point_wise_loss': create_model.cross_entropy_point_wise_loss,
        'weights_initialization':
        # {'all_layers': parameters_initialization.initialize_all_weights_from_method},
        # {'all_layers': partial(parameters_initialization.initialize_all_weights_from_method,
        #                        initialization_method=partial(parameters_initialization.random_normal, mean=1,
        #                                                      std=0.0025)),
            {'single_layers': [(0, partial(parameters_initialization.set_weights_to_constant_vector, vector=0.01*np.array([1, 1])))]},
        # 'single_layers': [(-1, partial(parameters_initialization.set_layer_weights_to_anti_symmetric_constant,
        #                                constant=1 / np.sqrt(2 * NUMBER_OF_NEURONS)))]},
        'evaluation_metrics': None,
        'optimizer': SGD_OPTIMIZER,
        'input_dimension': DIMENSION,
        'number_of_epochs': NUMBER_OF_EPOCHS,
        'batch_size': 1,
        'history_flags': {'batches': False, 'epochs': True},
        'number_of_runs': NUMBER_OF_RUNS
    },
    'tests': {
        'perceptron_layer': 'layer_0',
        'prediction_landscape': {'mesh_grid_step': 0.01}
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
