import numpy as np
import os
from keras.layers import LeakyReLU
from keras.activations import relu
from keras import optimizers
from keras import losses
from simulator import input_generator, create_model
from simulator.configurations.utils import set_network_architecture_linearly
from functools import partial
from simulator import parameters_initialization

DIMENSION = 10
N = 1000
MARGIN = 0.05
TRAIN_TEST_SPLIT_RATIO = 0.7
NUMBER_OF_EPOCHS = 10
RESULTS_PATH = os.path.join("results", "linear_regression_tests")
ACTIVATION_FUNCTION = LeakyReLU()
# ACTIVATION_FUNCTION = relu
LEARNING_RATE = 0.1
SGD_OPTIMIZER = optimizers.Adagrad(lr=LEARNING_RATE)
configuration = {
    'data': {
        'data_provider': partial(
            input_generator.get_linear_ground_truth, dimension=DIMENSION, slope_matrix=np.ones(DIMENSION),
            intercept=0),
        'sample_size': N,
        'sample_dimension': DIMENSION,
        'train_test_split_ratio': TRAIN_TEST_SPLIT_RATIO
    },

    'model': {
        'learning_rate': LEARNING_RATE,
        'network_architecture': {'layer_0': {'number_of_neruons': 200, 'is_trainable': True},
                                 'layer_1': {'number_of_neruons': 1, 'is_trainable': False}},
        'activation_type': ACTIVATION_FUNCTION,
        'loss_type': create_model.mse_loss,
        'weights_initialization':
            {'all_layers': partial(parameters_initialization.initialize_all_weights_from_method,
                                   initialization_method=partial(parameters_initialization.random_normal, std=1))},
        'evaluation_metrics': losses.mean_absolute_error,
        'optimizer': SGD_OPTIMIZER,
        'input_dimension': DIMENSION,
        'number_of_epochs': NUMBER_OF_EPOCHS,
        'number_of_runs': 3
    },
    'results': {
        'results_path': RESULTS_PATH,
        'weights':
            {'max_min_ratios_desired_graph_indices': [0, 1, 2, 3],
             'norms_ratios_desired_graph_indices': [0, 1]}
    }

}
