import numpy as np
import os
from keras.layers import LeakyReLU
from keras.activations import relu, linear
from keras import optimizers
from keras import losses
import input_generator, create_model
from  configurations.utils import set_network_architecture_linearly
from functools import partial
import parameters_initialization
import arrow

print("Using linear regression configuration, updates imports in metrics_visualizer")
CURRENT_TIME_STR = str(arrow.now().format('YYYY-MM-DD_HH_mm'))
np.random.seed(int(arrow.now().format('MMDDHHmm')))
DIMENSION = 1
N = 100
MARGIN = 0.05
TRAIN_TEST_SPLIT_RATIO = 0.1
NUMBER_OF_EPOCHS = 100
RESULTS_PATH = os.path.join("results", "linear_regression_tests")
ACTIVATION_FUNCTION = LeakyReLU()
# ACTIVATION_FUNCTION = relu
LEARNING_RATE = 0.01
# SGD_OPTIMIZER = optimizers.Adagrad(lr=LEARNING_RATE)
SGD_OPTIMIZER = optimizers.SGD(lr=LEARNING_RATE)
W_STAR = np.random.randn(DIMENSION)
W_STAR = np.full((DIMENSION,), -0.75010247)
INTERCEPT = 0
configuration = {
    'data': {
        'w_star': W_STAR,
        'data_provider': partial(
            input_generator.get_linear_ground_truth, dimension=DIMENSION,
            slope_matrix=W_STAR,
            intercept=INTERCEPT,
            samples_generator=partial(input_generator.random_normal_samples, sigma=1)),
        'sample_size': N,
        'sample_dimension': DIMENSION,
        'train_test_split_ratio': TRAIN_TEST_SPLIT_RATIO
    },

    'model': {
        'learning_rate': LEARNING_RATE,
        'network_architecture': {'layer_0': {'number_of_neruons': 2 * 400, 'use_bias': False, 'is_trainable': True},
                                 'layer_1': {'number_of_neruons': 1, 'use_bias': False, 'is_trainable': False}},
        'activation_type': ACTIVATION_FUNCTION,
        'loss_type': create_model.mse_loss,
        'point_wise_loss': create_model.regression_point_wise_loss,
        'weights_initialization':
            {'all_layers': partial(parameters_initialization.initialize_all_weights_from_method,
                                   initialization_method=partial(parameters_initialization.random_normal, std=1)),
             'single_layers': [(-1, partial(parameters_initialization.set_layer_weights_to_anti_symmetric_constant,
                                            constant=1 / np.sqrt(400)))]
             },
        'evaluation_metrics': None,
        'optimizer': SGD_OPTIMIZER,
        'input_dimension': DIMENSION,
        'number_of_epochs': NUMBER_OF_EPOCHS,
        'batch_size': 1,
        # 'batch_size': int(N * TRAIN_TEST_SPLIT_RATIO),
        'number_of_runs': 3
    },
    'results': {
        'results_path': RESULTS_PATH,
        'weights':
            {'max_min_ratios_desired_graph_indices': [0, 1, 2, 3],
             'norms_ratios_desired_graph_indices': [0, 1]}
    }

}
