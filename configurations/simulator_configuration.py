import numpy as np
import os
from keras.layers import LeakyReLU
from keras import optimizers
from keras import losses
import input_generator, create_model
from  configurations.utils import set_network_architecture_linearly

DIMENSION = 10
N = 1000
MARGIN = 0.05
NUMBER_OF_LAYERS = 2
MAX_NEURONS_IN_LAYER = 10
TRAIN_TEST_SPLIT_RATIO = 0.7
NUMBER_OF_EPOCHS = 1000
RESULTS_PATH = os.path.join("results", "._tests")
ACTIVATION_FUNCTION = LeakyReLU()
LEARNING_RATE = 0.001
SGD_OPTIMIZER = optimizers.SGD(lr=LEARNING_RATE)
configuration = {
    'data': {
        'data_provider': input_generator.generate_linearly_separable_samples,
        'sample_size': N,
        'sample_dimension': DIMENSION,
        'margin': MARGIN,
        'train_test_split_ratio': TRAIN_TEST_SPLIT_RATIO
    },

    'model': {
        'learning_rate': LEARNING_RATE,
        'network_architecture': set_network_architecture_linearly(MAX_NEURONS_IN_LAYER, NUMBER_OF_LAYERS),
        'activation_type': ACTIVATION_FUNCTION,
        'loss_type': create_model.logistic_loss,
        'optimizer': SGD_OPTIMIZER,
        'input_dimension': DIMENSION,
        'number_of_epochs': NUMBER_OF_EPOCHS,
        'number_of_runs': 1
    },
    'results': {
        'results_path': RESULTS_PATH
    }

}
