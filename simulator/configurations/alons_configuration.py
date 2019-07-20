import numpy as np
import os
from keras.layers import LeakyReLU
from keras import optimizers
from keras import losses
from simulator import input_generator, parameters_initialization
from simulator.configurations.utils import set_network_architecture
from functools import partial

N = 4000
DIMENSION = 30
NUMBER_OF_NEURONS = 10
NUMBER_OF_LAYERS = 2
LEARNING_RATE = 0.001
SGD_OPTIMIZER = optimizers.SGD(lr=LEARNING_RATE)
NUMBER_OF_EPOCHS = 3000
RESULTS_PATH = os.path.join("results", "alons_architecture_tests")
TRAIN_TEST_SPLIT_RATIO = 0.75
MARGIN = 0.05
# Weights initialization is a dictionary of the weights initializations,
# it support both initializations for all layers and for single layers,
# for single layers the the value expected is a list of tuples of (layer_index,initialization_functions)
alons_paper_configuration = {
    'data': {
        'data_provider': partial(input_generator.generate_linearly_separable_samples,margin=MARGIN,dimension=DIMENSION),
        'sample_size': N,
        'sample_dimension': DIMENSION,
        # 'margin': MARGIN,
        'train_test_split_ratio': TRAIN_TEST_SPLIT_RATIO
    },
    'model': {
        'learning_rate': LEARNING_RATE,
        'number_of_layers': NUMBER_OF_LAYERS,
        'network_architecture': set_network_architecture(NUMBER_OF_NEURONS, NUMBER_OF_LAYERS),
        'activation_type': LeakyReLU(),
        'loss_type': losses.hinge,

        'weights_initialization':
            {'single_layers': [(-1, partial(parameters_initialization.set_layer_weights_to_anti_symmetric_constant,
                                            constant=1 / np.sqrt(NUMBER_OF_NEURONS)))]},
        'optimizer': SGD_OPTIMIZER,
        'input_dimension': DIMENSION,
        'number_of_epochs': NUMBER_OF_EPOCHS,
        'number_of_runs': 3
    },
    'results': {
        'results_path': RESULTS_PATH,
        'weights':
            {'max_min_ratios_desired_graph_indices': [2, 3]}
    }
}
