from input_parser import GaussianLinearSeparableDataProvider

DIMENSION = 2
SAMPLE_SIZE = 100 * DIMENSION
configuration_parameters = {
    'data': {
        'data_provider': GaussianLinearSeparableDataProvider(margin=0.1),
        'filter_arguments': None,
        'sample_size': SAMPLE_SIZE,
        'sample_dimension': DIMENSION,
        'shuffle': True,
        'train_test_split_ration': 0.75
    },
    'model': {
        'learning_rate': 0.01,
        'batch_size': 1,
        'number_of_neurons_first_layer': 10,
        'number_of_neurons_second_layer': 10,
        'activation_type': 'leaky',
        'number_of_classes': 1,
        'input_dimension': DIMENSION,
        'number_of_epochs': 100,
        'number_of_runs': 10
    }

}


