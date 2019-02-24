from input_parser import MNISTDataProvider, GaussianLinearSeparableDataProvider

DIMENSION = 300
SAMPLE_SIZE = 20*DIMENSION
configuration_parameters = {
    'data': {
        'data_provider': GaussianLinearSeparableDataProvider(margin=0.001),
        'filter_arguments': None,
        'sample_size': SAMPLE_SIZE,
        'sample_dimension': DIMENSION,
        'shuffle': True,
        'train_test_split_ration': 0.75
    },
    'model': {
        'learning_rate': 0.01,
        'batch_size': 1,
        'number_of_neurons': 40,
        'number_of_classes': 1,
        'activation_type': 'leaky',
        'input_dimension': DIMENSION,
        'number_of_epochs': 100,
        'number_of_runs': 5
    }

}
