from input_parser import MNISTDataProvider, GaussianLinearSeparableDataProvider

DIMENSION = 10
SAMPLE_SIZE = 50*DIMENSION
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
        'number_of_neurons': 2,
        'number_of_classes': 1,
        'activation_type': 'leaky',
        'input_dimension': DIMENSION,
        'number_of_epochs': 100,
        'number_of_runs': 5
    }

}
