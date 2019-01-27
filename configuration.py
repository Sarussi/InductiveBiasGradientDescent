from input_parser import MNISTDataProvider, GaussianLinearSeparableDataProvider

DIMENSION = 70
configuration_parameters = {
    'data': {
        'data_provider': GaussianLinearSeparableDataProvider(margin=0.5),
        'filter_arguments': None,
        'sample_size': 1000,
        'sample_dimension': DIMENSION,
        'shuffle': True,
        'train_test_split_ration': 0.75
    },
    'model': {
        'learning_rate': 0.1,
        'batch_size': 1,
        'number_of_neurons': 10,
        'number_of_classes': 1,
        'input_dimension': DIMENSION,
        'number_of_epochs': 100,
        'number_of_runs': 5
    }

}
