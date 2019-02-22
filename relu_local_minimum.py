from sklearn.utils import shuffle
import numpy as np
import copy
from model import measure_model_average
import matplotlib.pyplot as plt
import arrow
import os
from input_parser import OrthogonalSingleClassDataProvider

DIMENSION = 800
configuration_parameters = {
    'data': {
        'data_provider': OrthogonalSingleClassDataProvider(),
        'filter_arguments': None,
        'sample_dimension': DIMENSION,
        'shuffle': True,
        'train_test_split_ration': 0.75
    },
    'model': {
        'learning_rate': 0.01,
        'batch_size': 1,
        'delta': 0.99,
        'number_of_neurons': 10,
        'number_of_classes': 1,
        'activation_type': 'standard',
        'number_of_non_random_neurons_initialization': 0,
        'input_dimension': DIMENSION,
        'number_of_epochs': 200,
        'number_of_runs': 5
    }

}

d = configuration_parameters["data"]["sample_dimension"]
train_test_split_ration = configuration_parameters["data"]["train_test_split_ration"]
delta = configuration_parameters["model"]["delta"]
data_provider = configuration_parameters["data"]["data_provider"]
x_data, y_data = data_provider.read(d)
if configuration_parameters["data"]["shuffle"]:
    x_data, y_data = shuffle(x_data, y_data)
x_train = x_data[0:int(x_data.shape[0] * train_test_split_ration), :]
y_train = y_data[0:int(x_data.shape[0] * train_test_split_ration)]
y_test = y_data[int(x_data.shape[0] * train_test_split_ration):int(x_data.shape[0])]
x_test = x_data[int(x_data.shape[0] * train_test_split_ration):int(x_data.shape[0]), :]
small_network_size = np.floor(np.math.log(d / -np.math.log(delta), 2))
large_network_size = np.floor(np.math.log(2 * d / delta, 2))
small_network_sizes = np.linspace(3, small_network_size, num=3, dtype=int)

# big_network_sizes = np.linspace(small_network_size, 10 * large_network_size, num=4, dtype=int)
# network_sizes = np.power(2, range(4, 9))
# network_sizes = np.concatenate((small_network_sizes, big_network_sizes), axis=None)
network_sizes = np.arange(2, 9, step=1, dtype=int)
for network_size in network_sizes:
    temp_configuration = copy.deepcopy(configuration_parameters)
    temp_configuration["model"]["number_of_neurons"] = network_size
    temp_configuration["model"]["number_of_non_random_neurons_initialization"] = int(float(network_size) / 2)
    train_error, test_error, avg_loss = measure_model_average(x_train, y_train, x_test, y_test,
                                                              temp_configuration)
    number_of_epochs = temp_configuration["model"]["number_of_epochs"]
    f_1 = plt.figure(1)
    plt.scatter(range(number_of_epochs), train_error)
    plt.xlabel('Epochs')
    plt.ylabel('train_error')
    f_2 = plt.figure(2)
    plt.scatter(range(number_of_epochs), test_error)
    plt.xlabel('Epochs')
    plt.ylabel('test_error')
    f_3 = plt.figure(3)
    plt.scatter(range(number_of_epochs), avg_loss)
    plt.xlabel('Epochs')
    plt.ylabel('cost_error')
    current_date = str(arrow.now().format('YYYY-MM-DD'))

    configuration_path = "number_of_neurons_{number_of_neurons}sample_dimension_{sample_dimension}_activation_type_{activation_type}_non_random_neurons_initialization_{non_random_neurons_initialization}".format(
        sample_dimension=temp_configuration["data"]["sample_dimension"],
        activation_type=temp_configuration["model"]["activation_type"],
        number_of_runs=temp_configuration["model"]["number_of_runs"],
        number_of_neurons=temp_configuration["model"]["number_of_neurons"],
        non_random_neurons_initialization=temp_configuration["model"]["number_of_non_random_neurons_initialization"])
    results_path = "{cwd_path}\{current_date}\one_layer\local_minimum_of_relu\{configuration_path}".format(
        cwd_path=str(os.getcwd()),
        current_date=current_date, configuration_path=configuration_path)
    if not os.path.isdir(results_path):
        os.makedirs(results_path)
    configuration_parameters_text = open(
        "{results_path}\configuration_parameters.txt".format(
            results_path=results_path), "w")
    configuration_parameters_text.write(str(temp_configuration))
    configuration_parameters_text.close()

    f_1.savefig("{results_path}\epochs_train_error.png".format(
        results_path=results_path))
    f_2.savefig("{results_path}\epoches_test_error.png".format(
        results_path=results_path))
    f_3.savefig(
        "{results_path}\epochs_cost_error.png".format(
            results_path=results_path))
    f_1.clf()
    f_2.clf()
    f_3.clf()
