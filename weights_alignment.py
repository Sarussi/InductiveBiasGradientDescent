from sklearn.utils import shuffle
import numpy as np
import matplotlib.pyplot as plt
from input_parser import MNISTDataProvider, GaussianLinearSeparableDataProvider
from model import measure_model
import copy
import arrow
import os

DIMENSION = 10
SAMPLE_SIZE = 300 * DIMENSION
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
        'decreasing_learning_rate': True,
        'batch_size': 1,
        'number_of_neurons_in_layers': {'layer_1': 100,'layer_2':10, 'layer_3': 5,'layer_4':1},
        'number_of_classes': 1,
        'activation_type': 'leaky',
        'loss_type': 'logistic',
        'input_dimension': DIMENSION,
        'number_of_epochs': 500,
        'number_of_runs': 5
    }

}

d = configuration_parameters["data"]["sample_dimension"]
N = configuration_parameters["data"]["sample_size"]
train_test_split_ration = configuration_parameters["data"]["train_test_split_ration"]
data_provider = configuration_parameters["data"]["data_provider"]
x_data, y_data = data_provider.read(N, d)
if configuration_parameters["data"]["shuffle"]:
    x_data, y_data = shuffle(x_data, y_data)
x_train = x_data[0:int(x_data.shape[0] * train_test_split_ration), :]
y_train = y_data[0:int(x_data.shape[0] * train_test_split_ration)]
y_test = y_data[int(x_data.shape[0] * train_test_split_ration):int(x_data.shape[0])]
x_test = x_data[int(x_data.shape[0] * train_test_split_ration):int(x_data.shape[0]), :]

train_error, test_error, avg_loss, weights = measure_model(x_train, y_train, x_test, y_test,
                                                           configuration_parameters)

weights_layers_epochs = {}
for key in weights[0].keys():
    weights_layers_epochs[key] = []
for index in range(0, len(weights)):
    current_weights = weights[index]
    for key in current_weights.keys():
        weights_layers_epochs[key].append(current_weights[key])
weights_norm = {}
for key in weights_layers_epochs.keys():
    weights_norm[key] = [np.linalg.norm(current_weights, ord='fro') for current_weights in weights_layers_epochs[key]]


def get_wprod(weights):
    wprod_epochs = []
    for epoch_weight in weights:
        epoch_weight_values = list(epoch_weight.values())
        norms_multiplication = np.linalg.norm(epoch_weight_values[0], ord='fro')
        matrix_multiplication = epoch_weight_values[0]
        for index in range(1, len(epoch_weight_values)):
            matrix_multiplication = np.matmul(epoch_weight_values[index], matrix_multiplication)
            norms_multiplication *= np.linalg.norm(epoch_weight_values[index], ord='fro')
        wprod_epochs.append(matrix_multiplication / norms_multiplication)
    return wprod_epochs


def get_v1(weights):
    v1_epochs = []
    for epoch_weight in weights:
        first_layer_weights = list(epoch_weight.values())[0]
        u, s, vh = np.linalg.svd(first_layer_weights / np.linalg.norm(first_layer_weights, ord='fro'))
        v1_epochs.append(vh[0, :])
    return v1_epochs


v1_epochs = get_v1(weights)
wprod_epochs = get_wprod(weights)
wprod_v1_product_epochs = []
for index in range(0, len(v1_epochs)):
    wprod_v1_product_epochs.append(np.abs(np.dot(wprod_epochs[index], v1_epochs[index])))

#
number_of_epochs = configuration_parameters["model"]["number_of_epochs"]
plot_index = 1
weight_plots_dict = {}
f_1 = plt.figure(plot_index)
for key in weights_norm.keys():
    weight_plots_dict[key] = plt.scatter(range(number_of_epochs), weights_norm[key], label=key)
plt.legend(tuple(weight_plots_dict.values()), tuple(weight_plots_dict.keys()))
plt.xlabel('Epochs')
plt.ylabel('norm')
plot_index += 1
f_2 = plt.figure(plot_index)
plt.scatter(range(number_of_epochs), wprod_v1_product_epochs)
plt.xlabel('Epochs')
plt.ylabel('wprod_v1_product')
plot_index += 1

f_3 = plt.figure(plot_index)
plt.scatter(range(number_of_epochs), avg_loss)
plt.xlabel('Epochs')
plt.ylabel('cost_error')
current_date = str(arrow.now().format('YYYY-MM-DD'))

configuration_path = "sample_dimension_{sample_dimension}_activation_type_{activation_type}_loss_type_{loss_type}_network_architecture_{number_of_neurons_for_layer}".format(
    sample_dimension=configuration_parameters["data"]["sample_dimension"],
    activation_type=configuration_parameters["model"]["activation_type"],
    loss_type=configuration_parameters["model"]["loss_type"],
    number_of_neurons_for_layer=str(list(configuration_parameters["model"]["number_of_neurons_in_layers"].values())))
results_path = "{cwd_path}\{current_date}\weights_alignment\{configuration_path}".format(
    cwd_path=str(os.getcwd()),
    current_date=current_date, configuration_path=configuration_path)
if not os.path.isdir(results_path):
    os.makedirs(results_path)
configuration_parameters_text = open(
    "{results_path}\configuration_parameters.txt".format(
        results_path=results_path), "w")
configuration_parameters_text.write(str(configuration_parameters))
configuration_parameters_text.close()

f_1.savefig("{results_path}\layers_norm.png".format(
    results_path=results_path))
f_2.savefig("{results_path}\wprod_v1_product.png".format(
    results_path=results_path))
f_3.savefig(
    "{results_path}\epochs_cost_error.png".format(
        results_path=results_path))
f_1.clf()
f_2.clf()
f_3.clf()
