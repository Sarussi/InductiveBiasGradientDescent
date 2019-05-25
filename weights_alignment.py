
from sklearn.utils import shuffle
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from input_parser import MNISTDataProvider, GaussianLinearSeparableDataProvider
from model import measure_model
import copy
import arrow
import os
import time
from datetime import timedelta
start_time = time.time()

def set_network_archeitecture(max_neurons_in_layer, number_of_layers):
    neurons_in_layers_array = np.linspace(start=max_neurons_in_layer, stop=1, num=number_of_layers,
                                          dtype=int)
    number_of_neurons_in_layers = dict()
    for layer_index in range(0, len(neurons_in_layers_array)):
        current_layer_key = 'layer_{index}'.format(index=layer_index + 1)
        number_of_neurons_in_layers[current_layer_key] = neurons_in_layers_array[layer_index]
    return number_of_neurons_in_layers


DIMENSION = 20
SAMPLE_SIZE = 100 * DIMENSION
NUMBER_OF_LAYERS = 3
MAX_NEURONS_IN_LAYER = 5
MARGIN = 0.00000001
configuration_parameters = {
    'data': {
        'data_provider': GaussianLinearSeparableDataProvider(margin=MARGIN),
        'filter_arguments': None,
        'sample_size': SAMPLE_SIZE,
        'sample_dimension': DIMENSION,
        'shuffle': True,
        'train_test_split_ration': 0.75
    },

    'model': {
        'learning_rate': 0.001,
        'increasing_learning_rate': True,
        'batch_size': 1,
        'number_of_neurons_in_layers': set_network_archeitecture(MAX_NEURONS_IN_LAYER, NUMBER_OF_LAYERS),
        'number_of_classes': 1,
        'activation_type': 'relu',
        'loss_type': 'logistic',
        'input_dimension': DIMENSION,
        'number_of_epochs': 100,
        'number_of_runs': 3
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


def average_weights_measurments(configuration_parameters, x_train, y_train, x_test, y_test):
    number_of_runs = configuration_parameters['model']['number_of_runs']
    number_of_epochs = configuration_parameters['model']['number_of_epochs']
    wprod_v1_product_epochs = np.zeros(number_of_epochs)
    avg_loss = np.zeros(number_of_epochs)
    weights_2_vs_fro_norm_ratio = {}
    weights_2_vs_nuc_norm_ratio = {}
    gradients_fro_norms = {}
    for key in configuration_parameters['model']['number_of_neurons_in_layers']:
        weights_2_vs_fro_norm_ratio[key] = np.zeros(number_of_epochs)
        weights_2_vs_nuc_norm_ratio[key] = np.zeros(number_of_epochs)
        gradients_fro_norms[key] = np.zeros(number_of_epochs)
    for run in range(0, number_of_runs):
        train_error_temp, test_error_temp, avg_loss_temp, weights_temp, gradients_between_epochs = measure_model(
            x_train, y_train, x_test, y_test,
            configuration_parameters)
        aa=1
        avg_loss += avg_loss_temp
        weights_layers_epochs_temp = {}
        for key in weights_temp[0].keys():
            weights_layers_epochs_temp[key] = []
        for index in range(0, len(weights_temp)):
            current_weights = weights_temp[index]
            for key in current_weights.keys():
                weights_layers_epochs_temp[key].append(current_weights[key])

        gradients_layers_epochs_temp = {}
        for key in gradients_between_epochs[0].keys():
            gradients_layers_epochs_temp[key] = []
        for index in range(0, len(gradients_between_epochs)):
            current_gradients = gradients_between_epochs[index]
            for key in current_gradients.keys():
                gradients_layers_epochs_temp[key].append(current_gradients[key])

        for key in weights_layers_epochs_temp.keys():
            weights_2_vs_fro_norm_ratio[key] += [
                np.linalg.norm(current_weights, ord=2) / np.linalg.norm(current_weights, ord='fro') for current_weights
                in
                weights_layers_epochs_temp[key]]
            weights_2_vs_nuc_norm_ratio[key] += [
                np.linalg.norm(current_weights, ord=2) / np.linalg.norm(current_weights, ord='nuc') for current_weights
                in
                weights_layers_epochs_temp[key]]
        for key in gradients_layers_epochs_temp.keys():
            gradients_fro_norms[key] += [np.linalg.norm(current_gradient, ord='fro') for current_gradient in
                                         gradients_layers_epochs_temp[key]]

        v1_epochs = get_v1(weights_temp)
        wprod_epochs_temp = get_wprod(weights_temp)
        wprod_v1_product_epochs_temp = []
        for index in range(0, len(v1_epochs)):
            wprod_v1_product_epochs_temp.append(np.abs(np.dot(wprod_epochs_temp[index], v1_epochs[index]))[0])
        wprod_v1_product_epochs += wprod_v1_product_epochs_temp
    for key in weights_2_vs_fro_norm_ratio.keys():
        weights_2_vs_fro_norm_ratio[key] /= number_of_runs
        weights_2_vs_nuc_norm_ratio[key] /= number_of_runs
    for key in gradients_fro_norms.keys():
        gradients_fro_norms[key] /= number_of_runs
    wprod_v1_product_epochs /= number_of_runs
    avg_loss /= number_of_runs
    return weights_2_vs_fro_norm_ratio, wprod_v1_product_epochs, avg_loss, weights_2_vs_nuc_norm_ratio, gradients_fro_norms


def plot_single_archeticture_vs_epochs(configuration_parameters, x_train, y_train, x_test,
                                       y_test):
    weights_2_vs_fro_norm, wprod_v1_product_epochs, avg_loss, weights_nuc_vs_fro_norm_ratio,gradients_fro_norms = average_weights_measurments(
        configuration_parameters, x_train,
        y_train, x_test,
        y_test)
    number_of_epochs = configuration_parameters["model"]["number_of_epochs"]
    plot_index = 1
    weight_plots_dict = {}
    gradients_plots_dict = {}
    f_1 = plt.figure(plot_index)
    for key in list(weights_2_vs_fro_norm.keys())[:-1]:
        weight_plots_dict[key] = plt.scatter(range(number_of_epochs), weights_2_vs_fro_norm[key], label=key)
    plt.legend(tuple(weight_plots_dict.values()), tuple(weight_plots_dict.keys()))
    plt.xlabel('Epochs')
    plt.ylabel('Norm 2 vs frobenius ratio')
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
    plot_index += 1
    f_4 = plt.figure(plot_index)
    for key in list(weights_2_vs_fro_norm.keys())[:-1]:
        weight_plots_dict[key] = plt.scatter(range(number_of_epochs), weights_2_vs_fro_norm[key], label=key)
    plt.legend(tuple(weight_plots_dict.values()), tuple(weight_plots_dict.keys()))
    plt.xlabel('Epochs')
    plt.ylabel('Norm 2 vs nuclear ratio')
    plot_index += 1
    f_5 = plt.figure(plot_index)
    for key in list(gradients_fro_norms.keys())[:-1]:
        gradients_plots_dict[key] = plt.scatter(range(number_of_epochs), gradients_fro_norms[key], label=key)
    plt.legend(tuple(gradients_plots_dict.values()), tuple(gradients_plots_dict.keys()))
    plt.xlabel('Epochs')
    plt.ylabel('Gradients norm')
    current_date = str(arrow.now().format('YYYY-MM-DD'))
    configuration_path = "dimension_{sample_dimension}_sample_size_{sample_size}_epochs_{number_of_epochs}_margin_{margin}_activation_type_{activation_type}_loss_type_{loss_type}_network_architecture_{number_of_neurons_for_layer}".format(
        sample_dimension=configuration_parameters["data"]["sample_dimension"],
        sample_size=configuration_parameters["data"]["sample_size"],
        number_of_epochs = configuration_parameters["model"]["number_of_epochs"],
        margin=MARGIN,
        activation_type=configuration_parameters["model"]["activation_type"],
        loss_type=configuration_parameters["model"]["loss_type"],
        number_of_neurons_for_layer=str(
            list(configuration_parameters["model"]["number_of_neurons_in_layers"].values())))
    results_path = "{cwd_path}/{current_date}\weights_alignment/{configuration_path}".format(
        cwd_path=str(os.getcwd()),
        current_date=current_date, configuration_path=configuration_path)
    if not os.path.isdir(results_path):
        os.makedirs(results_path)
    configuration_parameters_text = open(
        "{results_path}/configuration_parameters.txt".format(
            results_path=results_path), "w")
    configuration_parameters_text.write(str(configuration_parameters))
    configuration_parameters_text.close()

    f_1.savefig("{results_path}/layers_2_vs_fro_norms_ratio.png".format(
        results_path=results_path))
    # f_2.savefig("{results_path}\wprod_v1_product.png".format(
    #     results_path=results_path))
    f_3.savefig(
        "{results_path}/epochs_cost_error.png".format(
            results_path=results_path))
    f_4.savefig("{results_path}/layers_2_vs_nuc_norms_ratio.png".format(
        results_path=results_path))
    f_5.savefig("{results_path}/gradients_norm.png".format(
        results_path=results_path))
    f_1.clf()
    f_2.clf()
    f_3.clf()
    f_4.clf()
    f_5.clf()


plot_single_archeticture_vs_epochs(configuration_parameters, x_train, y_train, x_test, y_test)
print("elapsed time is {elapsed_time} seconds".format(elapsed_time=str(timedelta(seconds=time.time()-start_time))))