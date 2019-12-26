import inspect
import os

import copy
import matplotlib.pyplot as plt
from simulator import create_model, utils
from simulator import input_generator
from simulator import train_model
from simulator import metrics_visualizer
from simulator.configurations import simulator_configuration
from simulator.configurations.utils import set_network_architecture_linearly
from simulator.utils import array_of_dictionaries_mean
from simulator.metrics_visualizer import CURRENT_TIME_STR
import arrow
import numpy as np
from simulator.configurations import alons_configuration
from keras.utils import plot_model

CURRENT_TIME_STR = str(arrow.now().format('YYYY-MM-DD_HH_mm'))


def single_model_architecture_one_run(configuration, x_train, y_train, x_test, y_test):
    # network_architecture = configuration['model']['network_architecture']
    model = create_model.create_model(configuration['model']['network_architecture'],
                                      configuration['model']['activation_type'],
                                      configuration['model']['input_dimension'],
                                      configuration['model']['optimizer'],
                                      configuration['model']['loss_type'],
                                      configuration['model']['learning_rate'],
                                      configuration['model']['evaluation_metrics'])
    weights_initialization = configuration['model']['weights_initialization']
    if 'all_layers' in weights_initialization:
        weights_initialization['all_layers'](model)
    if 'single_layers' in weights_initialization:
        for layer_initialization_tuple in weights_initialization['single_layers']:
            layer_initialization_tuple[1](model.layers[layer_initialization_tuple[0]])
    weights, errors = train_model.train_model(model, x_train, y_train, x_test, y_test,
                                              configuration['model'][
                                                  'number_of_epochs'])
    return weights, errors, model


def single_model_architecture_average_run(configuration):
    weights_max_min_ratios_list = []
    weights_norm_ratios_list = []
    errors_dict_list = []
    x_data, y_data = configuration['data']['data_provider'](configuration['data']['sample_size'])

    try:
        leaky_alpha = configuration['model']['activation_type'].alpha
    except (TypeError, AttributeError):
        print("activation is not leaky relu")
    x_train, y_train, x_test, y_test = input_generator.train_and_test_shuffle_split(x_data, y_data,
                                                                                    train_test_split_ratio=
                                                                                    configuration[
                                                                                        'data'][
                                                                                        'train_test_split_ratio'])
    cols = list(configuration['model']['network_architecture'].keys())
    rows = ['Index {i}'.format(i=index) for index in range(configuration['model']['number_of_runs'])]
    weights_figure, axes = plt.subplots(nrows=len(rows), ncols=len(cols))
    axes = np.array(axes).reshape(len(rows), len(cols))
    for ax, col in zip(axes[0], cols):
        ax.set_title(col)

    for ax, row in zip(axes[:, 0], rows):
        ax.set_ylabel(row, rotation=0, size='large')
    weights_figure.tight_layout()
    weights_figure.suptitle('Weights')
    # axes[0, 0].imshow(weights['layer_0'][0], aspect='auto')
    # axes[0, 1].imshow(weights['layer_1'][0], aspect='auto')
    for iteration_index in range(configuration['model']['number_of_runs']):
        weights, errors, model = single_model_architecture_one_run(configuration, x_train, y_train, x_test,
                                                                   y_test)
        for layer_index in range(len(cols)):
            im = axes[iteration_index, layer_index].imshow(weights[cols[layer_index]][0], aspect='auto', vmin=0, vmax=1)
        weights_figure.subplots_adjust(right=0.8)
        cbar_ax = weights_figure.add_axes([0.85, 0.15, 0.05, 0.7])
        weights_figure.colorbar(im, cax=cbar_ax)
        _, weights_max_min_ratios = metrics_visualizer.extract_weights_values_ratio(weights)
        weights_max_min_ratios_list.append(weights_max_min_ratios)
        weights_2_vs_fro_norm_ratio = metrics_visualizer.extract_weights_norms_ratio(weights)
        weights_norm_ratios_list.append(weights_2_vs_fro_norm_ratio)
        errors_dict_list.append(errors)
    # TODO: add support for different sizes of arrays due to zero updates batches! otherwise it won't work
    weights_max_min_ratios_avg = array_of_dictionaries_mean(weights_max_min_ratios_list)
    weights_norm_ratios_avg = array_of_dictionaries_mean(weights_norm_ratios_list)
    errors_avg = dict()
    errors_avg['batches'] = array_of_dictionaries_mean([errors_dict['batches'] for errors_dict in errors_dict_list])
    errors_avg['epochs'] = array_of_dictionaries_mean([errors_dict['epochs'] for errors_dict in errors_dict_list])
    return weights_max_min_ratios_avg, weights_norm_ratios_avg, errors_avg, model, weights_figure


def visualize_single_model_architecture_average_run(configuration):
    weights_max_min_ratios_avg, weights_norm_ratios_avg, errors_avg, model, weights_figures = single_model_architecture_average_run(
        configuration)
    model_plot_path = os.path.join(configuration['results']['results_path'],
                                   "model_architecture")
    if not os.path.exists(model_plot_path):
        os.makedirs(model_plot_path)
    plot_model(model, to_file=os.path.join(model_plot_path, "{time_str}.png".format(time_str=CURRENT_TIME_STR)),
               show_shapes=True,
               show_layer_names=True)
    weights_max_min_figure = metrics_visualizer.visualize_weights_max_min_values_ratio(weights_max_min_ratios_avg,
                                                                                       desired_layers_visualizations_indices_list=
                                                                                       configuration['results'][
                                                                                           'weights'].get(
                                                                                           'max_min_ratios_desired_graph_indices'))
    metrics_visualizer.save_figure(weights_max_min_figure,
                                   os.path.join(configuration['results']['results_path'],
                                                "weights_values_min_and_max_ratios"), CURRENT_TIME_STR)
    metrics_visualizer.save_figure(weights_figures, os.path.join(configuration['results']['results_path'],
                                                                 "weights"), CURRENT_TIME_STR)
    metrics_visualizer.visualize_errors(errors_avg, configuration['results']['results_path'])
    weights_norm_figure = metrics_visualizer.visualize_weights_norms_ratio(weights_norm_ratios_avg,
                                                                           desired_layers_visualizations_indices_list=
                                                                           configuration['results']['weights'].get(
                                                                               'norms_ratios_desired_graph_indices'),
                                                                           label_title_suffix="")
    metrics_visualizer.save_figure(weights_norm_figure,
                                   os.path.join(configuration['results']['results_path'],
                                                'weights_2_norm_vs_fro_ratio'), CURRENT_TIME_STR)
    metrics_visualizer.save_configuration(configuration, os.path.join(configuration['results']['results_path'],
                                                                      "configuration"))


def visualize_change_learning_rate_and_architecture_average_run(configuration, learning_rate_list,
                                                                network_architecture_list):
    weights_max_min_figure = plt.figure()
    experiment_path = os.path.join(configuration['results']['results_path'],
                                   "weights_values_min_and_max_ratios_vs_learning_rate")
    for learning_rate, network_architecture in zip(learning_rate_list, network_architecture_list):
        current_configuration = copy.copy(configuration)
        current_configuration['model']['learning_rate'] = learning_rate
        current_configuration['model']['network_architecture'] = network_architecture
        weights_max_min_ratios_avg, weights_norm_ratios_avg, errors_avg = single_model_architecture_average_run(
            current_configuration)
        weights_max_min_figure = metrics_visualizer.visualize_weights_max_min_values_ratio(weights_max_min_ratios_avg,
                                                                                           desired_layers_visualizations_indices_list=
                                                                                           current_configuration[
                                                                                               'results'][
                                                                                               'weights'].get(
                                                                                               'max_min_ratios_desired_graph_indices'),
                                                                                           label_title_suffix="_lr_{lr}_layers_count_{number_of_layers}_max_neurons_{first_layer_weights}".format(
                                                                                               lr=current_configuration[
                                                                                                   'model'][
                                                                                                   'learning_rate'],
                                                                                               first_layer_weights=
                                                                                               current_configuration[
                                                                                                   'model'][
                                                                                                   'network_architecture'][
                                                                                                   'layer_1'],
                                                                                               number_of_layers=
                                                                                               len(
                                                                                                   current_configuration[
                                                                                                       'model'][
                                                                                                       'network_architecture'].keys())),
                                                                                           figure_handle=weights_max_min_figure)
        metrics_visualizer.visualize_errors(errors_avg, os.path.join(experiment_path,
                                                                     "learning_rate_{lr}_max_neurons_{first_layer_neurons}".format(
                                                                         lr=current_configuration[
                                                                             'model'][
                                                                             'learning_rate'], first_layer_neurons=
                                                                         network_architecture[
                                                                             'layer_1'])))
    metrics_visualizer.save_figure(weights_max_min_figure,
                                   experiment_path)
    configuration_for_save = copy.copy(configuration)
    configuration_for_save['run_type'] = 'weights_ratio_vs_learning_rate'
    configuration_for_save['model'].pop('learning_rate')
    configuration_for_save['model']['learning_rate'] = learning_rate_list
    configuration_for_save['model'].pop('network_architecture')
    configuration_for_save['results'].pop('results_path')
    configuration_for_save['results']['results_path'] = experiment_path
    configuration_for_save['model']['network_architecture'] = network_architecture_list
    metrics_visualizer.save_configuration(configuration_for_save, os.path.join(experiment_path, "configuration"))


def visualize_change_weights_initialization_and_architecture_average_run(configuration, weights_initializers_list,
                                                                         network_architecture_list):
    weights_max_min_figure = plt.figure()
    experiment_path = os.path.join(configuration['results']['results_path'],
                                   "weights_values_min_and_max_ratios_vs_learning_rate",
                                   CURRENT_TIME_STR)
    if not os.path.exists(experiment_path):
        os.makedirs(experiment_path)
    for weights_initializer, network_architecture in zip(weights_initializers_list, network_architecture_list):
        current_configuration = copy.copy(configuration)
        current_configuration['model']['weights_initialization']['all_layers'] = weights_initializer
        current_configuration['model']['network_architecture'] = network_architecture
        weights_max_min_ratios_avg, weights_norm_ratios_avg, errors_avg = single_model_architecture_average_run(
            current_configuration)
        initialization_parameters = \
            utils.dict_to_str(
                inspect.signature(current_configuration['model']['weights_initialization']['all_layers']).parameters[
                    'initialization_method'].default.keywords)
        weights_max_min_figure = metrics_visualizer.visualize_weights_max_min_values_ratio(weights_max_min_ratios_avg,
                                                                                           non_zero_updates_upper_bound=M_K,
                                                                                           desired_layers_visualizations_indices_list=
                                                                                           current_configuration[
                                                                                               'results'][
                                                                                               'weights'].get(
                                                                                               'max_min_ratios_desired_graph_indices'),
                                                                                           label_title_suffix="_initialization_{init}_layers_count_{number_of_layers}_max_neurons_{first_layer_weights}".format(
                                                                                               init=
                                                                                               initialization_parameters,
                                                                                               first_layer_weights=
                                                                                               current_configuration[
                                                                                                   'model'][
                                                                                                   'network_architecture'][
                                                                                                   'layer_1'],
                                                                                               number_of_layers=
                                                                                               len(
                                                                                                   current_configuration[
                                                                                                       'model'][
                                                                                                       'network_architecture'].keys())),
                                                                                           figure_handle=weights_max_min_figure)
        metrics_visualizer.visualize_errors(errors_avg, os.path.join(experiment_path,
                                                                     "initialization_parameters_{init}_max_neurons_{first_layer_neurons}".format(
                                                                         init=
                                                                         initialization_parameters, first_layer_neurons=
                                                                         network_architecture[
                                                                             'layer_1'])))
    metrics_visualizer.save_figure(weights_max_min_figure,
                                   experiment_path, "weights_min_max_ratio_vs_updates")
    configuration_for_save = copy.copy(configuration)
    configuration_for_save['run_type'] = 'weights_ratio_vs_learning_rate'
    configuration_for_save['model']['weights_initialization'].pop('all_layers')
    configuration_for_save['model']['weights_initialization']['all_layers'] = weights_initializers_list
    configuration_for_save['model'].pop('network_architecture')
    configuration_for_save['results'].pop('results_path')
    configuration_for_save['results']['results_path'] = experiment_path
    configuration_for_save['model']['network_architecture'] = network_architecture_list
    metrics_visualizer.save_configuration(configuration_for_save, os.path.join(experiment_path, "configuration"))
