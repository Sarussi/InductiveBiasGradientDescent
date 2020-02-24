import inspect
import os

import copy
from itertools import chain
# import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
import create_model, utils
import input_generator
import train_model
import metrics_visualizer
from  configurations import simulator_configuration
from  configurations.linear_regression_configuration import CURRENT_TIME_STR
from  configurations.utils import set_network_architecture_linearly
from  utils import array_of_dictionaries_mean
import arrow
import numpy as np
from  configurations import alons_configuration
from keras.utils import plot_model
import keras as K


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
        # if configuration['data']['sample_dimension'] == 2 and weights_initialization['near_max_margin'] is True:
        #     max_margin_seperator_of_train, max_margin_intercept, svm_classifier = input_generator.svm_normalized_max_margin(
        #         x_train, y_train)
        #     max_margin_seperator_of_train = max_margin_seperator_of_train / np.linalg.norm(
        #         max_margin_seperator_of_train)
        #     weights_initialization['all_layers'](model, max_margin_seperator_of_train)
        # else:
            weights_initialization['all_layers'](model)
    aa=1
    if 'single_layers' in weights_initialization:
        for layer_initialization_tuple in weights_initialization['single_layers']:
            layer_initialization_tuple[1](model.layers[layer_initialization_tuple[0]])
    aa=1
    weights, errors = train_model.train_model(model, x_train, y_train, x_test, y_test,
                                              configuration['model'][
                                                  'number_of_epochs'], configuration['model']['batch_size'],
                                              configuration['model']['history_flags'])
    return weights, errors, model


# TODO: change to fit new weights format (dictionary of batches and epochs)
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
    layer_names = list(configuration['model']['network_architecture'].keys())
    cols = list(chain.from_iterable((col + "_init", col + "_conve") for col in
                                    layer_names))
    rows = ['run_{i}'.format(i=index) for index in range(configuration['model']['number_of_runs'])]
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
        layer_index = 0
        for layer in layer_names:
            axes[iteration_index, layer_index].imshow(weights[layer][0], aspect='auto', vmin=0, vmax=1)
            im = axes[iteration_index, layer_index + 1].imshow(weights[layer][-1], aspect='auto', vmin=0,
                                                               vmax=1)
            layer_index += 2

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


def visualize_matus_proof_potential(configuration):
    experiment_path = os.path.join(configuration['results']['results_path'],
                                   "matus_proof_potential")
    if not os.path.exists(experiment_path):
        os.makedirs(experiment_path)
    x_data, y_data, w_star_perceptron_estimate = configuration['data']['data_provider'](
        configuration['data']['sample_size'])
    if 'w_star' not in configuration['data'].keys():
        configuration['data']['w_star'] = w_star_perceptron_estimate
    aa=1
    w_star_perceptron_estimate /= np.linalg.norm(w_star_perceptron_estimate)
    x_train, y_train, x_test, y_test = input_generator.train_and_test_shuffle_split(x_data, y_data,
                                                                                    train_test_split_ratio=
                                                                                    configuration[
                                                                                        'data'][
                                                                                        'train_test_split_ratio'])
    max_margin_seperator_of_train, max_margin_intercept, svm_classifier = input_generator.svm_normalized_max_margin(
        x_train, y_train)
    max_margin_seperator_of_train = max_margin_seperator_of_train / np.linalg.norm(max_margin_seperator_of_train)
    weights, errors, model = single_model_architecture_one_run(configuration, x_train, y_train, x_test, y_test)
    weights_resolution_flags = configuration['model']['history_flags']
    for key in weights_resolution_flags.keys():
        if weights_resolution_flags[key] and key == 'epochs':
            metrics_visualizer.visualize_errors(errors, configuration['results']['results_path'])
            weights_2_vs_fro_norm_ratio = metrics_visualizer.extract_weights_norms_ratio(weights[key])
            weights_norm_figure = metrics_visualizer.visualize_weights_norms_ratio(weights_2_vs_fro_norm_ratio, [0])
            metrics_visualizer.save_figure(weights_norm_figure,
                                           os.path.join(experiment_path, key,
                                                        'weights_2_norm_vs_fro_ratio'), CURRENT_TIME_STR)
            neurons_alignments_to_max_margin, weights_neurons_direction = metrics_visualizer.extract_neurons_alignment(
                weights[key],
                max_margin_seperator_of_train)
            neurons_alignments_figure = metrics_visualizer.visualize_randomly_sampled_rows_from_array(
                np.transpose(neurons_alignments_to_max_margin), y_axis_title='neurons_direction_from_max_margin_norm',
                sample_points_size=max(1, int(neurons_alignments_to_max_margin.shape[1] / 5)))
            metrics_visualizer.save_figure(neurons_alignments_figure,
                                           os.path.join(experiment_path, key, "neurons_alignment"), CURRENT_TIME_STR)
            prediction_landscape_path = os.path.join(experiment_path, key,
                                                     'predictions_landscape_through_time')
            if not os.path.exists(prediction_landscape_path):
                os.makedirs(prediction_landscape_path)
            if configuration['data']['sample_dimension'] == 2:
                # TODO: make sample points max(1,number_of_neurons/10)
                neurons_directions_figure = metrics_visualizer.visualize_neurons_directions(weights_neurons_direction,
                                                                                            max_margin_seperator_of_train,
                                                                                            x_train, y_train,
                                                                                            configuration['data'][
                                                                                                'sample_dimension'],
                                                                                            configuration['data'][
                                                                                                'first_label'],
                                                                                            configuration['data'][
                                                                                                'second_label'],
                                                                                            sample_points_size=max(1,
                                                                                                                   int(
                                                                                                                       configuration[
                                                                                                                           'model'][
                                                                                                                           'number_of_neurons'])))
                metrics_visualizer.save_figure(neurons_directions_figure,
                                               os.path.join(experiment_path, key, "neurons_directions"),
                                               CURRENT_TIME_STR)
                working_directory = os.getcwd()
                x_positive = x_train[np.where(y_train == configuration['data']['first_label'])[0], :]
                x_negative = x_train[np.where(y_train == configuration['data']['second_label'])[0], :]
                metrics_visualizer.extract_prediction_landscape_through_time(weights[key], model, svm_classifier,
                                                                             x_positive,
                                                                             x_negative,
                                                                             configuration['data']['sample_dimension'],
                                                                             prediction_landscape_path,
                                                                             CURRENT_TIME_STR, mesh_grid_step=
                                                                             configuration['tests'][
                                                                                 'prediction_landscape'][
                                                                                 'mesh_grid_step'])

                os.chdir(working_directory)
        if weights_resolution_flags[key] and key == 'batches':
            neurons_alignments_to_max_margin, weights_neurons_direction = metrics_visualizer.extract_neurons_alignment(
                weights[key],
                max_margin_seperator_of_train)
            neurons_alignments_figure = metrics_visualizer.visualize_randomly_sampled_rows_from_array(
                np.transpose(neurons_alignments_to_max_margin),
                y_axis_title='neurons_direction_from_max_margin_norm',
                sample_points_size=max(1, int(neurons_alignments_to_max_margin.shape[1] / 5)))
            metrics_visualizer.save_figure(neurons_alignments_figure,
                                           os.path.join(experiment_path, key, "neurons_alignment"),
                                           CURRENT_TIME_STR)
            prediction_landscape_path = os.path.join(experiment_path, key,
                                                     'predictions_landscape_through_time')
            if not os.path.exists(prediction_landscape_path):
                os.makedirs(prediction_landscape_path)
            if configuration['data']['sample_dimension'] == 2:
                neurons_directions_figure = metrics_visualizer.visualize_neurons_directions(weights_neurons_direction,
                                                                                            max_margin_seperator_of_train,
                                                                                            x_train, y_train,
                                                                                            configuration['data'][
                                                                                                'sample_dimension'],
                                                                                            configuration['data'][
                                                                                                'first_label'],
                                                                                            configuration['data'][
                                                                                                'second_label'],
                                                                                            sample_points_size=max(1,
                                                                                                                   int(
                                                                                                                       configuration[
                                                                                                                           'model'][
                                                                                                                           'number_of_neurons'])))
                metrics_visualizer.save_figure(neurons_directions_figure,
                                               os.path.join(experiment_path, key, "neurons_directions"),
                                               CURRENT_TIME_STR)
                working_directory = os.getcwd()
                x_positive = x_train[np.where(y_train == configuration['data']['first_label'])[0], :]
                x_negative = x_train[np.where(y_train == configuration['data']['second_label'])[0], :]
                metrics_visualizer.extract_prediction_landscape_through_time(weights[key], model, svm_classifier,
                                                                             x_positive,
                                                                             x_negative,
                                                                             configuration['data']['sample_dimension'],
                                                                             prediction_landscape_path,
                                                                             CURRENT_TIME_STR, mesh_grid_step=
                                                                             configuration['tests'][
                                                                                 'prediction_landscape'][
                                                                                 'mesh_grid_step'])

                os.chdir(working_directory)

    configuration_for_save = copy.copy(configuration)
    metrics_visualizer.save_configuration(configuration_for_save,
                                          os.path.join(experiment_path, "configuration"))


def visualize_perceptron_proof_potential(configuration):
    experiment_path = os.path.join(configuration['results']['results_path'],
                                   "perceptron_proof_potential")
    if not os.path.exists(experiment_path):
        os.makedirs(experiment_path)
    x_data, y_data, separation_boundary = configuration['data']['data_provider'](configuration['data']['sample_size'])
    if 'w_star' not in configuration['data'].keys():
        configuration['data']['w_star'] = separation_boundary
    aa = 1
    x_train, y_train, x_test, y_test = input_generator.train_and_test_shuffle_split(x_data, y_data,
                                                                                    train_test_split_ratio=
                                                                                    configuration[
                                                                                        'data'][
                                                                                        'train_test_split_ratio'])
    aa = 1
    weights, errors, model = single_model_architecture_one_run(configuration, x_train, y_train, x_test, y_test)
    # TODO: understand how to clone model using leaky relu
    # model_copy = K.models.clone_model(model)
    weights_resolution_flags = configuration['model']['history_flags']
    for key in weights_resolution_flags.keys():
        if weights_resolution_flags[key]:
            metrics_visualizer.visualize_errors(errors, configuration['results']['results_path'])
            perceptron_ratio_time = metrics_visualizer.extract_perceptron_ratio(weights[key]['layer_0'],
                                                                                separation_boundary)
            perceptron_ratio_fig = metrics_visualizer.visualize_perceptron_ratio(perceptron_ratio_time)
            training_points_loss = metrics_visualizer.extract_training_points_loss(weights[key], x_train,
                                                                                   y_train, configuration['data'][
                                                                                       'sample_dimension'],
                                                                                   model,
                                                                                   configuration['model'][
                                                                                       'point_wise_loss'])
            training_points_loss_fig = metrics_visualizer.visualize_randomly_sampled_rows_from_array(
                training_points_loss)
            metrics_visualizer.save_figure(perceptron_ratio_fig,
                                           os.path.join(experiment_path, key, "perceptron_ratio"), CURRENT_TIME_STR)
            metrics_visualizer.save_figure(training_points_loss_fig,
                                           os.path.join(experiment_path, key, "training_points_errors"),
                                           CURRENT_TIME_STR)
            configuration_for_save = copy.copy(configuration)
            metrics_visualizer.save_configuration(configuration_for_save,
                                                  os.path.join(experiment_path, key, "configuration"))


# TODO: create a save to the figs based on the average run visualization
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
