import matplotlib.pyplot as plt
import os
import arrow
import numpy as np

from simulator.configurations.linear_regression_configuration import CURRENT_TIME_STR


def visualize_errors(errors_dictionary, results_path):
    batch_errors = errors_dictionary['batches']
    epoch_errors = errors_dictionary['epochs']
    batch_errors_figure = plt.figure()
    for key in batch_errors.keys():
        plt.plot(range(len(batch_errors[key][0])), list(batch_errors[key][0]), label='{key}'.format(key=key))
        plt.xlabel('Batches')
        plt.ylabel('Error')
        plt.legend()
        save_figure(batch_errors_figure, plot_path=r"{result_path}\batch_errors".format(result_path=results_path),
                    plot_name=CURRENT_TIME_STR)

    epoch_errors_figure = plt.figure()
    for key in epoch_errors.keys():
        plt.plot(range(len(epoch_errors[key][0])), list(epoch_errors[key][0]), label='{key}'.format(key=key))
        plt.xlabel('Epochs')
        plt.ylabel('Error')
        plt.legend()
        save_figure(epoch_errors_figure, plot_path=r"{result_path}\epoch_errors".format(result_path=results_path),
                    plot_name=CURRENT_TIME_STR)


def save_figure(figure_handle, plot_path, plot_name):
    if not os.path.exists(plot_path):
        os.makedirs(plot_path)
    figure_handle.savefig(r"{results_path}\{plot_name}.png".format(
        results_path=plot_path, plot_name=plot_name), bbox_inches='tight')


def save_configuration(configuration, configuration_path):
    if not os.path.exists(configuration_path):
        os.makedirs(configuration_path)
    configuration_parameters_text = open(
        r"{results_path}\{current_date}.txt".format(
            results_path=configuration_path, current_date=CURRENT_TIME_STR), "w+")
    configuration_parameters_text.write(str(configuration))
    configuration_parameters_text.close()


def extract_weights_norms_ratio(weights_values_dictionary):
    weights_2_vs_fro_norm_ratio = {}
    for key in weights_values_dictionary.keys():
        weights_2_vs_fro_norm_ratio[key] = [
            np.linalg.norm(current_weights, ord=2) / np.linalg.norm(current_weights, ord='fro') for current_weights
            in
            weights_values_dictionary[key]]
    return weights_2_vs_fro_norm_ratio


# TODO: create a function which extracts the perceptron ratio for neural networks
# to see wether the perceptron like proof can be extended
def extract_perceptron_ratio(weights_through_time, w_star):
    preceptron_ratio = [np.dot(w_star, weight) / (np.linalg.norm(weight) * np.linalg.norm(w_star)) for weight in
                        weights_through_time]
    return preceptron_ratio

def visualize_perceptron_ratio(perceptron_ratio,figure_handle=None):
    if not figure_handle:
        figure_handle = plt.figure()
    axis = figure_handle.gca()
    axis.scatter(range(len(perceptron_ratio)),
                 perceptron_ratio)
    axis.set_xlabel('Batches')
    axis.set_ylabel('Cauchy Shwartz ratio')
    return figure_handle
# TODO:Extend it to a non linear case - need to find a way to generate the prediction using the weights values at time t
#  or equivalently save the model after each batch
def extract_training_points_regression_loss(weights, x_train, y_train, dimension):
    def linear_prediction(weights, features_samples, dimension):
        training_points_predictions = np.dot(np.array(weights).reshape(len(weights), dimension),
                                             np.transpose(features_samples))
        return training_points_predictions

    training_predictions = linear_prediction(weights, x_train, dimension)
    error_trough_training = np.tile(np.reshape(y_train, (1, training_predictions.shape[1])),
                                    (training_predictions.shape[0], 1)) - training_predictions
    return error_trough_training


def visualize_training_points_regression_loss(points_errors_through_training, abs_val_flag=True, figure_handle=None):
    if not figure_handle:
        figure_handle = plt.figure()
    axis = figure_handle.gca()
    if abs_val_flag:
        for train_point_index in range(points_errors_through_training.shape[1]):
            axis.scatter(range(len(points_errors_through_training[:, train_point_index])),
                     np.abs(points_errors_through_training[:, train_point_index]),
                     label='point_index_{index}'.format(index=train_point_index))
    else:
        for train_point_index in range(points_errors_through_training.shape[1]):
            axis.scatter(range(len(points_errors_through_training[:, train_point_index])),
                     points_errors_through_training[:, train_point_index],
                     label='point_index_{index}'.format(index=train_point_index))
    axis.legend()
    axis.set_xlabel('Batches')
    axis.set_ylabel('Point_error')
    return figure_handle

def visualize_weights_norms_ratio(weights_2_vs_fro_norm_ratio,
                                  desired_layers_visualizations_indices_list=None, figure_handle=None,
                                  label_title_suffix=""):
    number_of_epochs = len(list(weights_2_vs_fro_norm_ratio.values())[0])
    weights_plots_dict = {}
    if not figure_handle:
        figure_handle = plt.figure()
    axis = figure_handle.gca()
    if desired_layers_visualizations_indices_list:
        for key in np.array(list(weights_2_vs_fro_norm_ratio.keys()))[desired_layers_visualizations_indices_list]:
            weights_plots_dict[key] = axis.scatter(range(weights_2_vs_fro_norm_ratio[key].shape[1]),
                                                   weights_2_vs_fro_norm_ratio[key],
                                                   label="{key}{label_suffix}".format(key=key,
                                                                                      label_suffix=label_title_suffix))
    else:
        for key in np.array(list(weights_2_vs_fro_norm_ratio.keys()))[:-1]:
            weights_plots_dict[key] = axis.scatter(range(weights_2_vs_fro_norm_ratio[key].shape[1]),
                                                   weights_2_vs_fro_norm_ratio[key],
                                                   label="{key}{label_suffix}".format(key=key,
                                                                                      label_suffix=label_title_suffix))
    axis.legend()
    axis.set_xlabel('Epochs')
    axis.set_ylabel('Norm 2 vs frobenius ratio')
    return figure_handle


def extract_weights_values_ratio(weights_values_dictionary):
    weights_values_ratios_from_initialization = {}
    weights_max_min_ratios_from_initialization = {}
    division_by_zero = np.seterr(divide='raise')
    ignore_state = np.seterr(**division_by_zero)
    for key in weights_values_dictionary.keys():
        weights_values_ratios_from_initialization[key] = [
            np.divide(weights_values_dictionary[key][index], weights_values_dictionary[key][0]) for index in
            range(len(weights_values_dictionary[key]))]
    for key in weights_values_dictionary.keys():
        weights_max_min_ratios_from_initialization[
            "{layer_name}_{type}".format(layer_name=key, type="max")] = [
            np.amax(weights_values_ratio) for weights_values_ratio in weights_values_ratios_from_initialization[key]]
        weights_max_min_ratios_from_initialization[
            "{layer_name}_{type}".format(layer_name=key, type="min")] = [
            np.amin(weights_values_ratio) for weights_values_ratio in weights_values_ratios_from_initialization[key]]
    return weights_values_ratios_from_initialization, weights_max_min_ratios_from_initialization


def visualize_weights_max_min_values_ratio(weights_max_min_ratios_from_initialization,
                                           non_zero_updates_upper_bound=None,
                                           desired_layers_visualizations_indices_list=None, figure_handle=None,
                                           label_title_suffix="", visualization_substring="m"):
    weights_max_min_ratios_from_initialization_plots = {}
    number_of_samples = len(list(weights_max_min_ratios_from_initialization.values())[0][0])
    if not figure_handle:
        figure_handle = plt.figure()
    axis = figure_handle.gca()
    if desired_layers_visualizations_indices_list:
        for key in np.array(list(weights_max_min_ratios_from_initialization.keys()))[
            desired_layers_visualizations_indices_list]:
            if visualization_substring in key:
                weights_max_min_ratios_from_initialization_plots[key] = axis.scatter(range(number_of_samples),
                                                                                     weights_max_min_ratios_from_initialization[
                                                                                         key],
                                                                                     label="{ratio_type}{suffix}".format(
                                                                                         ratio_type=key,
                                                                                         suffix=label_title_suffix,
                                                                                         non_zero_updates=str(
                                                                                             non_zero_updates_upper_bound)))
    else:
        for key in np.array(list(weights_max_min_ratios_from_initialization.keys())):
            if visualization_substring in key:
                weights_max_min_ratios_from_initialization_plots[key] = axis.scatter(range(number_of_samples),
                                                                                     weights_max_min_ratios_from_initialization[
                                                                                         key],
                                                                                     label="{ratio_type}{suffix}".format(
                                                                                         ratio_type=key,
                                                                                         suffix=label_title_suffix))
    axis.legend()
    axis.set_xlabel('Number of batches')
    axis.set_ylabel('Weights Values Max/Min Ratio')
    # axis.set_title("non_zero_updates_upper_bound is {non_zero_updates}".format(non_zero_updates=str(non_zero_updates_upper_bound)))
    return figure_handle
