import matplotlib.pyplot as plt
import os
import arrow
import numpy as np

CURRENT_TIME_STR = str(arrow.now().format('YYYY-MM-DD_HH_mm'))


def visualize_errors(errors_dictionary, results_path):
    for key in errors_dictionary.keys():
        f = plt.figure()
        plt.scatter(range(len(errors_dictionary[key])), errors_dictionary[key])
        plt.xlabel('Epochs')
        plt.ylabel('{error_type}'.format(error_type=key))
        plot_path = os.path.join(results_path, key)
        save_figure(f, plot_path)


def save_figure(figure_handle, plot_path):
    if not os.path.isdir(plot_path):
        os.makedirs(plot_path)
    figure_handle.savefig(r"{results_path}\{current_date}.png".format(
        results_path=plot_path, current_date=CURRENT_TIME_STR))


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
            weights_plots_dict[key] = axis.scatter(range(number_of_epochs), weights_2_vs_fro_norm_ratio[key],
                                                   label="{key}{label_suffix}".format(key=key,
                                                                                      label_suffix=label_title_suffix))
    else:
        for key in np.array(list(weights_2_vs_fro_norm_ratio.keys()))[:-1]:
            weights_plots_dict[key] = axis.scatter(range(number_of_epochs), weights_2_vs_fro_norm_ratio[key],
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
                                           desired_layers_visualizations_indices_list=None, figure_handle=None,
                                           label_title_suffix="",visualization_substring="min"):
    weights_max_min_ratios_from_initialization_plots = {}
    number_of_samples = len(list(weights_max_min_ratios_from_initialization.values())[0])
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
                                                                                         suffix=label_title_suffix))
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
    return figure_handle
