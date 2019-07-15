import matplotlib.pyplot as plt
import os
import arrow
import numpy as np

CURRENT_TIME_STR = str(arrow.now().format('YYYY-MM-DD_HH_mm'))


def visualize_errors(errors_dictionary, results_path):
    plot_index = 1
    for key in errors_dictionary.keys():
        f = plt.figure(plot_index)
        plt.scatter(range(len(errors_dictionary[key])), errors_dictionary[key])
        plt.xlabel('Epochs')
        plt.ylabel('{error_type}'.format(error_type=key))
        plot_path = os.path.join(results_path, key,
                                 "{current_time}".format(current_time=CURRENT_TIME_STR))
        if not os.path.isdir(plot_path):
            os.makedirs(plot_path)
        f.savefig(plot_path)
        plot_index += 1


def extract_weights_norms_ratio(weights_values_dictionary):
    weights_2_vs_fro_norm_ratio = {}
    for key in weights_values_dictionary.keys():
        weights_2_vs_fro_norm_ratio[key] = [
            np.linalg.norm(current_weights, ord=2) / np.linalg.norm(current_weights, ord='fro') for current_weights
            in
            weights_values_dictionary[key]]
    return weights_2_vs_fro_norm_ratio


def visualize_weights_norms_ratio(weights_2_vs_fro_norm_ratio, results_path):
    number_of_epochs = len(list(weights_2_vs_fro_norm_ratio.values())[0])
    weights_plots_dict = {}
    f = plt.figure()
    for key in list(weights_2_vs_fro_norm_ratio.keys())[:-1]:
        weights_plots_dict[key] = plt.scatter(range(number_of_epochs), weights_2_vs_fro_norm_ratio[key], label=key)
    plt.legend(tuple(weights_plots_dict.values()), tuple(weights_plots_dict.keys()))
    plt.xlabel('Epochs')
    plt.ylabel('Norm 2 vs frobenius ratio')
    plot_path = os.path.join(results_path, "weights_2_norm_vs_fro_ratio",
                             "{current_time}".format(current_time=CURRENT_TIME_STR))
    f.savefig(plot_path)


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


def visualize_weights_max_min_values_ratio(weights_max_min_ratios_from_initialization, results_path):
    weights_max_min_ratios_from_initialization_plots = {}
    number_of_epochs = len(list(weights_max_min_ratios_from_initialization.values())[0])
    f = plt.figure()
    for key in list(weights_max_min_ratios_from_initialization.keys()):
        weights_max_min_ratios_from_initialization_plots[key] = plt.scatter(range(number_of_epochs),
                                                                            weights_max_min_ratios_from_initialization[
                                                                                key],
                                                                            label="{ratio_type}_ratio of weights change".format(
                                                                                ratio_type=key))

    plt.legend(tuple(weights_max_min_ratios_from_initialization_plots.values()),
               tuple(weights_max_min_ratios_from_initialization_plots.keys()))
    plt.xlabel('Epochs')
    plt.ylabel('Weights Values Max/Min Ratio')
    plot_path = os.path.join(results_path, "weights_values_min_and_max_ratios",
                             "{current_time}".format(current_time=CURRENT_TIME_STR))
    f.savefig(plot_path)
