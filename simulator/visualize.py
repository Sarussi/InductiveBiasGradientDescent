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


def visualize_weights_norms_ratio(weights_dictionary, results_path):
    weights_2_vs_fro_norm_ratio = {}
    number_of_epochs = len(list(weights_dictionary.values())[0])
    for key in weights_dictionary.keys():
        weights_2_vs_fro_norm_ratio[key] = [
            np.linalg.norm(current_weights, ord=2) / np.linalg.norm(current_weights, ord='fro') for current_weights
            in
            weights_dictionary[key]]
    weights_plots_dict = {}
    f = plt.figure()
    for key in list(weights_2_vs_fro_norm_ratio.keys())[:-1]:
        weights_plots_dict[key] = plt.scatter(range(number_of_epochs), weights_2_vs_fro_norm_ratio[key], label=key)
    plt.legend(tuple(weights_plots_dict.values()), tuple(weights_plots_dict.keys()))
    plt.xlabel('Epochs')
    plt.ylabel('Norm 2 vs frobenius ratio')
    plot_path = os.path.join(results_path, "weights_2_norm_vs_fro_ratio",
                             "{current_time}".format(current_time=CURRENT_TIME_STR))
    if not os.path.isdir(plot_path):
        os.makedirs(plot_path)
    f.savefig(plot_path)


def visualize_weights_values_ratio(weights_dictionary, results_path):
    weights_values_ratios_from_initialization = {}
    weights_ratios_from_initialization_plots = {}
    number_of_epochs = len(list(weights_dictionary.values())[0])
    division_by_zero = np.seterr(divide='raise')
    ignore_state = np.seterr(**division_by_zero)
    for key in weights_dictionary.keys():
        weights_values_ratios_from_initialization[key] = [
            np.divide(weights_dictionary[key][index], weights_dictionary[key][0]) for index in
            range(len(weights_dictionary[key]))]
    f = plt.figure()

    for key in list(weights_dictionary.keys())[1:]:
        weights_ratios_from_initialization_plots[
            "{layer_name}_{type}".format(layer_name=key, type="max")] = plt.scatter(range(number_of_epochs), [
            np.amax(weights_values_ratio) for weights_values_ratio in weights_values_ratios_from_initialization[key]],
                                                                                    label="maximum ratio of weights change for {layer_name}".format(
                                                                                        layer_name=key))
        weights_ratios_from_initialization_plots[
            "{layer_name}_{type}".format(layer_name=key, type="min")] = plt.scatter(range(number_of_epochs), [
            np.amin(weights_values_ratio) for weights_values_ratio in weights_values_ratios_from_initialization[key]],
                                                                                    label="minimum ratio of weights change for {layer_name}".format(
                                                                                        layer_name=key))
    plt.legend(tuple(weights_ratios_from_initialization_plots.values()),
               tuple(weights_ratios_from_initialization_plots.keys()))
    plt.xlabel('Epochs')
    plt.ylabel('Weights Values Max/Min Ratio')
    plot_path = os.path.join(results_path, "weights_values_min_and_max_ratios",
                             "{current_time}".format(current_time=CURRENT_TIME_STR))
    if not os.path.isdir(plot_path):
        os.makedirs(plot_path)
    f.savefig(plot_path)
