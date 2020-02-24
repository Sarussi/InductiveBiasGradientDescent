import glob

import cv2
import matplotlib
import matplotlib.pyplot as plt
import os
import arrow
import numpy as np
from sklearn.preprocessing import normalize

from  configurations.alons_configuration import CURRENT_TIME_STR

# TODO: find a way to bypass the need to manually change imports for each different configuration
# from  configurations.linear_regression_configuration import CURRENT_TIME_STR
from utils import natural_keys_string_sort


def visualize_errors(errors_dictionary, results_path):
    batch_errors = errors_dictionary['batches']
    epoch_errors = errors_dictionary['epochs']
    batch_errors_figure = plt.figure()
    for key in batch_errors.keys():
        plt.plot(range(len(batch_errors[key])), list(batch_errors[key]), label='{key}'.format(key=key))
        plt.xlabel('Batches')
        plt.ylabel('Error')
        plt.legend()
        save_figure(batch_errors_figure, plot_path=r"{result_path}\batch_errors".format(result_path=results_path),
                    plot_name=CURRENT_TIME_STR)

    epoch_errors_figure = plt.figure()
    for key in epoch_errors.keys():
        plt.plot(range(len(epoch_errors[key])), list(epoch_errors[key]), label='{key}'.format(key=key))
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
    # type(weights_layers_between_batches.get(list(weights_layers_between_batches.keys())[0])[0]) is dict
    if type(weights_values_dictionary.get(list(weights_values_dictionary.keys())[0])) is dict:
        for key in weights_values_dictionary.keys():
            weights_2_vs_fro_norm_ratio[key] = [
                np.linalg.norm(current_weights, ord=2) / np.linalg.norm(current_weights, ord='fro') for current_weights
                in
                weights_values_dictionary[key]['weights']]
    else:
        for key in weights_values_dictionary.keys():
            weights_2_vs_fro_norm_ratio[key] = [
                np.linalg.norm(current_weights, ord=2) / np.linalg.norm(current_weights, ord='fro') for
                current_weights
                in
                weights_values_dictionary[key]]
    return weights_2_vs_fro_norm_ratio


# TODO: create a function which extracts the perceptron ratio for neural networks
# to see wether the perceptron like proof can be extended
def extract_perceptron_ratio(weights_through_time, w_star):
    aa = 1
    if weights_through_time[0].shape == w_star.shape:
        preceptron_ratio = [np.dot(w_star, weight) / (np.linalg.norm(weight) * np.linalg.norm(w_star)) for weight in
                            weights_through_time]
    else:
        w_star_matrix_anti_sym_2_layer = np.concatenate((np.tile(w_star,
                                                                 (1, int(weights_through_time[0].shape[1] / 2))),
                                                         np.tile(-w_star,
                                                                 (1, int(weights_through_time[0].shape[1] / 2)))),
                                                        axis=1).flatten()
        preceptron_ratio = [np.dot(weight.flatten(order='F'), w_star_matrix_anti_sym_2_layer) / (
            np.linalg.norm(weight) * np.linalg.norm(w_star_matrix_anti_sym_2_layer)) for weight in weights_through_time]
        # preceptron_ratio = [np.norm(np.matmul(first_layer_weight,w_star_matrix_anti_sym_2_layer),]
    return preceptron_ratio


def visualize_perceptron_ratio(perceptron_ratio, figure_handle=None):
    if not figure_handle:
        figure_handle = plt.figure()
    axis = figure_handle.gca()
    axis.plot(range(len(perceptron_ratio)),
              perceptron_ratio)
    axis.set_xlabel('Train Index')
    axis.set_ylabel('Cauchy Shwartz ratio')
    return figure_handle


# TODO:Extend it to a non linear case - need to find a way to generate the prediction using the weights values at time t
#  or equivalently save the model after each batch
def predict_through_time(weights_through_time, features_points, model):
    layer_keys = list(weights_through_time.keys())
    aa=1
    if type(weights_through_time.get(list(weights_through_time.keys())[0])) is dict:
        number_of_batches = len(weights_through_time[layer_keys[0]]['weights'])
        predictions_through_time = np.zeros((features_points.shape[0], number_of_batches))
        for batch_index in range(number_of_batches):
            for layer_index in range(len(layer_keys)):
                aa=1
                model.layers[layer_index].set_weights(
                    [np.array(weights_through_time[layer_keys[layer_index]]['weights'][batch_index]),
                     np.array(weights_through_time[layer_keys[layer_index]]['bias'][batch_index])])
            predictions_through_time[:, batch_index] = model.predict(features_points).reshape(features_points.shape[0],
                                                                                              )
    else:
        number_of_batches = len(weights_through_time[layer_keys[0]])
        predictions_through_time = np.zeros((features_points.shape[0], number_of_batches))
        for batch_index in range(number_of_batches):
            for layer_index in range(len(layer_keys)):
                model.layers[layer_index].set_weights(
                    [np.array(weights_through_time[layer_keys[layer_index]][batch_index])])
            predictions_through_time[:, batch_index] = model.predict(features_points).reshape(
                features_points.shape[0],
            )
    return predictions_through_time


def extract_prediction_landscape(x_positive, x_negative
                                 , x1, x2, model_predictions, gt_predictions, sample_dimension
                                 ):
    if sample_dimension == 2:
        xx1, xx2 = np.meshgrid(x1, x2)
        figure, axs = plt.subplots(2)
        axs[0].pcolormesh(xx1, xx2, model_predictions.reshape(xx1.shape))
        axs[0].scatter(x_positive[:, 0], x_positive[:, 1], c='r')
        axs[0].scatter(x_negative[:, 0], x_negative[:, 1], c='b')
        axs[1].title.set_text('max margin seperator')
        axs[1].pcolormesh(xx1, xx2, gt_predictions)
        axs[1].scatter(x_positive[:, 0], x_positive[:, 1], c='r')
        axs[1].scatter(x_negative[:, 0], x_negative[:, 1], c='b')
        return figure, axs
    else:
        print("Can not display prediction landscape, insert 2d input data")
        return None


def extract_prediction_landscape_through_time(weights_through_time, model_copy, gt_model, x_positive, x_negative,
                                              sample_dimension, results_path, video_name,
                                              boundary_box_size=1, mesh_grid_step=0.05):
    if sample_dimension == 2:
        xmin = -boundary_box_size
        xmax = boundary_box_size
        ymin = -boundary_box_size
        ymax = boundary_box_size
        x1 = np.arange(xmin, xmax, mesh_grid_step)
        x2 = np.arange(ymin, ymax, mesh_grid_step)
        xx1, xx2 = np.meshgrid(x1, x2)
        stacked_x_for_model_prediction = np.c_[xx1.ravel(), xx2.ravel()]
        aa=1
        model_predictions_through_time = predict_through_time(weights_through_time, stacked_x_for_model_prediction,
                                                              model=model_copy)
        gt_predictions = gt_model.predict(stacked_x_for_model_prediction).reshape(xx1.shape)
        for run_index in range(model_predictions_through_time.shape[1]):
            current_landscape, current_landscape_axes = extract_prediction_landscape(x_positive, x_negative, x1, x2,
                                                                                     np.sign(
                                                                                         model_predictions_through_time[
                                                                                         :, run_index].reshape(
                                                                                             xx1.shape)),
                                                                                     gt_predictions, sample_dimension)
            current_landscape_axes[0].set_title('neural network at index {t} prediction'.format(t=str(run_index)))
            save_figure(current_landscape, results_path,
                        "/file{run_index}".format(run_index=run_index))
            plt.clf()
            plt.cla()
            plt.close()

        images = [img for img in os.listdir(results_path) if img.startswith("file") and img.endswith(".png")]
        images.sort(key=natural_keys_string_sort)
        frame = cv2.imread(os.path.join(results_path, images[0]))
        height, width, layers = frame.shape
        os.chdir(results_path)
        video = cv2.VideoWriter('temp.avi', 0, fps=5, frameSize=(width, height))
        for image in images:
            video.write(cv2.imread(image))
        cv2.destroyAllWindows()
        video.release()
        os.system("ffmpeg -y -i temp.avi -c:a aac -b:a 128k -c:v libx264 -crf 23 {video_name}.mp4".format(
            video_name=video_name))
        os.remove('temp.avi')
        for file_name in glob.glob("file*.png"):
            os.remove(file_name)


def extract_training_points_loss(weights, x_train, y_train, model_copy, predictions_vs_labels_arrays_loss):
    training_predictions = predict_through_time(weights, x_train, model=model_copy)
    training_labels = np.tile(np.reshape(y_train, (training_predictions.shape[0], 1)),
                              (1, training_predictions.shape[1]))
    error_trough_training = predictions_vs_labels_arrays_loss(training_predictions, training_labels)
    return error_trough_training


def extract_neurons_alignment(weights, w_star):
    if type(weights.get(list(weights.keys())[0])) is dict:
        first_layer_weights = weights['layer_0']['weights']
    else:
        first_layer_weights = weights['layer_0']

    weights_neurons_direction = [normalize(weight_t, axis=0, norm='l2') for weight_t in
                                 first_layer_weights]


    if first_layer_weights[0].shape[1] > 1:
        u_bar_matrix = np.concatenate((np.tile(np.transpose(w_star),
                                               (1, int(first_layer_weights[0].shape[1] / 2))),
                                       np.tile(-np.transpose(w_star),
                                               (1, int(first_layer_weights[0].shape[1] / 2)))),
                                      axis=1)
    else:
        u_bar_matrix = np.transpose(w_star)
    weights_direction_to_max_margin = [normalized_weight_t - u_bar_matrix for normalized_weight_t in
                                       weights_neurons_direction]
    neurons_direction_to_max_margin_distance = np.array([np.linalg.norm(weights_direction_to_max_margin_t, axis=0) for
                                                         weights_direction_to_max_margin_t in
                                                         weights_direction_to_max_margin])
    return neurons_direction_to_max_margin_distance, weights_neurons_direction


def visualize_neurons_directions(weights_neurons_direction, max_margin_seperator_of_train, x_train, y_train,
                                 sample_dimension, first_label, second_label, sample_points_size=5,
                                 figure_handle=None):
    if not figure_handle:
        figure_handle = plt.figure()
    axis = figure_handle.gca()
    if sample_dimension == 2:
        neurons_directions_through_time = np.array(weights_neurons_direction)
        aa=1
        points_indices = np.random.choice(neurons_directions_through_time.shape[2], sample_points_size, replace=False)
        for index in points_indices:
            neuron_direction = neurons_directions_through_time[:, :, index]
            max_updates_index = neuron_direction.shape[0] - 1
            axis.plot([0, neuron_direction[0, 0]],
                      [0, neuron_direction[0, 1]],
                      label='index_{index}'.format(index=index), linestyle=':')
            axis.plot([0, neuron_direction[int(max_updates_index / 2), 0]],
                      [0, neuron_direction[int(max_updates_index / 2), 1]],
                      label='index_{index}'.format(index=index), linestyle='--')
            axis.plot([0, neuron_direction[max_updates_index, 0]],
                      [0, neuron_direction[max_updates_index, 1]],
                      label='index_{index}'.format(index=index))
        handels_and_labels = axis.get_legend_handles_labels()
        labels_names = handels_and_labels[1]
        for i, p in enumerate(axis.get_lines()):  # this is the loop to change Labels and colors
            if p.get_label() in labels_names[:i]:  # check for Name already exists
                idx = labels_names.index(p.get_label())  # find ist index
                p.set_c(axis.get_lines()[idx].get_c())  # set color
                p.set_label('_' + p.get_label())  # hide label in auto-legend
                p.set_alpha(float(i) / (2 * len(points_indices)))
        axis.arrow(0, 0, max_margin_seperator_of_train[0][0], max_margin_seperator_of_train[0][1], color='blue',
                   width=0.01, label='max_margin', linewidth=2, linestyle='-.')
        axis.arrow(0, 0, -max_margin_seperator_of_train[0][0], -max_margin_seperator_of_train[0][1],
                   color='blue', width=0.01, linewidth=2, head_width=0.001, linestyle='-.')
        plt.legend(loc='center')
        x_positive = x_train[np.where(y_train == first_label)[0], :]
        x_negative = x_train[np.where(y_train == second_label)[0], :]
        axis.scatter(x_positive[:, 0], x_positive[:, 1], c='r', marker='*')
        axis.scatter(x_negative[:, 0], x_negative[:, 1], c='b', marker='*')
        axis.legend(loc='upper right', fontsize='xx-small')
        axis.set_xlabel('Index')
        return figure_handle
    else:
        return None


def visualize_randomly_sampled_rows_from_array(rows_through_index, y_axis_title='Point error', sample_points_size=10,
                                               figure_handle=None):
    if not figure_handle:
        figure_handle = plt.figure()
    axis = figure_handle.gca()
    points_indices = np.random.choice(rows_through_index.shape[0], sample_points_size, replace=False)
    for train_point_index in points_indices:
        axis.plot(range(len(rows_through_index[train_point_index, :])),
                  rows_through_index[train_point_index, :],
                  label='point_index_{index}'.format(index=train_point_index))

    axis.legend(loc='upper right', fontsize='xx-small')
    axis.set_xlabel('Index')
    axis.set_ylabel(y_axis_title)
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
            weights_plots_dict[key] = axis.scatter(range(len(weights_2_vs_fro_norm_ratio[key])),
                                                   weights_2_vs_fro_norm_ratio[key],
                                                   label="{key}{label_suffix}".format(key=key,
                                                                                      label_suffix=label_title_suffix))
    else:
        for key in np.array(list(weights_2_vs_fro_norm_ratio.keys()))[:-1]:
            weights_plots_dict[key] = axis.scatter(range(len(weights_2_vs_fro_norm_ratio[key])),
                                                   weights_2_vs_fro_norm_ratio[key],
                                                   label="{key}{label_suffix}".format(key=key,
                                                                                      label_suffix=label_title_suffix))
    axis.legend()
    axis.set_xlabel('Train Index')
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
