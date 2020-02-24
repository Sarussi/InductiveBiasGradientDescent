import numpy as np
from scipy.spatial.distance import cdist
from sklearn import svm
from sklearn.utils import shuffle
from sklearn.linear_model import perceptron
import tensorflow as tf
import time


def uniformly_distributed_samples_on_unit_sphere(number_of_samples, dimension):
    normally_distributed_points = np.random.randn(number_of_samples, dimension)
    points_on_unit_sphere = normally_distributed_points / np.linalg.norm(normally_distributed_points, axis=1).reshape(
        number_of_samples, 1)
    return points_on_unit_sphere


def radamacher_cube_vertices(number_of_samples, dimension):
    cube_vertices_sample = 2 * (np.random.binomial(1, 0.5, number_of_samples * dimension) - 0.5).reshape(
        number_of_samples, dimension)
    cube_vertices_sample = np.unique(cube_vertices_sample, axis=0)
    return cube_vertices_sample


def uniformly_distributed_samples_in_ball(number_of_samples, dimension, radius=1.0, center_location=[0, 0]):
    points_on_unit_sphere = uniformly_distributed_samples_on_unit_sphere(number_of_samples, dimension)
    contraction_scaler = np.random.uniform(size=(number_of_samples, 1))
    points_norm = radius * contraction_scaler
    points_in_ball = points_norm * points_on_unit_sphere
    points_in_ball = points_in_ball + center_location
    return points_in_ball


def generate_uniformly_distributed_on_simplex(N, d):
    y = np.random.standard_exponential(size=(N, d))
    y = np.array(y).reshape(N, d)
    t = [sum(y_i) for y_i in y]
    e = [i[0] / float(i[1]) for i in zip(y, t)]
    e = np.array(e).reshape(N, d)
    # e = e[~np.isnan(e).any(axis=1)]
    return e


def generate_two_simplex(N, d, symmetry_index=0):
    e_1 = generate_uniformly_distributed_on_simplex(N, d)
    e_2 = generate_uniformly_distributed_on_simplex(N, d)
    e_2[:, symmetry_index] = -e_2[:, symmetry_index]
    e = np.concatenate((e_1, e_2), axis=0)
    return e


def random_normal_samples(number_of_samples, dimension, mu=0, sigma=1):
    return sigma * np.random.randn(number_of_samples, dimension) + mu


def separate_to_classes(points, plane_vector):
    margins_from_boundary = np.dot(points, plane_vector.transpose()).reshape(points.shape[0], 1)
    X_positive = points[np.where(margins_from_boundary > 1)[0], :]
    X_negative = points[np.where(margins_from_boundary < -1)[0], :]
    return X_positive, X_negative


def generate_separation_boundary(dimension, margin):
    separating_boundary = np.random.randn(1, dimension)
    separating_boundary /= np.linalg.norm(separating_boundary, axis=1)
    separating_boundary *= (2 / margin)
    return separating_boundary


def generate_linearly_separable_samples(number_of_samples, dimension, separating_boundary, first_label=1,
                                        second_label=-1,
                                        samples_generator=uniformly_distributed_samples_in_ball):
    initial_points_in_ball = samples_generator(number_of_samples, dimension)
    X_positive, X_negative = separate_to_classes(initial_points_in_ball, separating_boundary)
    while X_positive.shape[0] < 0.95 * (number_of_samples / 2) or X_negative.shape[0] < 0.95 * (number_of_samples / 2):
        amount_of_missing_points = 2 * int(max(
            [number_of_samples / 2 - X_positive.shape[0], number_of_samples / 2 - X_negative.shape[0]]))
        current_points_in_ball = samples_generator(amount_of_missing_points, dimension)
        current_X_positive, current_X_negative = separate_to_classes(current_points_in_ball, separating_boundary)
        if X_positive.shape[0] < number_of_samples / 2:
            X_positive = np.concatenate((X_positive, current_X_positive), axis=0)
        if X_negative.shape[0] < number_of_samples / 2:
            X_negative = np.concatenate((X_negative, current_X_negative), axis=0)
        print("number of positive points: {}".format(X_positive.shape[0]))
        print("number of negative points: {}".format(X_negative.shape[0]))
    aa = 1
    margin = np.amin(cdist(X_positive, X_negative))
    print("margin is : {}".format(margin))
    y_positive = np.full((X_positive.shape[0], 1), first_label).reshape(X_positive.shape[0], 1)
    y_negative = np.full((X_negative.shape[0], 1), second_label).reshape(X_negative.shape[0], 1)
    x_data = np.concatenate((X_positive, X_negative), axis=0)
    y_data = np.concatenate((y_positive, y_negative), axis=0)
    x_data, y_data = shuffle(x_data, y_data)
    x_data /= np.linalg.norm(x_data, axis=1).max()
    seperating_boundary = measure_separability_perceptron(x_data, y_data)

    return x_data, y_data, seperating_boundary


def generate_two_seperable_balls(number_of_samples, dimension, first_label=1,
                                 second_label=-1, positive_center_loc=np.array([0.5, 0]), radius=0.2):
    x_positive = uniformly_distributed_samples_in_ball(int(number_of_samples / 2), dimension, radius,
                                                       center_location=positive_center_loc)
    x_negative = uniformly_distributed_samples_in_ball(int(number_of_samples / 2), dimension, radius,
                                                       center_location=-positive_center_loc)
    y_positive = np.full((x_positive.shape[0], 1), first_label).reshape(x_positive.shape[0], 1)
    y_negative = np.full((x_negative.shape[0], 1), second_label).reshape(x_negative.shape[0], 1)
    x_data = np.concatenate((x_positive, x_negative), axis=0)
    y_data = np.concatenate((y_positive, y_negative), axis=0)
    x_data, y_data = shuffle(x_data, y_data)
    seperating_boundary = measure_separability_perceptron(x_data, y_data)

    return x_data, y_data, seperating_boundary


def train_and_test_shuffle_split(x_data, y_data, train_test_split_ratio):
    x_data, y_data = shuffle(x_data, y_data)
    x_train = x_data[0:int(x_data.shape[0] * train_test_split_ratio), :]
    y_train = y_data[0:int(x_data.shape[0] * train_test_split_ratio)]
    x_test = x_data[int(x_data.shape[0] * train_test_split_ratio):int(x_data.shape[0]), :]
    y_test = y_data[int(x_data.shape[0] * train_test_split_ratio):int(x_data.shape[0])]
    return x_train, y_train, x_test, y_test


def read_mnist(demean_flag=False):
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_data = np.concatenate((x_train, x_test), axis=0)
    y_data = np.concatenate((y_train, y_test), axis=0)
    x_data = x_data.reshape((x_data.shape[0], x_data.shape[1] * x_data.shape[2]))
    if demean_flag:
        x_data_mean = x_data.mean(axis=0)
        x_data = x_data - x_data_mean[None, :]
    return x_data, y_data


def get_digits_3_and_5(data, labels, first_label=1, second_label=-1):
    new_data = []
    new_labels = []
    for i in range(len(data)):
        if labels[i] == 5 or labels[i] == 3:
            new_data.append(data[i].tolist())
            if labels[i] == 5:
                new_labels.append(np.array([first_label]).tolist())
            else:
                new_labels.append(np.array([second_label]).tolist())
    return np.array(new_data), np.array(new_labels)


def measure_separability_perceptron(data, labels):
    perceptron_model = perceptron.Perceptron()
    perceptron_model.fit(data, labels.ravel())
    seperating_boundary = perceptron_model.coef_
    print("Perceptron Data Accuracy   " + str(perceptron_model.score(data, labels) * 100) + "%")
    time.sleep(5)
    return seperating_boundary


def svm_normalized_max_margin(data_points, labels):
    # svm_classifier = svm.SVC(kernel='linear', C=1E10)
    svm_classifier = svm.LinearSVC(fit_intercept=False, C=1E12)
    svm_classifier.fit(data_points, labels)
    w = svm_classifier.coef_
    intercept = svm_classifier.intercept_
    return w, intercept, svm_classifier


def get_linearly_separable_mnist(number_of_samples, first_label=1,
                                 second_label=-1, demean_flag=False):
    x_data, y_data = read_mnist(demean_flag)
    row_norms = np.linalg.norm(x_data, axis=1)
    x_data = x_data / np.full(x_data.shape, max(row_norms))
    three_and_five_data, three_and_five_labels = get_digits_3_and_5(x_data, y_data, first_label, second_label)
    idx = np.random.RandomState(0).choice(len(three_and_five_data), number_of_samples)
    three_and_five_data = three_and_five_data[idx[:number_of_samples], :]
    three_and_five_labels = three_and_five_labels[idx[:number_of_samples]]
    seperating_boundary = measure_separability_perceptron(three_and_five_data, three_and_five_labels)
    return three_and_five_data, three_and_five_labels, seperating_boundary


def linear_line(data_param):
    if np.isscalar(data_param['slope']):
        return data_param['slope'] * data_param['x'] + data_param['intercept'] * np.ones(data_param['x'].shape)
    else:
        return [np.matmul(data_param['slope'], x) + data_param['intercept'] for x in data_param['x']]


def get_piecewise_ground_truth(data_params, one_piece_function=linear_line):
    y_values = [one_piece_function(data_param) for data_param in data_params]
    x_values = [data_param['x'] for data_param in data_params]
    gt_x_data = np.concatenate(x_values, axis=0)
    gt_y_data = np.concatenate(y_values, axis=0)
    return gt_x_data, gt_y_data


def get_linear_ground_truth(number_of_samples, dimension, slope_matrix, intercept,
                            samples_generator=uniformly_distributed_samples_in_ball):
    initial_points_in_ball = samples_generator(number_of_samples, dimension)
    data_parameters = {'slope': slope_matrix, 'intercept': intercept, 'x': initial_points_in_ball}
    gt_y_data = np.array(linear_line(data_parameters))
    return initial_points_in_ball, gt_y_data
