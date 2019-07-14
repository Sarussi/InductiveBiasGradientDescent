import numpy as np
from scipy.spatial.distance import cdist
from sklearn.utils import shuffle


def uniformly_distributed_samples_in_ball(number_of_samples, dimension, radius=1):
    normally_distributed_points = np.random.randn(number_of_samples, dimension)
    points_on_unit_sphere = normally_distributed_points / np.linalg.norm(normally_distributed_points, axis=1).reshape(
        number_of_samples, 1)
    contraction_scaler = np.random.uniform(size=(number_of_samples, 1))
    points_norm = radius * contraction_scaler
    points_in_ball = points_norm * points_on_unit_sphere
    return points_in_ball


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


def generate_linearly_separable_samples(number_of_samples, dimension, margin,
                                        samples_generator=uniformly_distributed_samples_in_ball):
    initial_points_in_ball = samples_generator(number_of_samples, dimension)
    separating_boundary = generate_separation_boundary(dimension, margin)
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
    margin = np.amin(cdist(X_positive, X_negative))
    print("margin is : {}".format(margin))
    y_positive = np.ones((X_positive.shape[0], 1)).reshape(X_positive.shape[0], 1)
    y_negative = -1 * np.ones((X_negative.shape[0], 1)).reshape(X_negative.shape[0], 1)
    x_data = np.concatenate((X_positive, X_negative), axis=0)
    y_data = np.concatenate((y_positive, y_negative), axis=0)
    return x_data, y_data


def train_and_test_shuffle_split(x_data, y_data, train_test_split_ratio):
    x_data, y_data = shuffle(x_data, y_data)
    x_train = x_data[0:int(x_data.shape[0] * train_test_split_ratio), :]
    y_train = y_data[0:int(x_data.shape[0] * train_test_split_ratio)]
    x_test = x_data[int(x_data.shape[0] * train_test_split_ratio):int(x_data.shape[0]), :]
    y_test = y_data[int(x_data.shape[0] * train_test_split_ratio):int(x_data.shape[0])]
    return x_train, y_train, x_test, y_test
