from sklearn import datasets

import tensorflow as tf
import pandas as pd
import numpy as np
import abc
import matplotlib.pyplot as plt


class DataProvider(abc.ABC):
    @abc.abstractmethod
    def read(self, *args, **kwargs):
        pass

    def filter(self, *args, **kwargs):
        pass

    def normalize(self, *args, **kwargs):
        pass


MAX_PIXEL_VALUE = 255


class MNISTDataProvider(DataProvider):
    def __init__(self):
        self.type = 'MNIST'

    def read(self):
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
        x_data = np.concatenate((x_train, x_test), axis=0)
        y_data = np.concatenate((y_train, y_test), axis=0)
        return x_data, y_data

    def filter(self, features_array, label_array, desired_labels_list):
        features_array = features_array.astype('float32')
        label_array = label_array.astype('float32')
        new_features_array = features_array.reshape(features_array.shape[0],
                                                    features_array.shape[1] * features_array.shape[2])
        features_data_df = pd.DataFrame(data=new_features_array)
        features_data_df["label"] = label_array
        filtered_data_df = features_data_df.loc[features_data_df['label'].isin(desired_labels_list)]
        filtered_features_values = filtered_data_df.drop(columns=["label"]).values
        filtered_labeled_values = filtered_data_df["label"].values
        return filtered_features_values, filtered_labeled_values

    def normalize(self, features_data, max_value=MAX_PIXEL_VALUE):
        return features_data / max_value


class GaussianLinearSeparableDataProvider(DataProvider):
    def __init__(self, k=0, margin=0.01, mu=0, sigma=1):
        self.type = 'Gaussian'
        self.k = k
        self.margin = margin * sigma
        self.mu = mu
        self.sigma = sigma
        self.w = None

    def random_normal(self, N, d, mu, sigma):
        return sigma * np.random.randn(N, d) + mu

    def read(self, N, d):
        w = self.random_normal(1, d, self.mu, self.sigma)
        self.w = w / np.linalg.norm(w)
        self.w = (1 / self.margin) * self.w
        X_positive = np.empty([1, d])
        while X_positive.shape[0] < float(N) / 2:
            temp_point = self.random_normal(1, d, mu=self.mu, sigma=self.sigma)
            if (np.inner(temp_point, self.w) > 1):
                X_positive = np.append(X_positive, temp_point, axis=0)
            positive_precentage_gathered = (X_positive.shape[0] / (N / 2)) * 100
            # if positive_precentage_gathered % 10 == 0:
                # print("Amount of positive samples created by now is: {precentage_of_positive_data}%".format(
                #     precentage_of_positive_data=positive_precentage_gathered))
        X_positive = self.normalize(X_positive)

        X_negative = np.empty([1, d])
        while X_negative.shape[0] < float(N) / 2:
            temp_point = self.random_normal(1, d, mu=self.mu, sigma=self.sigma)
            if (np.inner(temp_point, self.w) < -1):
                X_negative = np.append(X_negative, temp_point, axis=0)
            negative_precentage_gathered = (X_negative.shape[0] / (N / 2)) * 100
            # if negative_precentage_gathered % 10 == 0:
            #     print("Amount of negative samples created by now is: {precentage_of_negative_data}%".format(
            #         precentage_of_negative_data=negative_precentage_gathered))
        X_negative = self.normalize(X_negative)

        y_positive = np.ones((X_positive.shape[0], 1)).reshape(X_positive.shape[0], 1)
        y_negative = -1 * np.ones((X_negative.shape[0], 1)).reshape(X_negative.shape[0], 1)
        x_data = np.concatenate((X_positive, X_negative), axis=0)
        y_data = np.concatenate((y_positive, y_negative), axis=0)
        return x_data, y_data

    def filter(self, features_array, label_array, desired_labels_list=None):
        return features_array, label_array

    def normalize(self, features_data):
        for i in range(features_data.shape[0]):
            max_feature_norm = np.amax(np.linalg.norm(features_data, axis=1))
            if np.linalg.norm(features_data[i, :]) > 1:
                features_data[i, :] /= max_feature_norm
        return features_data


class OrthogonalSingleClassDataProvider(DataProvider):
    def __init__(self):
        self.type = 'OrthogonalBasis'

    def read(self, dimension):
        x_data = np.eye(dimension, dimension)
        y_data = np.ones((dimension, 1)).reshape(dimension, 1)
        return x_data, y_data

    def filter(self, features_array, label_array, desired_labels_list=None):
        return features_array, label_array

    def normalize(self, features_data):
        for i in range(features_data.shape[0]):
            max_feature_norm = np.amax(np.linalg.norm(features_data, axis=1))
            if np.linalg.norm(features_data[i, :]) > 1:
                features_data[i, :] /= max_feature_norm
        return features_data


class BlobsLinearSeparableDataProvider(DataProvider):
    def __init__(self, k=0, margin=0.01, mu=0, sigma=1):
        self.type = 'Blobs'
        self.k = k
        self.margin = margin * sigma
        self.mu = mu
        self.sigma = sigma
        self.w = None

    def random_normal(self, N, d, mu, sigma):
        return sigma * np.random.randn(N, d) + mu

    def read(self, N, d):
        first_center = self.random_normal(1, d, self.mu, self.sigma)
        second_center = self.random_normal(1, d, self.mu, self.sigma)
        cluster_std = float(np.linalg.norm(first_center - second_center, ord=2)) / 10
        X, y = datasets.make_blobs(n_samples=N, centers=np.concatenate((first_center, second_center), axis=0),
                                   n_features=d, cluster_std=cluster_std)
        X_positive = X[:, :][y == 0]
        X_positive = self.normalize(X_positive)
        y_positive = y[y == 0]
        X_negative = X[:, :][y == 1]
        X_negative = self.normalize(X_negative)
        y_negative = y[y == 1]
        y_positive = np.ones((X_positive.shape[0], 1)).reshape(X_positive.shape[0], 1)
        y_negative = -1 * np.ones((X_negative.shape[0], 1)).reshape(X_negative.shape[0], 1)
        x_data = np.concatenate((X_positive, X_negative), axis=0)
        y_data = np.concatenate((y_positive, y_negative), axis=0)
        return x_data, y_data

    def filter(self, features_array, label_array, desired_labels_list=None):
        return features_array, label_array

    def normalize(self, features_data):
        for i in range(features_data.shape[0]):
            max_feature_norm = np.amax(np.linalg.norm(features_data, axis=1))
            if np.linalg.norm(features_data[i, :]) > 1:
                features_data[i, :] /= max_feature_norm
        return features_data
