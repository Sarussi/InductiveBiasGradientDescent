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
        # w=np.array([0,1])
        self.w = w / np.linalg.norm(w)
        self.w = (1 / self.margin) * self.w
        X_positive = np.empty([1, d])
        while X_positive.shape[0] < N / 2:
            temp_point = self.random_normal(1, d, mu=self.mu, sigma=self.sigma)
            if (np.inner(temp_point, self.w) > 1):
                X_positive = np.append(X_positive, temp_point, axis=0)
        for i in range(X_positive.shape[0]):
            if np.linalg.norm(X_positive[i, :]) > 1:
                X_positive[i, :] /= np.amax(np.linalg.norm(X_positive, axis=1))

        X_negative = np.empty([1, d])
        while X_negative.shape[0] < N / 2:
            temp_point = self.random_normal(1, d, mu=self.mu, sigma=self.sigma)
            if (np.inner(temp_point, self.w) < -1):
                X_negative = np.append(X_negative, temp_point, axis=0)
        for i in range(X_negative.shape[0]):
            if np.linalg.norm(X_negative[i, :]) > 1:
                X_negative[i, :] /= np.amax(np.linalg.norm(X_negative, axis=1))

        y_positive = np.ones((X_positive.shape[0], 1)).reshape(X_positive.shape[0], 1)
        y_negative = -1 * np.ones((X_negative.shape[0], 1)).reshape(X_negative.shape[0], 1)
        if d == 2:
            fig = plt.figure()
            ax1 = fig.add_subplot(111)
            ax1.scatter(X_positive[:, 0], X_positive[:, 1], c='r', marker='+')
            ax1.scatter(X_negative[:, 0], X_negative[:, 1], c='b', marker='o')
            plt.show()
        x_data = np.concatenate((X_positive, X_negative), axis=0)
        y_data = np.concatenate((y_positive, y_negative), axis=0)
        return x_data, y_data

    def filter(self, features_array, label_array, desired_labels_list=None):
        return features_array, label_array

    def normalize(self, features_data):
        return (features_data - self.mu) / self.sigma
