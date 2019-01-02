import tensorflow as tf
import pandas as pd
import numpy as np
import abc

class DataProvider(abc.ABC):
    @abc.abstractmethod
    def read(self,*args,**kwargs):
        pass
    def filter(self,*args,**kwargs):
        pass
    def normalize(self,*args,**kwargs):
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
        return features_data/max_value
