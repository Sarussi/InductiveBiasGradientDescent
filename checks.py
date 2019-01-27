from input_parser import MNISTDataProvider
from configuration import configuration_parameters
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle

# mnist_data = MNISTDataProvider()

N = 1000
d = 2
train_test_split_ration = 0.75
data_provider = configuration_parameters["data"]["data_provider"]
x_data, y_data = data_provider.read(N, d)
x_data, y_data = shuffle(x_data, y_data)

