from sklearn.utils import shuffle
import numpy as np
import copy
from configuration import configuration_parameters
from model import measure_model_average
import matplotlib.pyplot as plt
import arrow
import os

N = configuration_parameters["data"]["sample_size"]
d = configuration_parameters["data"]["sample_dimension"]
train_test_split_ration = configuration_parameters["data"]["train_test_split_ration"]
data_provider = configuration_parameters["data"]["data_provider"]
x_data, y_data = data_provider.read(N, d)
if configuration_parameters["data"]["shuffle"]:
    x_data, y_data = shuffle(x_data, y_data)
x_train = x_data[0:int(x_data.shape[0] * train_test_split_ration), :]
y_train = y_data[0:int(x_data.shape[0] * train_test_split_ration)]
y_test = y_data[int(x_data.shape[0] * train_test_split_ration):int(x_data.shape[0])]
x_test = x_data[int(x_data.shape[0] * train_test_split_ration):int(x_data.shape[0]), :]
small_network_sizes = np.linspace(1, 100, num=2, dtype=int)
big_network_sizes = np.linspace(100, 300, num=1, dtype=int)
# network_sizes = np.power(2, range(4, 9))
network_sizes = np.concatenate((small_network_sizes, big_network_sizes), axis=None)
final_train_error_list = []
final_test_error_list = []
for network_size in network_sizes:
    temp_configuration = copy.deepcopy(configuration_parameters)
    temp_configuration["data"]["number_of_neurons"] = network_size
    train_error_epochs, test_error_epochs, _ = measure_model_average(x_train, y_train, x_test, y_test,
                                                                     temp_configuration)
    final_train_error = train_error_epochs[-1]
    final_train_error_list.append(final_train_error)
    final_test_error = test_error_epochs[-1]
    final_test_error_list.append(final_test_error)
final_train_error_np = np.array(final_train_error_list)
final_test_error_np = np.array(final_test_error_list)
f_1 = plt.figure(1)
train_error_line, = plt.plot(network_sizes, final_train_error_np, 'ro-')
test_error_line, = plt.plot(network_sizes, final_test_error_np, 'bo-')
plt.xticks(network_sizes)
plt.xlabel('Network Size In Neurons')
plt.ylabel('Error')
plt.legend([train_error_line, test_error_line], ['train_error_line', 'test_error_line'])
plt.show()
current_date = str(arrow.now().format('YYYY-MM-DD'))
results_path = "{cwd_path}\{current_date}\one_layer\error_vs_network_size".format(cwd_path=str(os.getcwd()),
                                                                                  current_date=current_date)
if not os.path.isdir(results_path):
    os.makedirs(results_path)
configuration_path = "sample_size_{sample_size}sample_dimension_{sample_dimension}_activation_type_{activation_type}_number_of_averages_{number_of_runs}".format(
    sample_size=configuration_parameters["data"]["sample_size"],
    sample_dimension=configuration_parameters["data"]["sample_dimension"],
    activation_type=configuration_parameters["model"]["activation_type"],
    number_of_runs=configuration_parameters["model"]["number_of_runs"])

f_1.savefig("{results_path}\{configuration_path}_error_vs_size.png".format(
    results_path=results_path, configuration_path=configuration_path))
