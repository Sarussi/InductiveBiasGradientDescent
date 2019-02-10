import os
import json
from configuration import configuration_parameters
from sklearn.utils import shuffle
import arrow
from model import measure_model
import matplotlib.pyplot as plt

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
train_error_results, test_error_results, avg_loss, _ = measure_model(x_train, y_train, x_test, y_test,
                                                                     configuration_parameters)
for i in range(configuration_parameters["model"]["number_of_runs"]):
    train_error_results_temp, test_error_results_temp, avg_loss_temp, weights = measure_model(x_train, y_train, x_test,
                                                                                              y_test,
                                                                                              configuration_parameters)
    train_error_results += train_error_results_temp
    test_error_results += test_error_results_temp
    avg_loss += avg_loss_temp
number_of_epochs = configuration_parameters["model"]["number_of_epochs"]
train_error_results /= configuration_parameters["model"]["number_of_runs"] + 1
test_error_results /= configuration_parameters["model"]["number_of_runs"] + 1
avg_loss /= configuration_parameters["model"]["number_of_runs"] + 1
ylimits = (0, 0.06)
f_1 = plt.figure(1)
plt.scatter(range(number_of_epochs), train_error_results)
plt.xlabel('Epochs')
plt.ylabel('train_error')
plt.ylim(ylimits)
f_2 = plt.figure(2)
plt.scatter(range(number_of_epochs), test_error_results)
plt.xlabel('Epochs')
plt.ylabel('test_error')
plt.ylim(ylimits)
f_3 = plt.figure(3)
plt.scatter(range(number_of_epochs), avg_loss)
plt.xlabel('Epochs')
plt.ylabel('cost_error')
plt.ylim(ylimits)
plt.show()
current_date = str(arrow.now().format('YYYY-MM-DD'))
results_path = "{cwd_path}\{current_date}".format(cwd_path=str(os.getcwd()), current_date=current_date)
if not os.path.isdir(results_path):
    os.makedirs(results_path)

configuration_path = "sample_size_{sample_size}sample_dimension_{sample_dimension}neurons{neurons}_activation_type_{activation_type}".format(
    sample_size=configuration_parameters["data"]["sample_size"],
    sample_dimension=configuration_parameters["data"]["sample_dimension"],
    neurons=configuration_parameters["model"]["number_of_neurons_second_layer"],
    activation_type=configuration_parameters["model"]["activation_type"])
configuration_parameters_text = open(
    "{cwd_path}\{current_date}\{configuration_path}configuration_parameters.txt".format(
        cwd_path=str(os.getcwd()), current_date=current_date, configuration_path=configuration_path), "w")
configuration_parameters_text.write(str(configuration_parameters))
configuration_parameters_text.close()

f_1.savefig("{cwd_path}\{current_date}\{configuration_path}_train_error.png".format(
    cwd_path=str(os.getcwd()), current_date=current_date, configuration_path=configuration_path))
f_2.savefig("{cwd_path}\{current_date}\{configuration_path}_test_error.png".format(
    cwd_path=str(os.getcwd()), current_date=current_date, configuration_path=configuration_path))
f_3.savefig(
    "{cwd_path}\{current_date}\{configuration_path}_cost_error.png".format(
        cwd_path=str(os.getcwd()), current_date=current_date, configuration_path=configuration_path))



