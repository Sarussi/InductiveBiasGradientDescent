from functools import partial

from mpl_toolkits.mplot3d import Axes3D

from simulator import tests, parameters_initialization
from simulator.configurations import alons_configuration,linear_regression_configuration
import numpy as np
from simulator.input_generator import get_linear_ground_truth
from simulator.configurations import utils
import os
os.environ["PATH"] += os.pathsep + r'C:\Program Files (x86)\Graphviz2.38\bin'
# # tests.visualize_single_model_architecture_average_run(alons_configuration.alons_paper_configuration)
# network_architecture_list = []
# weights_initialization_methods = []
# for max_number_of_neurons in np.power(float(10), np.ones(3)):
#     network_architecture_list.append(utils.set_network_architecture_linearly(2 * max_number_of_neurons,
#                                                                              alons_configuration.alons_paper_configuration[
#                                                                         'model']['number_of_layers']))
# for std in 0.5 * np.power(float(10), -np.array(range(0, 3))):
#     weights_initialization_methods.append(partial(parameters_initialization.initialize_all_weights_from_method,
#                                                   initialization_method=partial(parameters_initialization.random_normal,
#                                                                                 std=std)))
# tests.visualize_change_weights_initialization_and_architecture_average_run(
#     alons_configuration.alons_paper_configuration, weights_initializers_list=weights_initialization_methods,
#     network_architecture_list=network_architecture_list)
# tests.visualize_single_model_architecture_average_run(alons_configuration.alons_paper_configuration)
# tests.single_model_architecture_average_run(alons_configuration.alons_paper_configuration)
tests.visualize_single_model_architecture_average_run(linear_regression_configuration.configuration)
