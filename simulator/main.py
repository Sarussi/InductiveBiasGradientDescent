from simulator import tests
from simulator.configurations import alons_configuration
import numpy as np

from simulator.configurations import utils

# tests.visualize_single_model_architecture_average_run(alons_configuration.alons_paper_configuration)
network_architecture_list = []
for max_number_of_neurons in np.power(float(10), np.ones((3,1))):
    network_architecture_list.append(utils.set_network_architecture(max_number_of_neurons,
                                                                    alons_configuration.alons_paper_configuration[
                                                                        'model']['number_of_layers']))
print(network_architecture_list)
tests.visualize_change_learning_rate_and_architecture_average_run(alons_configuration.alons_paper_configuration,
                                                                  learning_rate_list=np.power(float(10),
                                                                                              -np.array(range(3, 6))),
                                                                  network_architecture_list=network_architecture_list)
# tests.visualize_single_model_architecture_average_run(alons_configuration.alons_paper_configuration)
# tests.single_model_architecture_average_run(alons_configuration.alons_paper_configuration)
