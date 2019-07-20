import numpy as np


def array_of_dictionaries_mean(dict_list):
    number_of_dict = float(len(dict_list))
    mean_dict = {}
    for key in list(dict_list[0].keys()):
        mean_dict[key] = np.zeros(len(dict_list[0][key]))
        for dictionary in dict_list:
            mean_dict[key] += np.array(dictionary[key])
        mean_dict[key] /= number_of_dict
    return mean_dict
