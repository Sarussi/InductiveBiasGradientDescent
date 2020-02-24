import numpy as np

import re

def atoi(text):
    return int(text) if text.isdigit() else text


def natural_keys_string_sort(text):
    '''
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    '''
    return [atoi(c) for c in re.split(r'(\d+)', text)]


def array_of_dictionaries_mean(dict_list):
    number_of_dict = float(len(dict_list))
    mean_dict = {}
    intersection_indexes = {}
    for key in list(dict_list[0].keys()):
        intersection_indexes[key] = min([np.size(np.array(dictionary[key])) for dictionary in dict_list])
        mean_dict[key] = np.zeros((1, intersection_indexes[key]))
        for dictionary in dict_list:
            mean_dict[key] += np.array(dictionary[key][:intersection_indexes[key]])
        mean_dict[key] /= number_of_dict
    return mean_dict


def dict_to_str(dictionary):
    return str(dictionary).replace('{', '').replace('}', '').replace(':', '_').replace(' ', '').replace("'",
                                                                                                        '').replace(',',
                                                                                                                    '_')
