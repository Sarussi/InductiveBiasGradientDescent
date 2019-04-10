from sklearn.utils import shuffle
import numpy as np
import matplotlib.pyplot as plt
from input_parser import GaussianLinearSeparableDataProvider
import copy
import arrow
import os
from keras.layers import activations,advanced_activations
# from temp_functions import average_linear_model_weights_measurments
#
#
import keras
def set_network_archeitecture(max_neurons_in_layer, number_of_layers):
    neurons_in_layers_array = np.linspace(start=max_neurons_in_layer, stop=1, num=number_of_layers,
                                          dtype=int)
    number_of_neurons_in_layers = dict()
    for layer_index in range(0, len(neurons_in_layers_array)):
        current_layer_key = 'layer_{index}'.format(index=layer_index + 1)
        number_of_neurons_in_layers[current_layer_key] = neurons_in_layers_array[layer_index]
    return number_of_neurons_in_layers
import keras.backend as K
def exponent_loss(y_true,y_pred):
    return K.mean(K.exp(-y_true * y_pred))
#
DIMENSION = 3
SAMPLE_SIZE = 1000 * DIMENSION
NUMBER_OF_LAYERS = 3
MAX_NEURONS_IN_LAYER = 5
configuration_parameters = {
    'data': {
        'data_provider': GaussianLinearSeparableDataProvider(margin=0.00000001),
        'filter_arguments': None,
        'sample_size': SAMPLE_SIZE,
        'sample_dimension': DIMENSION,
        'shuffle': True,
        'train_test_split_ration': 0.75
    },

    'model': {
        'learning_rate': 0.001,
        'decreasing_learning_rate': True,
        'batch_size': 1,
        'number_of_neurons_in_layers': set_network_archeitecture(MAX_NEURONS_IN_LAYER, NUMBER_OF_LAYERS),
        'activation_type': activations.linear,
        'loss_type': keras.losses.hinge,
        'input_dimension': DIMENSION,
        'number_of_epochs': 1000,
        'number_of_runs': 5
    }

}


d = configuration_parameters["data"]["sample_dimension"]
N = configuration_parameters["data"]["sample_size"]
train_test_split_ration = configuration_parameters["data"]["train_test_split_ration"]
data_provider = configuration_parameters["data"]["data_provider"]
x_data, y_data = data_provider.read(N, d)
if configuration_parameters["data"]["shuffle"]:
    x_data, y_data = shuffle(x_data, y_data)
x_train = x_data[0:int(x_data.shape[0] * train_test_split_ration), :]
y_train = y_data[0:int(x_data.shape[0] * train_test_split_ration)]
y_test = y_data[int(x_data.shape[0] * train_test_split_ration):int(x_data.shape[0])]
x_test = x_data[int(x_data.shape[0] * train_test_split_ration):int(x_data.shape[0]), :]
print(configuration_parameters['model']['number_of_neurons_in_layers'])
from keras.layers import Dense
from keras.models import Sequential
import numpy as np
import keras
model=Sequential()
neurons_in_layer = list(configuration_parameters['model']['number_of_neurons_in_layers'].values())
model.add(Dense(neurons_in_layer[0], input_dim=x_train.shape[1],bias=False,activation=configuration_parameters['model']['activation_type']))
for index in range(1,len(neurons_in_layer)):
    model.add(Dense(neurons_in_layer[index], bias=False,activation=configuration_parameters['model']['activation_type']))
aa=1
model.compile(optimizer='sgd',
              loss=configuration_parameters['model']['loss_type'],
              metrics=['accuracy'])

class History_LAW(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.epoch = []
        self.weights = []
        self.history = {}
        self.weights.append(self.model.weights)

    def on_epoch_end(self, epoch, logs={}):
        logs = logs or {}
        self.epoch.append(epoch)
        for k, v in logs.items():
            self.history.setdefault(k, []).append(v)

        modelWeights = dict()
        for layer_index in range(0,len(model.layers)):
            current_layer = model.layers[layer_index]
            modelWeights['layer_{layer_index}'.format(layer_index=layer_index)]=current_layer.get_weights()[0]
        self.weights.append(modelWeights)

# mc = keras.callbacks.ModelCheckpoint('weights{epoch:08d}.h5',save_weights_only=True,period=1)
aa=1
model_Hist = History_LAW()
number_of_epochs = configuration_parameters['model']['number_of_epochs']
model.fit(x_train,y_train,epochs=number_of_epochs,callbacks=[model_Hist])
aa=1
print('========== Type of w')
for weight in model_Hist.weights:
    print([w for w in weight])

weights_temp = model_Hist.weights
weights_temp.remove(weights_temp[0])
aa=1
weights_layers_epochs_temp = {}
weights_2_vs_fro_norm_ratio = {}
weights_2_vs_nuc_norm_ratio = {}
for key in weights_temp[0].keys():
    weights_layers_epochs_temp[key] = []
    weights_2_vs_fro_norm_ratio[key] = np.zeros(number_of_epochs)
    weights_2_vs_nuc_norm_ratio[key] = np.zeros(number_of_epochs)
for index in range(0, len(weights_temp)):
    current_weights = weights_temp[index]
    for key in current_weights.keys():
        weights_layers_epochs_temp[key].append(current_weights[key])
aa=1
print(weights_layers_epochs_temp)

for key in weights_layers_epochs_temp.keys():
    weights_2_vs_fro_norm_ratio[key] += [
        np.linalg.norm(current_weights, ord=2) / np.linalg.norm(current_weights, ord='fro') for current_weights
        in
        weights_layers_epochs_temp[key]]
    weights_2_vs_nuc_norm_ratio[key] += [
        np.linalg.norm(current_weights, ord=2) / np.linalg.norm(current_weights, ord='nuc') for current_weights
        in
        weights_layers_epochs_temp[key]]

aa=1
plot_index = 1

weight_plots_dict = {}
f_1 = plt.figure(plot_index)
for key in list(weights_2_vs_fro_norm_ratio.keys())[:-1]:
    weight_plots_dict[key] = plt.scatter(range(number_of_epochs), weights_2_vs_fro_norm_ratio[key], label=key)
plt.legend(tuple(weight_plots_dict.values()), tuple(weight_plots_dict.keys()))
plt.xlabel('Epochs')
plt.ylabel('Norm 2 vs frobenius ratio')
f_1.savefig("layers_2_vs_fro_norms_ratio.png")