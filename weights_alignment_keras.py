from keras.callbacks import LearningRateScheduler
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
def logistic_loss(y_true,y_pred):
    return K.mean(K.log(1 + K.exp(-y_true * y_pred)))
DIMENSION = 100
SAMPLE_SIZE = 150 * DIMENSION
NUMBER_OF_LAYERS = 4
MAX_NEURONS_IN_LAYER = 100
MARGIN = 0.000000001
configuration_parameters = {
    'data': {
        'data_provider': GaussianLinearSeparableDataProvider(margin=MARGIN),
        'filter_arguments': None,
        'sample_size': SAMPLE_SIZE,
        'sample_dimension': DIMENSION,
        'shuffle': True,
        'train_test_split_ration': 0.75
    },

    'model': {
        'learning_rate': 0.01,
        'change_learning_rate': False,
        'learning_rate_change_rate':1.5,
        'learning_rate_updates_count':6,
        'learning_rate_epochs_between_updates':100,
        'batch_size': 1,
        'number_of_neurons_in_layers': set_network_archeitecture(MAX_NEURONS_IN_LAYER, NUMBER_OF_LAYERS),
        'activation_type': activations.linear,
        'loss_type': logistic_loss,
        'input_dimension': DIMENSION,
        'number_of_epochs': 20000,
        'number_of_runs': 1
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
model.compile(optimizer='sgd',
              loss=configuration_parameters['model']['loss_type'],
              metrics=['accuracy'])
def step_decay(epoch):
    if configuration_parameters['model']['change_learning_rate']:
        initial_lrate = configuration_parameters['model']['learning_rate']
        increase = configuration_parameters['model']['learning_rate_change_rate']
        epochs_between_updates =configuration_parameters['model']['learning_rate_epochs_between_updates']
        # epochs_increase = configuration_parameters['model']['number_of_epochs']/(1+configuration_parameters['model']['learning_rate_updates_count'])
        if epoch < (epochs_between_updates*configuration_parameters['model']['learning_rate_updates_count'])/2:
            lrate = initial_lrate * np.math.pow(increase,min(np.math.floor((1+epoch)/epochs_between_updates),2*(configuration_parameters['model']['learning_rate_updates_count']-1)+configuration_parameters['model']['learning_rate_updates_count']))
        elif epoch < (epochs_between_updates*configuration_parameters['model']['learning_rate_updates_count']):
            lrate = initial_lrate * np.math.pow(increase,min(np.math.floor((1+epoch)/epochs_between_updates),configuration_parameters['model']['learning_rate_updates_count']))
        else:
            lrate = initial_lrate * np.math.pow(increase,configuration_parameters['model']['learning_rate_updates_count'])

        print(lrate)
    else:
        lrate = configuration_parameters['model']['learning_rate']
    return lrate


class History_LAW(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.epoch = []
        self.weights = []
        self.history = {}
        self.losses = []
        self.lr = []
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
        self.losses.append(logs.get('loss'))
        self.lr.append(step_decay(len(self.losses)))
# mc = keras.callbacks.ModelCheckpoint('weights{epoch:08d}.h5',save_weights_only=True,period=1)
number_of_epochs = configuration_parameters['model']['number_of_epochs']

weights_2_vs_fro_norm_ratio = {}
weights_2_vs_nuc_norm_ratio = {}
for layer_index in range(0, len(model.layers)):
    key ='layer_{layer_index}'.format(layer_index=layer_index)
    weights_2_vs_fro_norm_ratio[key] = np.zeros(number_of_epochs)
    weights_2_vs_nuc_norm_ratio[key] = np.zeros(number_of_epochs)

number_of_runs = configuration_parameters['model']['number_of_runs']
for run in range(0,number_of_runs):
    model_Hist = History_LAW()
    lrate = LearningRateScheduler(step_decay)
    model.fit(x_train,y_train,epochs=number_of_epochs,callbacks=[model_Hist,lrate])
    weights_temp = model_Hist.weights
    weights_temp.remove(weights_temp[0])
    weights_layers_epochs_temp = {}
    for layer_index in range(0, len(model.layers)):
        key = 'layer_{layer_index}'.format(layer_index=layer_index)
        weights_layers_epochs_temp[key]=[]
    for index in range(0, len(weights_temp)):
        current_weights = weights_temp[index]
        for key in current_weights.keys():
            weights_layers_epochs_temp[key].append(current_weights[key])
    for key in weights_layers_epochs_temp.keys():
        weights_2_vs_fro_norm_ratio[key] += [
            np.linalg.norm(current_weights, ord=2) / np.linalg.norm(current_weights, ord='fro') for current_weights
            in
            weights_layers_epochs_temp[key]]
        weights_2_vs_nuc_norm_ratio[key] += [
            np.linalg.norm(current_weights, ord=2) / np.linalg.norm(current_weights, ord='nuc') for current_weights
            in
            weights_layers_epochs_temp[key]]
for key in weights_2_vs_fro_norm_ratio.keys():
    weights_2_vs_fro_norm_ratio[key] /= number_of_runs
    weights_2_vs_nuc_norm_ratio[key] /= number_of_runs
current_date = str(arrow.now().format('YYYY-MM-DD'))
configuration_path = "dimension_{sample_dimension}_sample_{sample_size}_epochs_{number_of_epochs}_margin_{margin}_activation_{activation_type}_loss_type_{loss_type}_rate_{learning_rate_state}_architecture_{number_of_neurons_for_layer}".format(
    sample_dimension=configuration_parameters["data"]["sample_dimension"],
    sample_size=configuration_parameters["data"]["sample_size"],
    number_of_epochs=configuration_parameters["model"]["number_of_epochs"],
    margin=MARGIN,
    activation_type='linear',
    loss_type=configuration_parameters["model"]["loss_type"].__name__,
    learning_rate_state=str(configuration_parameters['model']['change_learning_rate']),
    number_of_neurons_for_layer=str(
        list(configuration_parameters["model"]["number_of_neurons_in_layers"].values())))

results_path = "{cwd_path}\{current_date}\weights_alignment\{configuration_path}".format(
        cwd_path=str(os.getcwd()),
        current_date=current_date, configuration_path=configuration_path)
if not os.path.exists(results_path):
    os.makedirs(results_path)
configuration_parameters_text = open(
    "{results_path}\configuration_parameters.txt".format(
        results_path=results_path), "w+")
configuration_parameters_text.write(str(configuration_parameters))
configuration_parameters_text.close()

plot_index = 1

weight_plots_dict = {}
f_1 = plt.figure(plot_index)
for key in list(weights_2_vs_fro_norm_ratio.keys())[:-1]:
    weight_plots_dict[key] = plt.scatter(range(number_of_epochs), weights_2_vs_fro_norm_ratio[key], label=key)
plt.legend(tuple(weight_plots_dict.values()), tuple(weight_plots_dict.keys()))
plt.xlabel('Epochs')
plt.ylabel('Norm 2 vs frobenius ratio')
plt.ylim((0.2,1))
f_1.savefig(r"{results_path}\layers_2_vs_fro_norms_ratio.png".format(
        results_path=results_path))
plot_index +=1
