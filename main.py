import glob
from functools import partial
import matplotlib

from configurations.alons_configuration import CURRENT_TIME_STR

# matplotlib.use("Agg")
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import normalize

import metrics_visualizer
import tests, parameters_initialization, input_generator
import numpy as np
from input_generator import get_linear_ground_truth
from configurations import utils
import os
import arrow

from metrics_visualizer import extract_perceptron_ratio

os.environ["PATH"] += os.pathsep + r'C:\Program Files (x86)\Graphviz2.38\bin'
import time
import numpy as np
from functools import partial

np.seterr(divide='ignore', invalid='ignore')

from configurations import alons_configuration, linear_regression_configuration

# configuration = alons_configuration.alons_paper_configuration
tests.visualize_matus_proof_potential(alons_configuration.alons_paper_configuration)
# period = 10
# slope_sign = -1
# intercept_sign = -1
# intercept = 2*period-1
# x = []
# y = []
# aa = 1
# for i in np.linspace(-1, -float(1) / period, period):
#     x_temp = np.linspace(i, i + 1 / period)
#     y_temp = slope_sign * 2 * period * x_temp + intercept_sign * intercept
#     slope_sign *= -1
#     intercept_sign *= -1
#     intercept -= 2
#     x = np.append(x, x_temp)
#     y = np.append(y, y_temp)
# # slope_sign *= -1
# intercept_sign *= -1
# intercept = 1
# for i in np.linspace(0,float(period-1)/period,period):
#     x_temp = np.linspace(i, i + 1 / period)
#     y_temp = slope_sign * 2 * period * x_temp + intercept_sign * intercept
#     slope_sign *= -1
#     intercept_sign *= -1
#     intercept += 2
#     x = np.append(x, x_temp)
#     y = np.append(y, y_temp)
# import matplotlib.pyplot as plt
#
# plt.plot(x, y)
# plt.show()

# def relu_function(x,slope,bias):
#     return np.maximum(0,slope*(x-bias))
#
# x= np.linspace(-1,1)
# m=2
import matplotlib.pyplot as plt
# y=-(relu_function(x,2,-1)-relu_function(x,2,-0.5))+relu_function(x,2,-0.5)-relu_function(x,2,0)+(relu_function(x,2,0)-relu_function(x,2,0.5))-(relu_function(x,2,0.5)-relu_function(x,2,1))
# plt.plot(x,y)
# plt.show()
# NUMBER_OF_NEURONS=10
# d=2
# boundary_box_size=1
# mesh_grid_step=0.05
# xmin = -boundary_box_size
# xmax = boundary_box_size
# ymin = -boundary_box_size
# ymax = boundary_box_size
# x1 = np.arange(xmin, xmax, mesh_grid_step)
# x2 = np.arange(ymin, ymax, mesh_grid_step)
# xx1, xx2 = np.meshgrid(x1, x2)
# stacked_x_for_model_prediction = np.c_[xx1.ravel(), xx2.ravel()]
# W=np.zeros(shape=(2*NUMBER_OF_NEURONS,d))
# W[1,:] = 2*np.array([1,1])
# W[2,:] = 0.5*np.array([-1,-1])
# # W[5,:] = -np.eye(1,d,0)
# # W[15,:] = -np.array([1,-1])
# temp = np.maximum(0, np.matmul(W, np.transpose(stacked_x_for_model_prediction)))
# constant=1/np.sqrt(2*NUMBER_OF_NEURONS)
# v = np.concatenate((constant * np.ones(NUMBER_OF_NEURONS),
#                      -constant * np.ones(NUMBER_OF_NEURONS))).reshape(1,2*NUMBER_OF_NEURONS)
# prediction=np.sign(np.matmul(v,temp)).reshape(xx1.shape)
# figure =plt.figure()
# axes = figure.gca()
# axes.pcolormesh(xx1, xx2,prediction)
# # figure.colorbar()
# plt.show()
aa=1
#W=np.zeros(shape=(2*NUMBER_OF_NEURONS,d))
#W[1,:] = 2*np.array([1,1])
#W[2,:] = 0.5*np.array([-1,-1])
#bias = np.zeros(shape=(2*NUMBER_OF_NEURONS,1))
#bias[1,:]=0.25
#bias[2,:]=-0.6