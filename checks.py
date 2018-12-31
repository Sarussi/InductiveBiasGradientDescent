
from input_parser import MNISTDataProvider
from configuration import configuration_parameters

mnist_data = MNISTDataProvider()
data_provider = configuration_parameters["data_provider"]
x_data,y_data = data_provider.read()
filtered_features, filtered_labels = data_provider.filter(x_data, y_data, configuration_parameters["filter_arguments"])
""" Neural Network.
A 2-Hidden Layers Fully Connected Neural Network (a.k.a Multilayer Perceptron)
implementation with TensorFlow. This example is using the MNIST database
of handwritten digits (http://yann.lecun.com/exdb/mnist/).
Links:
    [MNIST Dataset](http://yann.lecun.com/exdb/mnist/).
Author: Aymeric Damien
Project: https://github.com/aymericdamien/TensorFlow-Examples/
"""

import tensorflow as tf
import numpy as np
# Import MNIST data
# from tensorflow.examples.tutorials.mnist import input_data
# mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Parameters
learning_rate = 0.1
num_steps = 500
batch_size = 128
display_step = 100
k=10
# Network Parameters
n_hidden_1 = 2*k # 1st layer number of neurons
num_input = 784 # MNIST data input (img shape: 28*28)
num_classes = 1 # MNIST total classes (0-9 digits)
training_epochs = 15

# tf Graph input
X = tf.placeholder("float", [num_input])
Y = tf.placeholder("float", [num_classes])

# Store layers weight & bias
weights = {
    'h1': tf.Variable(tf.random_normal([num_input, n_hidden_1])),
}
biases = {
    'b1': tf.Variable(tf.random_normal([n_hidden_1])),
}
v = tf.constant(np.concatenate((np.ones(k), -1*np.ones(k)), axis=0).astype('float32'))
aa=1
# Create model
def neural_net(x):
    # Hidden fully connected layer with 2k neurons
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    # pooling non linearity
    layer_2 = tf.nn.leaky_relu(layer_1)
    # Output multiplication by constant v
    aa=1
    out_layer = tf.matmul(tf.transpose(v),layer_2)
    return out_layer

# Construct model
logits = neural_net(X)

# Define loss and optimizer
loss_op = tf.reduce_mean(tf.losses.hinge_loss(
    labels=Y,
    logits=logits))

optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
train_op = optimizer.minimize(loss_op)


# Initializing the variables
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)

    # Training cycle
    for epoch in range(training_epochs):
        avg_cost = 0.
        total_batch = int(x_train.shape(0)/batch_size)
        # Loop over all batches
        for i in range(total_batch):
            batch_x, batch_y = x_train[i*batch_size:(i+1)*batch_size], y_train[i*batch_size:(i+1)*batch_size]
            # Run optimization op (backprop) and cost op (to get loss value)
            _, c = sess.run([train_op, loss_op], feed_dict={X: batch_x,
                                                            Y: batch_y})
            # Compute average loss
            avg_cost += c / total_batch
        # Display logs per epoch step
        if epoch % display_step == 0:
            print("Epoch:", '%04d' % (epoch+1), "cost={:.9f}".format(avg_cost))
    print("Optimization Finished!")

    # Test model
    pred = tf.nn.softmax(logits)  # Apply softmax to logits
    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(Y, 1))
    # Calculate accuracy
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    print("Accuracy:", accuracy.eval({X: x_test, Y: y_test}))