
from input_parser import MNISTDataProvider
from configuration import configuration_parameters
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
# mnist_data = MNISTDataProvider()

N=1000
d=784
train_test_split_ration = 0.75
data_provider = configuration_parameters["data_provider"]
x_data, y_data = data_provider.read(N,d)
x_data, y_data = shuffle(x_data,y_data)

x_train = x_data[0:int(x_data.shape[0]*train_test_split_ration),:]
y_train = y_data[0:int(x_data.shape[0]*train_test_split_ration)]
y_test = y_data[int(x_data.shape[0]*train_test_split_ration):int(x_data.shape[0])]
x_test = x_data[int(x_data.shape[0]*train_test_split_ration):int(x_data.shape[0]),:]


# Parameters
learning_rate = 0.0001
num_steps = 500
batch_size = 1
display_step = 100
k=10
# Network Parameters
n_hidden_1 = 2*k # 1st layer number of neurons
num_input = 784 # MNIST data input (img shape: 28*28)
num_classes = 1 # MNIST total classes (0-9 digits)
training_epochs = 100

# tf Graph input
X = tf.placeholder("float", [num_input, None])
Y = tf.placeholder("float", [num_classes, None])

# Store layers weight & bias
weights = {
    'h1': tf.Variable(tf.random_normal([n_hidden_1, num_input])),
}
v = tf.constant(np.concatenate((np.ones(k), -1*np.ones(k)), axis=0).astype('float32').reshape([1,n_hidden_1]))
# Create model
def neural_net(x):
    # Hidden fully connected layer with 2k neurons
    layer_1 = tf.matmul(weights['h1'], x)
    # pooling non linearity
    layer_2 = tf.nn.leaky_relu(layer_1)
    # Output multiplication by constant v
    out_layer = tf.matmul(v,layer_2)
    return out_layer

# Construct model
logits = neural_net(X)
# Define loss and optimizer
loss_op = tf.reduce_mean(tf.losses.hinge_loss(
    labels=Y,
    logits=logits))

optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
train_op = optimizer.minimize(loss_op)
accuracy_epoch_array = np.zeros(shape=(1,training_epochs)).reshape(training_epochs,1)


# Test model
train_loss_results=[]
train_accuracy_results=[]
test_accuracy_results=[]
# Initializing the variables
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)

    # Training cycle
    for epoch in range(training_epochs):
        avg_cost = 0.
        total_batch = int(x_train.shape[0]/batch_size)
        # Loop over all batches
        for i in range(total_batch):
            batch_x, batch_y = x_train[i*batch_size:(i+1)*batch_size], y_train[i*batch_size:(i+1)*batch_size]
            # Run optimization op (backprop) and cost op (to get loss value)
            batch_x = np.transpose(batch_x)
            _, c = sess.run([train_op, loss_op], feed_dict={X: batch_x,
                                                            Y: batch_y})
            # Compute average loss
            avg_cost += c / total_batch
        # train_accuracy_results.append(tf.metrics.accuracy(logits(x_train),y_train))
        # test_accuracy_results.append(tf.metrics.accuracy(logits(x_test),y_test))

        # Display logs per epoch step
        accuracy_epoch_array[epoch] = avg_cost
        print("Epoch:", '%04d' % (epoch+1), "cost={:.9f}".format(avg_cost))
    print("Optimization Finished!")


    pred = logits
    correct_prediction = tf.equal(pred, Y)
    # Calculate accuracy
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    print("Accuracy:", accuracy.eval({X: np.transpose(x_test), Y: np.transpose(y_test)}))
    plt.scatter(range(training_epochs), accuracy_epoch_array)
    plt.show()