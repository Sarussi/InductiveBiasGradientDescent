import tensorflow as tf
import numpy as np


def create_model(configuration):
    # tf Graph input
    input_dimension = configuration['model']['input_dimension']
    num_classes = configuration['model']['number_of_classes']
    number_of_neurons_first_layer = configuration['model']['number_of_neurons_first_layer']
    number_of_neurons_second_layer = configuration['model']['number_of_neurons_second_layer']
    learning_rate = configuration['model']['learning_rate']
    activation_type = configuration['model']['activation_type']
    X = tf.placeholder("float", [input_dimension, None])
    Y = tf.placeholder("float", [num_classes, None])

    # Store layers weight & bias
    weights = {
        'h1': tf.Variable(tf.random_normal([number_of_neurons_first_layer, input_dimension])),
        'h2': tf.Variable(tf.random_normal([2 * number_of_neurons_second_layer, number_of_neurons_first_layer]))
    }
    v = tf.constant(
        (1 / np.sqrt(2 * number_of_neurons_second_layer)) * np.concatenate(
            (np.ones(number_of_neurons_second_layer), -1 * np.ones(number_of_neurons_second_layer)), axis=0).astype(
            'float32').reshape(
            [1, 2 * number_of_neurons_second_layer]))

    # Create model
    def neural_net_two_layers(x):
        # Hidden fully connected layer with 2k neurons
        layer_1 = tf.matmul(weights['h1'], x)
        # pooling non linearity
        if activation_type == 'leaky':
            layer_2 = tf.nn.leaky_relu(layer_1)
        else:
            layer_2 = tf.nn.relu(layer_1)
        # Hidden second layer
        layer_3 = tf.matmul(weights['h2'], layer_2)
        # pooling non linearity
        if activation_type == 'leaky':
            layer_4 = tf.nn.leaky_relu(layer_3)
        else:
            layer_4 = tf.nn.relu(layer_3)

        # Output multiplication by constant v
        out_layer = tf.matmul(v, layer_4)
        return out_layer

    # Construct model
    logits = neural_net_two_layers(X)
    # Define loss and optimizer
    loss_op = tf.reduce_mean(tf.losses.hinge_loss(
        labels=Y,
        logits=logits))

    optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
    train_op = optimizer.minimize(loss_op)
    return loss_op, train_op, logits, X, Y, weights


def measure_model(x_train, y_train, x_test, y_test, configuration):
    loss_op, train_op, logits, X, Y, weights = create_model(configuration)
    training_epochs = configuration["model"]["number_of_epochs"]
    batch_size = configuration["model"]["batch_size"]
    # Test model
    avg_loss = []
    train_error_results = []
    test_error_results = []
    # Initializing the variables
    init = tf.global_variables_initializer()
    # TODO: check the loss function
    zero_one_loss = tf.not_equal(tf.sign(logits), Y)
    # zero_one_loss = tf.not_equal(logits, Y)
    # Calculate accuracy
    zeros_one_loss_mean = tf.reduce_mean(tf.cast(zero_one_loss, "float"))

    with tf.Session() as sess:
        sess.run(init)
        sess.run(tf.local_variables_initializer())

        # Training cycle
        for epoch in range(training_epochs):
            avg_cost = 0.
            total_batch = int(x_train.shape[0] / batch_size)
            # Loop over all batches
            for i in range(total_batch):
                k = np.random.randint(x_train.shape[0])
                batch_x, batch_y = x_train[k * batch_size:(k + 1) * batch_size], y_train[
                                                                                 k * batch_size:(k + 1) * batch_size]
                # Run optimization op (backprop) and cost op (to get loss value)
                _, cost = sess.run([train_op, loss_op], feed_dict={X: np.transpose(batch_x),
                                                                   Y: np.transpose(batch_y)})

                # Compute average loss
                avg_cost += cost / total_batch

            train_error = sess.run(zeros_one_loss_mean, feed_dict={X: np.transpose(x_train), Y: np.transpose(y_train)})
            print('train_error:', train_error)
            test_error = sess.run(zeros_one_loss_mean, feed_dict={X: np.transpose(x_test), Y: np.transpose(y_test)})
            print('test_error:', test_error)
            train_error_results.append(train_error)
            test_error_results.append(test_error)

            # Display logs per epoch step
            avg_loss.append(avg_cost)
            print("Epoch:", '%04d' % (epoch + 1), "cost={:.9f}".format(avg_cost))
        print("Optimization Finished!")
        weights_values = {}
        for layer_key in weights.keys():
            weights_values[layer_key] = sess.run(weights[layer_key])
        print("Accuracy:", zeros_one_loss_mean.eval({X: np.transpose(x_test), Y: np.transpose(y_test)}))
        return np.array(train_error_results), np.array(test_error_results), np.array(avg_loss), weights_values

