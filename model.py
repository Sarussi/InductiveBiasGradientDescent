import tensorflow as tf
import numpy as np


def create_model(configuration):
    # tf Graph input
    input_dimension = configuration['model']['input_dimension']
    num_classes = configuration['model']['number_of_classes']
    number_of_neurons = configuration['model']['number_of_neurons']
    # learning_rate = configuration['model']['learning_rate']
    activation_type = configuration['model']['activation_type']
    number_of_non_random_neurons_initialization = configuration['model']['number_of_non_random_neurons_initialization']
    loss_type = configuration['model']['loss_type']
    X = tf.placeholder("float", [input_dimension, None])
    Y = tf.placeholder("float", [num_classes, None])
    learning_rate = tf.placeholder("float", shape=[])

    # Store layers weight & bias
    if number_of_non_random_neurons_initialization != 0:
        all_indices = np.arange(0, 2 * number_of_neurons)
        non_random_initialization_indices = []
        while len(non_random_initialization_indices) < number_of_non_random_neurons_initialization:
            temp_index = np.random.randint(low=0, high=2 * number_of_neurons)
            if temp_index not in non_random_initialization_indices: non_random_initialization_indices.append(temp_index)
        non_random_initialization_indices = np.array(non_random_initialization_indices)
        random_initialization_indices = [index for index in all_indices if
                                         index not in non_random_initialization_indices]
        weights = {
            'h1': tf.Variable(tf.concat([np.eye(2 * number_of_neurons, input_dimension).astype('float32')[
                                         non_random_initialization_indices, :],
                                         np.random.randn(2 * number_of_neurons,
                                                         input_dimension).astype('float32')[
                                         random_initialization_indices, :]], axis=0))
        }
    else:
        weights = {
            'h1': tf.Variable(tf.concat([tf.random_normal(
                [2 * number_of_neurons, input_dimension])], axis=0))
        }
    if configuration['data']['data_provider'].__class__.__name__ == 'OrthogonalSingleClassDataProvider':
        v = tf.constant(
            np.concatenate(
                (np.ones(number_of_neurons), -1 * np.ones(number_of_neurons)), axis=0).astype('float32').reshape(
                [1, 2 * number_of_neurons]))
    else:
        v = tf.constant(
            (1 / np.sqrt(2 * number_of_neurons)) * np.concatenate(
                (np.ones(number_of_neurons), -1 * np.ones(number_of_neurons)), axis=0).astype('float32').reshape(
                [1, 2 * number_of_neurons]))

    # Create model
    def neural_net(x):
        # Hidden fully connected layer with 2k neurons
        layer_1 = tf.matmul(weights['h1'], x)
        # pooling non linearity
        if activation_type == 'leaky':
            layer_2 = tf.nn.leaky_relu(layer_1)
        else:
            layer_2 = tf.nn.relu(layer_1)
        # Output multiplication by constant v
        out_layer = tf.matmul(v, layer_2)
        return out_layer

    # Construct model
    logits = neural_net(X)
    # Define loss and optimizer
    if loss_type == 'logistic':
        loss_op = tf.reduce_mean(tf.log(1 + tf.exp(-Y * logits)))
    elif loss_type == 'exponential':
        loss_op = tf.reduce_mean(tf.exp(-Y * logits))
    else:
        loss_op = tf.reduce_mean(tf.losses.hinge_loss(
            labels=Y,
            logits=logits))
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
    train_op = optimizer.minimize(loss_op)
    return loss_op, train_op, logits, X, Y, learning_rate, weights


def measure_model_average(x_train, y_train, x_test, y_test, configuration):
    train_error_results = np.zeros(configuration["model"]["number_of_epochs"])
    test_error_results = np.zeros(configuration["model"]["number_of_epochs"])
    avg_loss = np.zeros(configuration["model"]["number_of_epochs"])
    for i in range(configuration["model"]["number_of_runs"]):
        train_error_results_temp, test_error_results_temp, avg_loss_temp, _ = measure_model(x_train, y_train, x_test,
                                                                                            y_test,
                                                                                            configuration)
        train_error_results += train_error_results_temp
        test_error_results += test_error_results_temp
        avg_loss += avg_loss_temp
    train_error_results /= configuration["model"]["number_of_runs"]
    test_error_results /= configuration["model"]["number_of_runs"]
    avg_loss /= configuration["model"]["number_of_runs"]
    return train_error_results, test_error_results, avg_loss


def measure_model(x_train, y_train, x_test, y_test, configuration):
    loss_op, train_op, logits, X, Y, learning_rate, weights = create_model(configuration)
    training_epochs = configuration["model"]["number_of_epochs"]
    batch_size = configuration["model"]["batch_size"]
    # Test model
    avg_loss = []
    train_error_epochs = []
    test_error_epochs = []
    weights_values_epochs = []
    # Initializing the variables
    init = tf.global_variables_initializer()

    zero_one_loss = tf.not_equal(tf.sign(logits), Y)
    # Calculate accuracy
    zeros_one_loss_mean = tf.reduce_mean(tf.cast(zero_one_loss, "float"))
    current_learning_rate = configuration['model']['learning_rate']
    with tf.Session() as sess:
        sess.run(init)
        sess.run(tf.local_variables_initializer())

        # Training cycle
        for epoch in range(training_epochs):

            avg_cost = 0.
            total_batch = int(x_train.shape[0] / batch_size)
            # Loop over all batches
            for i in range(total_batch):
                k = np.random.randint(total_batch)
                batch_x, batch_y = x_train[k * batch_size:(k + 1) * batch_size], y_train[
                                                                                 k * batch_size:(k + 1) * batch_size]
                # Run optimization op (backprop) and cost op (to get loss value)
                _, cost, weights_values = sess.run([train_op, loss_op, weights],
                                                   feed_dict={X: np.transpose(batch_x),
                                                              Y: np.transpose(batch_y),
                                                              learning_rate: current_learning_rate})

                # Compute average loss
                avg_cost += cost / total_batch
            if configuration['model']['decreasing_learning_rate']:
                beta = 1
                G= 2
                r = np.max([np.linalg.norm(weights_val,ord='fro') for weights_val in list(weights_values.values())])
                number_of_layers = len(list(weights_values.values()))
                beta_r = 2*(number_of_layers**2)*(r**(2*number_of_layers-2))*(beta+G)
                #
                current_learning_rate = min([1,1/(beta_r)])/2
                # if (float(epoch) / training_epochs) * 100 % 10 == 0:
                #     current_learning_rate *= 0.1
            else:
                current_learning_rate = configuration['model']['learning_rate']
            train_error = sess.run(zeros_one_loss_mean, feed_dict={X: np.transpose(x_train), Y: np.transpose(y_train)})
            print('train_error:', train_error)
            test_error = sess.run(zeros_one_loss_mean, feed_dict={X: np.transpose(x_test), Y: np.transpose(y_test)})
            print('test_error:', test_error)
            weights_values_epochs.append(weights_values)
            # weights_values = sess.run(weights['h1'])
            train_error_epochs.append(train_error)
            test_error_epochs.append(test_error)

            # Display logs per epoch step
            avg_loss.append(avg_cost)
            print("Epoch:", '%04d' % (epoch + 1), "cost={:.9f}".format(avg_cost))
        print("Optimization Finished!")
        print("Accuracy:", zeros_one_loss_mean.eval({X: np.transpose(x_test), Y: np.transpose(y_test)}))
        return np.array(train_error_epochs), np.array(test_error_epochs), np.array(avg_loss), weights_values_epochs
