import tensorflow as tf
import numpy as np


def create_model(configuration):
    # tf Graph input
    input_dimension = configuration['model']['input_dimension']
    num_classes = configuration['model']['number_of_classes']
    number_of_neurons_in_layers_dict = configuration['model']['number_of_neurons_in_layers']
    activation_type = configuration['model']['activation_type']
    loss_type = configuration['model']['loss_type']
    X = tf.placeholder("float", [input_dimension, None])
    Y = tf.placeholder("float", [num_classes, None])
    learning_rate = tf.placeholder("float", shape=[])

    layers_keys = list(number_of_neurons_in_layers_dict.keys())
    weights_layers_sizes = {layers_keys[0]: [number_of_neurons_in_layers_dict[layers_keys[0]], input_dimension]}
    previous_key = layers_keys[0]
    for current_key in number_of_neurons_in_layers_dict.keys():
        if current_key not in weights_layers_sizes.keys():
            weights_layers_sizes[current_key] = [number_of_neurons_in_layers_dict[current_key],
                                                 number_of_neurons_in_layers_dict[previous_key]]
        previous_key = current_key
    # Store layers weight & bias
    weights = {}
    for key in weights_layers_sizes.keys():
        weights[key] = tf.Variable(tf.random_normal(weights_layers_sizes[key]))

    # Create model
    def neural_net_N_layers(x):
        # Hidden fully connected layer with 2k neurons
        current_features = x
        for key in weights.keys():
            matrix_multiply = tf.matmul(weights[key], current_features)
            # pooling non linearity
            if activation_type == 'leaky':
                current_features = tf.nn.leaky_relu(matrix_multiply)
            elif activation_type == 'relu':
                current_features = tf.nn.relu(matrix_multiply)
            else:
                current_features = matrix_multiply
        # Hidden second layer
        return current_features

    # Construct model
    logits = neural_net_N_layers(X)
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
    # gradients_varaibles_list = optimizer.compute_gradients(loss_op)
    # gradients_values_list = [gradient_variable[0] for gradient_variable in gradients_varaibles_list]
    # train_op = optimizer.apply_gradients(gradients_varaibles_list)
    train_op = optimizer.minimize(loss_op)
    return loss_op, train_op, logits, X, Y, learning_rate, weights


def measure_model(x_train, y_train, x_test, y_test, configuration):
    loss_op, train_op, logits, X, Y, learning_rate, weights = create_model(configuration)
    var_grad = tf.gradients(loss_op,list(weights.values()))
    training_epochs = configuration["model"]["number_of_epochs"]
    batch_size = configuration["model"]["batch_size"]
    # Test model
    avg_loss = []
    train_error_results = []
    test_error_results = []
    weights_between_epochs = []
    gradients_between_epochs =[]
    # Initializing the variables
    init = tf.global_variables_initializer()
    # TODO: check the loss function
    zero_one_loss = tf.not_equal(tf.sign(logits), Y)
    # zero_one_loss = tf.not_equal(logits, Y)
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
                k = np.random.randint(x_train.shape[0])
                batch_x, batch_y = x_train[k * batch_size:(k + 1) * batch_size], y_train[
                                                                                 k * batch_size:(k + 1) * batch_size]
                # Run optimization op (backprop) and cost op (to get loss value)

                _, cost, weights_values = sess.run([train_op, loss_op, weights],
                                                   feed_dict={X: np.transpose(batch_x),
                                                              Y: np.transpose(batch_y),
                                                              learning_rate: current_learning_rate})
                gradients_variables = sess.run(var_grad,feed_dict={X: np.transpose(batch_x),
                                                              Y: np.transpose(batch_y),
                                                              learning_rate: current_learning_rate})
                aa=1
                # gradients_values_list = [gradient_variable[0] for gradient_variable in gradients_variables]
                # Compute average loss
                avg_cost += cost / total_batch
            if configuration['model']['decreasing_learning_rate']:
                # beta = 1
                # G = 2
                # r = np.max([np.linalg.norm(weights_val, ord='fro') for weights_val in list(weights_values.values())])
                # number_of_layers = len(list(weights_values.values()))
                # beta_r = 2 * (number_of_layers ** 2) * (r ** (2 * number_of_layers - 2)) * (beta + G)
                # #
                # current_learning_rate = min([1, 1 / (beta_r)]) / 2
                if int(epoch) % 100 == 0:
                    current_learning_rate *= 0.5
                    print(current_learning_rate)
            else:
                current_learning_rate = configuration['model']['learning_rate']
            train_error = sess.run(zeros_one_loss_mean, feed_dict={X: np.transpose(x_train), Y: np.transpose(y_train)})
            print('train_error:', train_error)
            test_error = sess.run(zeros_one_loss_mean, feed_dict={X: np.transpose(x_test), Y: np.transpose(y_test)})
            print('test_error:', test_error)
            train_error_results.append(train_error)
            test_error_results.append(test_error)
            weights_between_epochs.append(weights_values)
            gradients_dict = dict(zip(weights_values.keys(), gradients_variables))
            gradients_between_epochs.append(gradients_dict)
            # Display logs per epoch step
            avg_loss.append(avg_cost)
            print("Epoch:", '%04d' % (epoch + 1), "cost={:.9f}".format(avg_cost))
        print("Optimization Finished!")
        print("Accuracy:", zeros_one_loss_mean.eval({X: np.transpose(x_test), Y: np.transpose(y_test)}))
        var_grad =[]
        return np.array(train_error_results), np.array(test_error_results), np.array(avg_loss), weights_between_epochs,gradients_between_epochs
