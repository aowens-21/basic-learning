from random import random, choice
import numpy as np

# XOR function will just be a tuple of tuple pairs
BOOLEAN_XOR = ((np.array([0, 0]), 0), (np.array([0, 1]), 1), (np.array([1, 0]), 1), (np.array([1, 1]), 0))

def compute_excitation_vec(x, l1_weights, l1_biases):
    """ This function takes in the layer 1 weights and biases and
    then computes the first layer exciations.
    
    Returns: a numpy array with the excitation of both neurons. """
    first_exc = l1_weights[0][0]*x[0] + l1_weights[1][0]*x[1] + l1_biases[0]
    second_exc = l1_weights[0][1]*x[0] + l1_weights[1][1]*x[1] + l1_biases[1]
    return np.array([first_exc, second_exc])

def sigmoid(exc):
    """ Computes sigmoid function on a given excitation.
    
    Returns: a floating point value representing neuron output. """
    return 1 / (1 + np.exp(-exc))

def compute_l1_outputs(exc_vec):
    """ Is passed an excitation vector for the 2 neuron layer and computes the output
    of each neuron. 
    
    Returns: a numpy array with output of both neurons. """
    out1 = sigmoid(exc_vec[0])
    out2 = sigmoid(exc_vec[1])
    return np.array([out1, out2])

def compute_excitation(z, l2_weights, l2_bias):
    """ This function takes in the final layer weights and bias and
    computes the final neuron's excitation.
    
    Returns: a floating point value representing neuron excitation. """
    return z[0]*l2_weights[0] + z[1]*l2_weights[1] + l2_bias

def compute_output(exc):
    """ Just a wrapper for the sigmoid function for the output of the final
    neuron, arguably unnecessary, I just added this for consistency in the
    code.
    
    Returns: a floating point value representing neuron output. """
    return sigmoid(exc)

def compute_loss(output, expected):
    """ Takes in an output and expected output and computes loss.
    
    Returns: a floating point value representing neuron's loss. """
    return ((output - expected) ** 2) / 2

def compute_network_loss(neuron, l1_weights, l1_biases, l2_weights, l2_bias):
    """ This function takes the entire network's information (weights and biases)
    and computes the loss for the whole network on a given neuron.
    
    Returns: a foating point value representing neuron's loss. """
    # Decompose neuron into pair and expected value
    (input_pair, expected) = neuron
    
    # Compute the overall network output
    excitation_vec = compute_excitation_vec(input_pair, l1_weights, l1_biases)
    l1_outputs = compute_l1_outputs(excitation_vec)
    excitation = compute_excitation(l1_outputs, l2_weights, l2_bias)
    output = compute_output(excitation)
    
    # Use output and expected to compute loss
    loss = compute_loss(output, expected)
    return loss

def compute_l1_weight_gradients(output, expected, x, z, v):
    """ This function takes an output, expected value, inputs (x), outputs (z)
    and the next layer's weights (v), and with this information it computes
    the gradients for the layer's weight values.
    
    Returns: a 2x2 numpy array representing the gradient for each weight vector
    in the INITIAL layer. """
    # Each gradient is a 1x2 array since we have 2 neurons in this layer
    weight1_grads = [(output - expected) * output * (1 - output) * z[0] * (1 - z[0]) * v[0] * x[0],
                   (output - expected) * output * (1 - output) * z[1] * (1 - z[1]) * v[1] * x[0]]
    weight2_grads = [(output - expected) * output * (1 - output) * z[0] * (1 - z[0]) * v[0] * x[1],
               (output - expected) * output * (1 - output) * z[1] * (1 - z[1]) * v[1] * x[1]]
    return np.array([weight1_grads, weight2_grads])

def compute_l1_bias_gradients(output, expected, z, v):
    """ This function takes in an output, expected value, outputs (z) and the
    next layer's weights (v), and it computes the gradients for the layer's
    biases.
    
    Returns: a numpy array representing the gradient for each bias in
    the INITIAL layer. """
    # One bias for each neuron
    b1_grad = (output - expected) * output * (1 - output) * z[0] * (1 - z[0]) * v[0]
    b2_grad = (output - expected) * output * (1 - output) * z[1] * (1 - z[1]) * v[1]
    return np.array([b1_grad, b2_grad])

def compute_l2_weight_gradients(output, expected, z):
    """ This function takes in an output, expected value, and this layer's
    inputs (z), and uses them to compute the gradients for this layer's weights.
    
    Returns: a numpy array representing the gradient for each weight in the
    FINAL layer. """
    w1_grad = (output - expected) * output * (1 - output) * z[0]
    w2_grad = (output - expected) * output * (1 - output) * z[1]
    return np.array([w1_grad, w2_grad])

def compute_l2_bias_gradient(output, expected):
    """ This function takes in an output and expected value, and computes the
    gradients for this layer's bias.
    
    Returns: a floating point value representing the gradient for the FINAL
    layer's bias. """
    return (output - expected) * output * (1 - output)
    
def test_if_we_learned(l1_weights, l1_biases, l2_weights, l2_bias):
    """ Takes in each layers' weights and biases and tests them on the
    boolean function XOR to see if the neural network has succeeded in
    learning. 
    
    Returns: Nothing, but it prints the result of each test. """
    for pair in BOOLEAN_XOR:
        # Decompose our pair down to inputs and expected value
        (inputs, expected) = pair
        
        # Get overall output with the given weights and biases, no need for loss
        l1_exc_vec = compute_excitation_vec(inputs, l1_weights, l1_biases)
        l1_outputs = compute_l1_outputs(l1_exc_vec)
        excitation = compute_excitation(l1_outputs, l2_weights, l2_bias)
        output = compute_output(excitation)
        print("Expected: " + str(expected) + " Actual: " + str(output))
        

def learn_boolean_XOR(learning_rate=0.5, loss_goal=0.00001, max_iters=50000):
    """ This function iterates a certain number of times or until
    the loss goal for the steepest descent algorithm is met, in each iteration
    it computes new weights and biases for layers of a 2-2-1 neural network
    trying to learn the boolean XOR function. After the terminating condition
    has been met, it tests the weights and biases found to see if they are
    working. 
    
    Optional:
    learning_rate - multiplier we apply to gradient steps
    loss_goal - we want the loss to be this small before stopping
    max_iters - number of iterations the loop will do, use this especially
    if you're using a function you don't know is possible
    
    Returns: Nothing, but calls tests. """
    # First pick random weights and biases
    layer1_weights = np.array([[random(), random()],
                      [random(), random()]])
    layer1_biases = np.array([random(), random()])
    
    layer2_weights = np.array([random(), random()])
    layer2_bias = random()
    
    # Set up our looping variables
    iters = 0
    continue_learning = True
    
    while (continue_learning):
        # Choose a random pair from the BOOLEAN_XOR representation
        neuron = choice(BOOLEAN_XOR)
        (inputs, expected) = neuron
        
        # Compute the network loss
        loss = compute_network_loss(neuron, layer1_weights, layer1_biases, layer2_weights, layer2_bias)
        
        #Compute the output of the entire network, first layer 1 then layer2
        l1_exc_vec = compute_excitation_vec(inputs, layer1_weights, layer1_biases)
        l1_outputs = compute_l1_outputs(l1_exc_vec)
        excitation = compute_excitation(l1_outputs, layer2_weights, layer2_bias)
        # The overall output
        output = compute_output(excitation)
        
        # Using the output compute layer 1's gradients
        l1_w_grads = compute_l1_weight_gradients(output, expected, inputs, l1_outputs, layer2_weights)
        l1_b_grads = compute_l1_bias_gradients(output, expected, l1_outputs, layer2_weights)
        
        # Do the same for layer 2
        l2_w_grads = compute_l2_weight_gradients(output, expected, l1_outputs)
        l2_b_grad = compute_l2_bias_gradient(output, expected)
        
        # Apply gradient step to both layers, with learning_rate applied
        layer1_weights -= learning_rate * l1_w_grads
        layer1_biases -= learning_rate * l1_b_grads
        layer2_weights -= learning_rate * l2_w_grads
        layer2_bias -= learning_rate * l2_b_grad
        
        # Check to see if we've met termination condition
        if loss < loss_goal or iters > max_iters:
            continue_learning = False
        
    # Test to see if the weights and biases worked
    test_if_we_learned(layer1_weights, layer1_biases, layer2_weights, layer2_bias)

# A little test
learn_boolean_XOR()