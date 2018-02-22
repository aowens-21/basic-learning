import numpy as np
import random

# Two test boolean functions that are possible to "learn" with this algorithm
BOOLEAN_AND = ((np.array([0, 0]), 0), (np.array([0, 1]), 0), (np.array([1, 0]), 0), (np.array([1, 1]), 1))
BOOLEAN_OR = ((np.array([0, 0]), 0), (np.array([0, 1]), 1), (np.array([1, 0]), 1), (np.array([1, 1]), 1))

def sigmoid(excitation):
    return 1 / (1 + np.exp(-excitation))

def compute_output(weight_vec, input_vec, bias_vec):
    """ Returns neuron's actual output with given inputs, weights, and bias. """
    return sigmoid(compute_excitation(weight_vec, input_vec, bias_vec))

def compute_excitation(weight_vec, input_vec, bias_vec):
    """ Returns neuron excitation. """
    return weight_vec@input_vec + bias_vec

def compute_gradient(output, expected_output, input_vec=1):
    """ Computes the gradient vector (or scalar if you don't pass a vector) based
    on output and expected output and returns it.
    
    Optional Parameters:
    input_vec - This is an input vector that we multiply by the computation which
    gives us the weight gradient vector. If this isn't provided, we get a 
    scalar which can be used to instead calculate bias gradient.
    """
    return (output - expected_output) * output * (1 - output) * input_vec

def compute_loss(output, expected_output):
    """ Computes a neuron's loss based on output and expected output."""
    return ((output - expected_output) ** 2) / 2

def output_learning_results(func_learned, weight_vec, bias_vec, iterations):
    """ Outputs the results of running a boolean function through a neuron after
    having learned the function's proper weights and bias, also outputs number
    of iterations it took to learn."""
    for func_pair in func_learned:
        # Just compute with weights and bias the output for the function pair
        (input_vec, expected_output) = func_pair
        output = compute_output(weight_vec, input_vec, bias_vec)
        print("Expected: " + str(expected_output) + " Actual: " + str(output))    
    print("Number of Iterations: " + str(iterations))

def learn_boolean_function(bool_func, loss_goal=0.00001, learning_rate=0.5, max_iters=0):
    """ This function takes in a function and tries to learn the desired outputs
    using the steepest descent algorithm.
    
    Parameters:
    bool_func - This will be a tuple of tuples representing the two inputs (a numpy array) and
    the expected output for those inputs. These represent the function definition.
    
    loss_goal - This will be a very small number that our neuron's loss much reach
    before we stop iteratively trying to learn the boolean function.
    
    learning_rate - the rate at which we apply our gradients to the weights and
    bias for the neuron, also a number but preferably not super small if you're
    aiming for a low loss goal.
    
    max_iters - Sometimes you won't be sure if a boolean function can be learned, so
    you'll want to cap the number of iterations somewhere or the loss will never shrink
    enough to break out. Defaults to 0, which means we don't apply it.
    """
    # Generate an initial weight vector and bias
    weight_vec = np.random.randn(1, 2)
    bias_vec = np.random.randn(1, 1)
    
    # This will count the number of iterations
    iters = 0
    
    # This is our loop condition
    continue_learning = True
    
    while (continue_learning):
        # Get a random choice from our function to compute
        (input_vec, expected_output) = random.choice(bool_func)
        
        output = compute_output(weight_vec, input_vec, bias_vec)
        loss = compute_loss(output, expected_output)
        weight_gradients = compute_gradient(output, expected_output, input_vec)
        bias_gradient = compute_gradient(output, expected_output)
        
        # Apply our gradients to weights and bias, with learning rate
        weight_vec -= learning_rate * weight_gradients
        bias_vec -= learning_rate * bias_gradient
        
        iters += 1
        
        # Stop iterating if we've met our loss goal        
        continue_learning = False if (loss <= loss_goal) else True
        
        # Also stop iterating if we have a max iters and have reached it
        if (max_iters != 0):
            continue_learning = False if (iters >= max_iters) else True
    
    output_learning_results(bool_func, weight_vec, bias_vec, iters)
    
# Test our learning function on AND/OR
learn_boolean_function(bool_func=BOOLEAN_AND, loss_goal=0.0000000001)
learn_boolean_function(bool_func=BOOLEAN_OR, loss_goal=0.0000000001)