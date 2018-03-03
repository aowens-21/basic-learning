# basic-learning
A few python examples of the [steepest descent](https://en.wikipedia.org/wiki/Gradient_descent) algorithm to learn 
boolean functions with (almost) pure python

## One Neuron
The one neuron learning example just repeatedly pushes input and expected output into one neuron and computes gradients
using [backpropogation](https://en.wikipedia.org/wiki/Backpropagation) which it then uses to find the correct weights for 
the inputs. Doing this, it can learn boolean AND and OR.

## Simple Network
This is a very simple neural network learning structure, working in a very similar way to the One Neuron example. It instead pushes
inputs through a 2 neuron layer whose outputs get pushed into a second layer with only one neuron which gives us our output. The
power in this is that the calculus used to get the gradient calculation formulas stays essentially the same up to N layers. Because
this is a network with more than one neuron, we can use it to learn the boolean XOR.

### Firing Function
As of right now these examples only support the [Sigmoid](https://en.wikipedia.org/wiki/Sigmoid_function) firing function.
