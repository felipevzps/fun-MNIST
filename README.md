# MNIST Digit Prediction with Simple Neural Network
This project implements a basic neural network to classify digits from the [MNIST database](https://yann.lecun.com/exdb/mnist/). 
>The model consists of two layers and was built from scratch without using any machine learning framework.

## Model Structure
Layer 1: A fully connected layer with 10 neurons, taking in input vectors of 784 features (28x28 pixels).

Layer 2: Another fully connected layer with 10 neurons, representing the 10 digit classes (0-9).

## Parameter Initialization
The weights and biases for both layers are initialized randomly with small values. This ensures that the model starts with a diverse set of parameters, which is crucial for effective learning during training.

## Forward Propagation
Data passes through the network in two stages: first, a linear combination of inputs and weights is calculated, followed by a ReLU activation in the first layer. The second layer uses the softmax function to output probabilities for each digit class.

## Activation Function
The ReLU (Rectified Linear Unit) activation function is used in the first layer to introduce non-linearity, enabling the network to learn more complex patterns. In the final layer, softmax converts the raw scores into a probability distribution.

## Backpropagation
Backpropagation is employed to adjust the weights and biases by computing the gradient of the loss with respect to each parameter. The model uses these gradients to update the parameters and reduce the overall prediction error.

## Optimization
The model is optimized using gradient descent, with a learning rate that controls the speed of convergence. The training is run for a set number of iterations to progressively minimize the loss.

## Results
After training for 1000 iterations, the model achieved an accuracy of 87.2% on the training dataset and 85.1% on the validation set.

## Possible Improvements
* Experiment with different hyperparameters (learning rate, number of neurons).
* Apply regularization techniques to avoid overfitting.
* Test alternative loss functions for potentially better convergence.
* Explore deeper architectures by adding more layers for improved accuracy.
