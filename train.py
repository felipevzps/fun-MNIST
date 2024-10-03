import load
import model
import config
import numpy as np
import argparse

def load_and_prepare_data(images_path, labels_path, split_ratio=0.1):
    """
    Load dataset, normalize, and split into training and validation sets.
    
    Args:
        images_path (str): Path to the images file.
        labels_path (str): Path to the labels file.
        split_ratio (float): Proportion of the data to be used as validation set.
        
    Returns:
        X_train (np.ndarray): Training data.
        Y_train (np.ndarray): Training labels.
        X_eval (np.ndarray): Validation data.
        Y_eval (np.ndarray): Validation labels.
    """
    # Load data
    images = load.load_mnist_images(images_path)
    labels = load.load_mnist_labels(labels_path)
    
    # Combine labels and images
    data = np.column_stack((labels, images))
    np.random.shuffle(data)
    
    # Split into training and dev sets
    m = data.shape[0]
    eval_size = int(m * split_ratio)
    
    data_eval = data[:eval_size].T
    Y_dev = data_eval[0]
    X_dev = data_eval[1:] / 255.           # Normalize
    
    data_train = data[eval_size:].T
    Y_train = data_train[0]
    X_train = data_train[1:] / 255.        # Normalize
    
    return X_train, Y_train, X_dev, Y_dev

def train_neural_network(X_train, Y_train, learning_rate, iterations, model_save_path=None):
    """
    Train the neural network using gradient descent.
    
    Args:
        X_train (np.ndarray): Training data.
        Y_train (np.ndarray): Training labels.
        learning_rate (float): Learning rate for gradient descent.
        iterations (int): Number of iterations for training.
        model_save_path (str): Path to save the trained model parameters.
    
    Returns:
        W1, b1, W2, b2 (np.ndarray): Trained weights and biases.
    """
    m = X_train.shape[1]
    
    # Train model
    W1, b1, W2, b2 = model.gradient_descent(X_train, Y_train, m, learning_rate, iterations)
    
    if model_save_path:
        np.savez(model_save_path, W1=W1, b1=b1, W2=W2, b2=b2)
        print(f"Model parameters saved to {model_save_path}")
    
    return W1, b1, W2, b2

def main():
    parser = argparse.ArgumentParser(prog='train.py', description='Train neural network using different datasets, learning rates, and iterations', add_help=True)
    parser.add_argument('-dataset', dest='dataset', metavar='mnist', help='dataset name', type=str, required=True)
    parser.add_argument('-lr', '--learning_rate', dest='learning_rate', metavar=0.1, help='training learning rate', type=float, required=True)
    parser.add_argument('-i', '--iterations', dest='iterations', metavar=1000, help='training iterations', type=int, required=True)
    args = parser.parse_args()

    dataset = config.configs[args.dataset]
    learning_rate = args.learning_rate
    iterations = args.iterations

    X_train, Y_train, X_dev, Y_dev = load_and_prepare_data(dataset['images'], dataset['labels'])
    W1, b1, W2, b2 = train_neural_network(X_train, Y_train, learning_rate, iterations, dataset['model'])

if __name__ == "__main__":
    main()