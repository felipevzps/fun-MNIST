import numpy as np
from load import load_mnist_images
from load import load_mnist_labels
from train import load_params
from train import make_predictions
from train import get_accuracy

test_images_path = "dataset/t10k-images.idx3-ubyte"
test_labels_path = "dataset/t10k-labels.idx1-ubyte"

test_images = load_mnist_images(test_images_path)
test_labels = load_mnist_labels(test_labels_path)

# add label to images array
data = np.column_stack((test_labels, test_images))
#print(data.shape)

m, n = data.shape

# splitting dataset in dev set (test set)
data_dev = data[0:1000].T
Y_dev = data_dev[0]
X_dev = data_dev[1:n]
X_dev = X_dev / 255.

W1, b1, W2, b2 = load_params('model/nn_parameters.npz')
dev_predictions = make_predictions(X_dev, W1, b1, W2, b2)
print(get_accuracy(dev_predictions, Y_dev))