import tensorflow as tf

from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt

# Download and prepare the CIFAR10 dataset which contains 60k images in 10 classes
(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()

