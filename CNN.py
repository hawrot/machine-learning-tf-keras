import numpy as np
import tensorflow as tf

from sklearn.datasets import load_sample_image
import matplotlib.pyplot as plt

# Load sample images
china = load_sample_image("china.jpg") / 255
flower = load_sample_image("flower.jpg") / 255

images = np.array([china, flower])
#print(images.shape) # Check the shape | 2 , 427, 640, 3

batch_size, height, width, channels = images.shape

# Create two filters
filters = np.zeros(shape=(7,7, channels, 2), dtype=np.float32)
filters[:, 3, :, 0] = 1 # vertical line
filters[3, :, :, 1] = 1 # horizontal line

# Outputs
outputs = tf.nn.conv2d(images, filters, strides=1, padding="SAME") # Implementation without pooling layer
outputs2 = tf.nn.max_pool(images, ksize=(1,1,1,1), strides=(1,1,1,1), padding="VALID") # Implementation with pooling layer

# Plot
plt.imshow(outputs2[0, :, :, 1], cmap="gray") # plot 1st images's 2nd feature map
plt.show()