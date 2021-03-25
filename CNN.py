import numpy as np
import tensorflow as tf
from sklearn.datasets import load_sample_image

# Load sample images
china = load_sample_image("china.jpg") / 255
flower = load_sample_image("flower.jpg") / 255

images = np.array([china, flower])
#print(images.shape) # Check the shape | 2 , 427, 640, 3

batch_size, height, width, channels = images.shape

# Create two filters

filers = np.zeros(shape=(7,7, channels, 2), dtype=np.float32)
filers2 = tf.zeros(shape=(7,7, channels, 2), dtype=np.float32)

print(filers)
print(filers2)
