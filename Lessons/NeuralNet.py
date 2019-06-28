import tensorflow as tf
import matplotlib.pyplot as plt
mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()

fig = plt.figure(figsize=(50,50))
for i in range(64):
    sub = fig.add_subplot(8, 8, i + 1)
    sub.imshow(x_train[i], interpolation='nearest')

plt.imshow(x_train[0])
plt.show()