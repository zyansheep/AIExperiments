import tensorflow as tf
import numpy as np
mnist = tf.keras.datasets.mnist
from mlxtend.data import loadlocal_mnist
import matplotlib.pyplot as plt

def shift(a, dx, dy):
    r = np.roll(a, int(dy), axis=0)
    r = np.roll(r, int(dx), axis=1)
    return r


def center(a):
    ymax, xmax = np.max(np.where(a != 0), 1)
    ymin, xmin = np.min(np.where(a != 0), 1)
    mx = (xmin + xmax) / 2
    my = (ymin + ymax) / 2
    sx = int((len(a) / 2) - mx)
    sy = int((len(a) / 2) - my)
    return shift(a, sx, sy)

x_train,y_train = loadlocal_mnist(
    images_path='Training/Emnist/ByClass/train-images',
    labels_path='Training/Emnist/ByClass/train-labels')
x_test,y_test = loadlocal_mnist(
    images_path='Training/Emnist/ByClass/test-images',
    labels_path='Training/Emnist/ByClass/test-labels')
print("Loaded Training Data")

#(mx_train, my_train),(mx_test, my_test) = mnist.load_data()
x_train, x_test = np.round(x_train / 255.0), np.round(x_test / 255.0)

x_train = x_train.reshape((-1,28,28))
x_test = x_test.reshape((-1,28,28))

#preprocessing to make sure the images are right side up
x_train = np.rot90(x_train, 1, (2,1))
x_train = np.flip(x_train, 2)
x_test = np.rot90(x_test, 1, (2,1))
x_test = np.flip(x_test, 2)

'''for i in range(len(x_train)):
    x_train[i] = center(x_train[i])'''

print("Done Preprocessing")

'''print(x_train.shape)
for i in range(100):
    print("Label: ", y_train[i])
    plt.imshow(x_train[i], cmap="gray")
    plt.savefig("imgs/img"+str(y_train[i])+".png")'''

model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(512, activation=tf.nn.relu),
    tf.keras.layers.Dropout(0.05),
    tf.keras.layers.Dense(256, activation=tf.nn.relu),
    tf.keras.layers.Dropout(0.1),
    tf.keras.layers.Dense(128, activation=tf.nn.relu),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(62, activation=tf.nn.softmax)
])
#Model Stucture
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

#Model train (inputs, supposed outputs, num times to train)
model.fit(x_train, y_train, epochs=5, batch_size=100)
#Test the network
model.evaluate(x_test, y_test)

filename = "model"
model_json = model.to_json()
with open("models/"+filename+"model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("models/"+filename+".h5")