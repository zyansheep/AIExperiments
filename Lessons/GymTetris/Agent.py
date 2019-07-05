import tensorflow as tf
import numpy as np
import random
from collections import deque

class Agent:
    memory = []


    def __init__(self, input_shape, output_shape):
        self.input_shape = input_shape
        self.output_shape = output_shape
        print(input_shape)
        self.memory = deque(maxlen=2000) #Reserve memory for nn memory

        #constants
        self.gamma = 0.95
        self.exploreRate = 0.9
        self.exploreMin = 0.01
        self.exploreDecay = 0.995
        self.learnRate = 0.001

        #Keras Model
        self.model = tf.keras.Sequential([
            tf.keras.layers.Flatten(input_shape=input_shape),
            tf.keras.layers.Dense(25, activation='relu'),
            tf.keras.layers.Dense(25, activation='relu'),
            tf.keras.layers.Dense(output_shape, activation='linear'),  # output layer
        ])

        self.model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(lr=self.learnRate));
        self.model.summary();

    #Collect states (training data)
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        #If network just started out, move randomly and see what works
        if np.random.rand() <= self.exploreRate:
            return random.randrange(self.output_shape)

        action_weights = self.model.predict(state)

        return np.argmax(action_weights[0])

    #train network by replaying memory
    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = reward + self.gamma * \
                         np.amax(self.model.predict(next_state)[0])
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
        if self.exploreRate > self.exploreMin:
            self.exploreRate *= self.exploreDecay
