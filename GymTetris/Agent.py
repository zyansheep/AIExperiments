import tensorflow as tf
import numpy as np
import random
from collections import deque

class Agent:
    memory = []
    
    def __init__(self, input_shape, output_shape, filename=""):
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
        
        if not filename:
            #Keras Model
            self.model = tf.keras.Sequential([
                #tf.keras.layers.Flatten(batch_input_shape=input_shape),
                tf.keras.layers.Dense(128, input_shape=input_shape, activation='relu'),
                tf.keras.layers.Dense(64, activation='relu'),
                tf.keras.layers.Dropout(0.25),
                tf.keras.layers.Dense(output_shape, activation='linear'),  # output layer
            ])

            self.model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(lr=self.learnRate))
            self.model.summary()
        else:
            self.load(filename)

    #Collect states (training data)
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        #If network just started out, move randomly and see what works
        '''if np.random.rand() <= self.exploreRate:
            return random.randrange(self.output_shape)'''
            
        if type(state) is list:
            state = np.array(state)
        
        print("State shape: ", state.shape)
        print("Acting on state ", state)
        action_weights = self.model.predict(state)
        return action_weights
        #return np.argmax(action_weights[0])

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
    
    def save(self, filename):
        model_json = self.model.to_json()
        with open("models/"+filename+".json", "w") as json_file:
            json_file.write(model_json)
        # serialize weights to HDF5
        self.model.save_weights("models/"+filename+".h5")
    
    def load(self, filename):
        json_file = open("models/"+filename+'.json', 'r')
        self.model = tf.keras.models.model_from_json(json_file.read())
        json_file.close()
        self.model.load_weights("models/"+filename+".h5")
        print("Loaded model from disk")
    