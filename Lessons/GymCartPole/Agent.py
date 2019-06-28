import tensorflow as tf
import numpy as np

class Agent:
    def __init__(self, num_actions, state_size):
        initializer = tf.contrib.layers.xavier_initializer()

        # Neural net starts here (input shape size)
        self.input_layer = tf.placeholder(dtype=tf.float32, shape=[None, state_size])

        hidden_layer = tf.layers.dense(self.input_layer, 8, activation=tf.nn.relu, kernel_initializer=initializer)
        hidden_layer_2 = tf.layers.dense(hidden_layer, 8, activation=tf.nn.relu, kernel_initializer=initializer)


        out = tf.layers.dense(hidden_layer_2, num_actions, activation=None)
        # Neural network ends here (output action)

        self.outputs = tf.nn.softmax(out)
        self.choice = tf.argmax(self.outputs, axis=1)  #Get best action choice

        # Training Procedure
        self.rewards = tf.placeholder(shape=[None, ], dtype=tf.float32)
        self.actions = tf.placeholder(shape=[None, ], dtype=tf.int32)

        one_hot_actions = tf.one_hot(self.actions, num_actions)

        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=out, labels=one_hot_actions)

        self.loss = tf.reduce_mean(cross_entropy * self.rewards)

        self.gradients = tf.gradients(self.loss, tf.trainable_variables())

        # Create a placeholder list for gradients
        self.gradients_to_apply = []
        for index, variable in enumerate(tf.trainable_variables()):
            gradient_placeholder = tf.placeholder(tf.float32)
            self.gradients_to_apply.append(gradient_placeholder)

        # Create the operation to update gradients with the gradients placeholder.
        optimizer = tf.train.AdamOptimizer(learning_rate=1e-2)
        self.update_gradients = optimizer.apply_gradients(zip(self.gradients_to_apply, tf.trainable_variables()))

    discount_rate = 0.95

    def discount_normalize_rewards(self, rewards):
        discounted_rewards = np.zeros_like(rewards)
        total_rewards = 0

        for i in reversed(range(len(rewards))):
            total_rewards = total_rewards * self.discount_rate + rewards[i]
            discounted_rewards[i] = total_rewards

        discounted_rewards -= np.mean(discounted_rewards)
        discounted_rewards /= np.std(discounted_rewards)

        return discounted_rewards
