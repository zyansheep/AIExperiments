import gym
import time
import tensorflow as tf
from Agent import Agent
import numpy as np

training_cutoff = 5000
#Make Cartpole Game
gym.envs.register(
    id='CartPoleLong-v1',
    entry_point='gym.envs.classic_control:CartPoleEnv',
    max_episode_steps=training_cutoff+1,
    reward_threshold=-110.0,
)
env = gym.make("CartPoleLong-v1")
num_actions = 2 #2 actions for cartpole game
state_size = 4 #4 states for cartpole game

tf.reset_default_graph()

path = "GymCartPole/cartpole-pg/" #checkpoint dir

training_episodes_per_update = 10
episode_batch_size = 1

agent = Agent(num_actions, state_size) #neural network

init = tf.global_variables_initializer()

saver = tf.train.Saver(max_to_keep=2)

sess = tf.Session()
sess.run(init)

total_episode_rewards = []

# Create a buffer of 0'd gradients
gradient_buffer = sess.run(tf.trainable_variables())
for index, gradient in enumerate(gradient_buffer):
    gradient_buffer[index] = gradient * 0

episode = 0

running = True
isTrained = False
while running:
    episode += 1
    state = env.reset()

    episode_history = []
    episode_rewards = 0

    #do training
    done = False
    frames = 0
    while not done:
        frames += 1
        if frames > training_cutoff:
            isTrained = True
            env._max_episode_steps = 1000000000
            break

        if isTrained:
            time.sleep(0.01)

        if episode % training_episodes_per_update == 0 or isTrained:
            env.render()


        # Get weights for each action
        action_probabilities = sess.run(agent.outputs, feed_dict={agent.input_layer: [state]})
        action_choice = np.random.choice(range(num_actions), p=action_probabilities[0])

        #get new state, reward on how good ai did, and if ai failed or not
        state_next, reward, done, _ = env.step(action_choice)
        #reward = frame

        #log history of what happened during state
        episode_history.append([state, action_choice, reward, state_next])
        state = state_next

        #rewards should also include amount of time survived
        episode_rewards += reward


    #When look quits (ai failed) calculate gradient for episode
    total_episode_rewards.append(episode_rewards)
    episode_history = np.array(episode_history)
    episode_history[:, 2] = agent.discount_normalize_rewards(episode_history[:, 2])

    # calculate gradient descent based on reward
    ep_gradients = sess.run(agent.gradients, feed_dict={agent.input_layer: np.vstack(episode_history[:, 0]),
                                                        agent.actions: episode_history[:, 1],
                                                        agent.rewards: episode_history[:, 2]})
    # add the gradients to the grad buffer:
    for index, gradient in enumerate(ep_gradients):
        gradient_buffer[index] += gradient

    if episode % episode_batch_size == 0:

        feed_dict_gradients = dict(zip(agent.gradients_to_apply, gradient_buffer))

        sess.run(agent.update_gradients, feed_dict=feed_dict_gradients)

        for index, gradient in enumerate(gradient_buffer):
            gradient_buffer[index] = gradient * 0

        if episode % training_episodes_per_update == 0:
            saver.save(sess, path + "pg-checkpoint", episode)

    print("Episode: " + str(episode) + ", Frames: " + str(frames) + "/" + str(training_cutoff) + " Average reward / 100 eps: " + str(np.mean(total_episode_rewards[-100:])))
