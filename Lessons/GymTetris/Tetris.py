import numpy as np
import tensorflow as tf
import time

from nes_py.wrappers import JoypadSpace
import gym_tetris
import gym
from gym_tetris.actions import SIMPLE_MOVEMENT
from Agent import Agent

env = gym_tetris.make("TetrisA-v0")
env = JoypadSpace(env, SIMPLE_MOVEMENT)

#action size is NES simple input for tetris
#state size is (240, 256, 3)
num_actions = len(SIMPLE_MOVEMENT)
state_size = (240, 256, 3)

agent = Agent(state_size, num_actions)

episode = 0
running = True
isTrained = False
while running:
    episode += 1
    state = env.reset()
    #do training
    done = False
    frames = 0
    while not done:
        env.render()

        action = agent.act(state)
        next_state, reward, done, _ = env.step(action)
        agent.remember(state, action, reward, next_state, done)
        state = next_state #update state

    agent.replay(32)
    print("Episode: " + str(episode) + ", Average reward / 100 eps: " + str(np.mean(total_episode_rewards[-100:])))
