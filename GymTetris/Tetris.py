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

episodeRenderInterval = 10;

def getBoard(state):
    state = state[48:208, 96:176] #Get tetris board only (not whole screen)
    state = np.mean(state, -1);
    
    #take middle of pixel
    board = np.zeros((20,10))
    for line in range(4,len(state), 8):
        for item in range(4,len(state[line]), 8):
            ind1 = (int)((line-4)/8);
            ind2 = (int)((item-4)/8);
            board[ind1, ind2] = state[line, item]
    
    return board

#Receives piece string
pieceDict = {'L': 0, 'T': 0, 'Z': 0, 'O': 0, 'S': 0, 'J': 0, 'I': 0}
def getPiece(piece):
    piece = piece[0] #first character is piece signifier
    dict = pieceDict
    dict[piece] = 1;
    return dict.values()

def getState(board, curPiece, nextPiece):
    state = getPiece(curPiece)
    state.extend(getPiece(nextPiece))
    state.extend(np.flatten(board))

episode = 0
running = True
isTrained = False
while running:
    episode += 1
    state = env.reset()
    #do training
    done = False
    frames = 0
    lastPiece = 0;
    while not done:
        lastState = state
        if(episode % episodeRenderInterval == 0):
            env.render()
        
        action = agent.act(lastState) #Act on last state
        
        rawstate, reward, done, info = env.step(action)
        nextPiece = info["next_piece"]
        while info["current_piece"] != nextPiece:
            env.step(5) #Move down
        
        board = getBoard(rawstate)
        state = getState(board, info["current_piece"], info["next_piece"])
        
        agent.remember(state, action, reward, state, done)
         #update state
        lastPiece = info.current_piece

    agent.replay(32)
    print("Episode: " + str(episode))
