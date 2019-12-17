import numpy as np
import tensorflow as tf
import time
import matplotlib.pyplot as plt

from nes_py.wrappers import JoypadSpace
import gym_tetris
import gym
from gym_tetris.actions import SIMPLE_MOVEMENT

env = gym_tetris.make("TetrisA-v2")
env = JoypadSpace(env, SIMPLE_MOVEMENT)

#action size is NES simple input for tetris
#state size is (240, 256, 3)
num_actions = len(SIMPLE_MOVEMENT)
state_size = 214

episode = 0
running = True
isTrained = False
state = env.reset()

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
    
    board = (board != 0.0)
    return board.astype(int)

#Receives piece string
pieceDict = {'L': 0, 'T': 0, 'Z': 0, 'O': 0, 'S': 0, 'J': 0, 'I': 0}
def getPiece(piece):
    piece = piece[0] #first character is piece signifier
    dict = pieceDict.copy()
    dict[piece] = 1;
    return list(dict.values())

def getState(board, curPiece, nextPiece):
    state = getPiece(curPiece)
    state.extend(getPiece(nextPiece))
    state.extend(board.flatten())
    return state

def printBoard(board):
    print('\n'.join([''.join(['{:4}'.format(item) for item in row]) 
      for row in board]))

skip = 1

while running:
    episode += 1
    #do training
    done = False
    skip = 0
    frames = 0
    env.reset();
    
    while not done:
        
        rawstate, reward, done, info = env.step(5)
        board = getBoard(rawstate)
        state = getState(board, info["current_piece"], info["next_piece"])
        
        print(info)
        print("State Len:",len(state))
        
        #network training here
        
        prev_state = state #update state
        
        '''plt.imshow(state)
        plt.show()'''
        
        env.render()
        
        if skip < 0:
            userInput = input("Command>");
            inputArr = userInput.split(' ')
            if userInput == "stop": running = False
            if userInput == "reset": state = env.reset()
            if inputArr[0] == "skip": skip += int(inputArr[1])
            if inputArr[0] == "printboard": printBoard(board)
            if inputArr[0] == "help": print("stop, reset, skip, printboard, help")
        else: skip -= 1;
    
    print("Episode: " + str(episode))
