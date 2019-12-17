import numpy as np

from nes_py.wrappers import JoypadSpace
import gym_tetris
from gym_tetris.actions import SIMPLE_MOVEMENT
from Agent import Agent

env = gym_tetris.make("TetrisA-v0")
env = JoypadSpace(env, SIMPLE_MOVEMENT)

#action size is NES simple input for tetris
#state size is (240, 256, 3)
num_actions = 14
state_size = (180,)

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

defaultRotations = [2,1,2,1,2,1,0]
def getActions(netOut, curPiece):
    col = netOut[:9] #Collumn to move to
    col = np.amax(col) #collumn 0-9
    rot = netOut[10:] #Rotation orientation
    rot = np.amax(rot) #0-up/vert, 1-right/horz, 2-left, 4-down
    
    actions = []
    defRot = defaultRotations[curPiece]
    #rotation actions
    while True:
        if rot > defRot:
            defRot += 1
            actions.append(1) #A button - Rotate clockwise
        elif defRot < rot:
            defRot -= 1
            actions.append(2) #B button - Rotate counterclockwise
        else: break
    
    #movement actions
    while True:
        if col > 4:
            col += 1
            actions.append(3) #Right button
        elif col < 4:
            col -= 1
            actions.append(4) #Left button
        else: break
        
    

episode = 0
running = True
isTrained = False
while running:
    episode += 1
    state = env.reset() #Reset enviroment
    state, _, done, info = env.step(0) #get game details
    while not done:
        beginState = state
        beginState = getState(getBoard(beginState), info["current_piece"], info["next_piece"])
        
        #output [column, rotation] eg. [0,0,0,1,0,0,0,0,0,0], [0,0,1,0]
        netOut = agent.act(beginState) #Act on last state
        print(netOut)
        curPiece = list(info["statistics"].keys()).index(info["current_piece"][0])
        actionArr = getActions(netOut, curPiece)
        #do actions based on what rotation and location network wants piece in
        for action in actionArr:
            _, _, done, info = env.step(action)
        
        #until next piece
        nextPiece = info["next_piece"]
        while info["current_piece"] != nextPiece:
            rawstate, reward, done, info = env.step(5) #Move down
        
        board = getBoard(rawstate)
        state = getState(board, info["current_piece"], info["next_piece"])
        
        agent.remember(beginState, action, reward, state, done)
        #update state
        env.render()

    agent.replay(32) #batch size
    print("Episode: " + str(episode))
