import sys, pygame
import tensorflow as tf
import numpy as np
import string

pygame.init()

size = width, height = 600, 600
WHITE = (255,255,255)
BLACK = (0,0,0)
mouseX, mouseY = 0,0;

screen = pygame.display.set_mode(size)
pygame.display.set_caption("Character Recognition")
clock = pygame.time.Clock()
font = pygame.font.SysFont("Ubuntu", 20);

board = [[0 for i in range(28)] for j in range(28)];

# load json and weight files to import trained model
modelName = "model_characters"
json_file = open("models/"+modelName+'.json', 'r')
model = tf.keras.models.model_from_json(json_file.read())
json_file.close()
model.load_weights("models/"+modelName+".h5")
print("Loaded model from disk")

netGuess = np.zeros((2,62), dtype=int);

def shift(a, dx, dy):
    r = np.roll(a, int(dy), axis=0)
    r = np.roll(r, int(dx), axis=1)
    return r;


def center(a):
    ymax, xmax = np.max(np.where(a != 0), 1)
    ymin, xmin = np.min(np.where(a != 0), 1)
    mx = (xmin + xmax) / 2
    my = (ymin + ymax) / 2
    sx = int((len(a) / 2) - mx)
    sy = int((len(a) / 2) - my)
    return shift(a, sx, sy);

def printArray(a):
    print('\n'.join([''.join(['{:4}'.format(item) for item in row])
                                 for row in a]))

def dispText(text, font, x, y, color):
    TextSurf = font.render(text, True, color)
    TextRect = TextSurf.get_rect()
    TextRect.x = x; TextRect.y = y;
    screen.blit(TextSurf, TextRect)

def getNetGuess(arr):
    centered = center(np.array(arr))
    x = centered[np.newaxis, ...]  # add 3rd dim (needed for predict func)
    print(x.shape)
    #x = np.expand_dims(x, axis=4)
    print(x.shape)
    y = model.predict(x)
    y=y[0]
    print(y.shape)
    print(y)
    results = np.flip(np.argsort(y), axis=0)
    netGuess[0] = results
    netGuess[1] = np.flip(np.sort(np.round(y * 100)), axis=0)


isRunning = True
while isRunning:
    clock.tick(60) #Make sure pygame doesn't use all the resources
    #Register close window event
    for event in pygame.event.get():
        if event.type == pygame.QUIT: isRunning = False;
        elif event.type == pygame.MOUSEBUTTONUP:
            if event.button == pygame.BUTTON_LEFT:
                getNetGuess(board)
            if event.button == pygame.BUTTON_RIGHT:
                getNetGuess(board)
        elif event.type == pygame.KEYUP:
            if event.key == pygame.K_c:
                board = center(np.array(board)).tolist();



    pygame.event.get()
    mouseX, mouseY = pygame.mouse.get_pos()

    screen.fill(BLACK)

    # Register mouse drawing
    if 0 <= mouseX <= width and 0 <= mouseY <= width and pygame.mouse.get_focused():
        num = -1;
        if pygame.mouse.get_pressed()[0]: num = 1;
        elif pygame.mouse.get_pressed()[2]: num = 0;
        if num != -1:
            blockSize = width / len(board)
            board[round((mouseY - blockSize / 2) / blockSize)][round((mouseX - blockSize / 2) / blockSize)] = num;

    if pygame.key.get_pressed()[pygame.K_r]:
        board = [[0 for i in range(28)] for j in range(28)];

    # Draw board
    for y in range(len(board)):
        for x in range(len(board)):
            cColor = BLACK if board[y][x] == 0 else WHITE
            rSize = width / len(board)
            pygame.draw.rect(screen, cColor, pygame.Rect(rSize * x, y * rSize, rSize, rSize))

    textObj = pygame.font.SysFont('Loma',20)
    for i in range(3):
        #convert network output number to letter or number
        character = str(netGuess[0][i])
        if 10 <= netGuess[0][i] <= 35: #between A and Z
            character = chr(netGuess[0][i] + 55);
        elif 36 <= netGuess[0][i] <= 61: #between a and z
            character = chr(netGuess[0][i] + 61);
        text = character + " - " + str(netGuess[1][i])+"%"
        dispText(text, textObj, 10, i*textObj.size(text)[1], WHITE)

    pygame.display.flip()
