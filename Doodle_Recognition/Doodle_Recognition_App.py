
import numpy as np
import pygame
import NeuralNetwork as nn
import numpy as np
import joblib
import sys

IMAGE_DIM = 28
IMAGE_SHAPE = (IMAGE_DIM, IMAGE_DIM)
TOTAL_PIXELS = IMAGE_DIM ** 2
TOTAL_CLASSES = 10

class Colors:
    WHITE = (255, 255, 255)
    GREY = (153, 153, 53)
    BLACK = (0, 0, 0)


class Spot:
    def __init__(self, row, col, width):
        self.row = row
        self.col = col
        self.x = col * width
        self.y = row * width
        self.color = list(Colors.BLACK)
        self.width = width
    
    def add_color(self):
        for c in range(3):
            self.color[c] += 50. if self.color[c] < 250. else 0
    
    def get_pos(self):
        return self.row, self.col
    
    def reset(self):
        self.color = list(Colors.BLACK)
    
    def draw(self, win):
        pygame.draw.rect(win, self.color, (self.x, self.y, self.width, self.width))

# create a grid full of Spots
def make_grid(rows, window_width):
    grid = []
    gap = window_width / rows
    for i in range(rows):
        grid.append([])
        for j in range(rows):
            spot = Spot(i, j, gap)
            grid[i].append(spot)
    
    return grid

# draw the window
def draw(win, grid, rows, window_width):
    win.fill(Colors.BLACK)
    
    for row in grid:
        for spot in row:
            spot.draw(win)
    
    pygame.display.update()

# get the index of the box the user clicked on
def get_indices(pos, rows, window_width):
    gap = window_width // rows
    x, y = pos
    row = y // gap
    col = x // gap
    return row, col
    
def main():
    model = joblib.load('SSmodel.joblib')
    win = pygame.display.set_mode((TOTAL_PIXELS, TOTAL_PIXELS))
    pygame.display.set_caption("Doodle Classifier")
    ans = ["broccoli","cruise ship","angel","bicycle","umbrella","octopus","plant","windmill","airplane","popsicle","axe","rainbow","envelope","eye","donut","lightning","smiley_face","helicopter","sun","dolphin"]
    grid = make_grid(IMAGE_DIM, TOTAL_PIXELS)
    doodle = np.empty((1,784),dtype=np.float64)

    run = True
    while run:
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                run = False
                break
            
            if e.type == pygame.KEYDOWN:
                # clear
                if e.key == pygame.K_c:
                    doodle = np.empty((1,784),dtype=np.float64)
                    for row in grid:
                        for spot in row:
                            spot.reset()
        
        # draw
        if pygame.mouse.get_pressed()[0]:  # left mouse
            pos = pygame.mouse.get_pos()
            row, col = get_indices(pos, 28, 784)
            if row < 28 and col<28 and row >=0 and col >=0:
                grid[row][col].add_color()
                doodle[0,row*28+col] += 50. if doodle[0,row*28+col] < 250. else 0
                #image = ((doodle/255.)-.5)*2
                image = doodle/255.
                #res = model.network_predict(image)
                procents,idx_list = model.predict(image)
                procents = -np.sort(-procents)
                procents = procents*100
                #print(procents)
                sys.stderr.write('\r%s: %.2f , %s: %.2f , %s: %.2f , %s: %.2f , %s: %.2f                      '%(ans[int(idx_list[0])],procents[0,0],ans[int(idx_list[1])],procents[0,1],ans[int(idx_list[2])],procents[0,2],ans[int(idx_list[3])],procents[0,3],ans[int(idx_list[4])],procents[0,4]))
                sys.stderr.flush()
        
        # erase
        if pygame.mouse.get_pressed()[2]:  # right mouse
            pos = pygame.mouse.get_pos()
            row, col = get_indices(pos, 28, 784)
            grid[row][col].reset()
            doodle[0,row*28+col] = 0.
        
        draw(win, grid, IMAGE_DIM, TOTAL_PIXELS)


main()