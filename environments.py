import copy
import sys
import functions

from matplotlib import pyplot as plt
import torch

import numpy as np

'''
0 = Red:Front, 1 = Blue:Right, 2 = Orange:Back, 3 = Green:Left, 4 = Yellow:Bottom, 5 = White:Top
Move List:
0:
↑--
↑--
↑--
1:
-↑-
-↑-
-↑-
2:
--↑
--↑
--↑
3:
→→→
---
---
4:
---
→→→
---
5:
---
---
→→→
6:
---
-↻-
---
7 (Middle):
---
-↻-
---
8 (Back):
---
-↻-
---
'''


class Rubiks:  # Rubik's cube environment.
    def __init__(self):
        self.state = np.array([[[f] * 3 for i in range(3)] for f in range(6)])  # Creates a 3D array representing cube.

    def rotate(self, move: int):  # Rotations verified by comparing to an actual cube's rotations.
        # Above comment shows all rotations and corresponding numbers. Numbers past 8 are just inverses.
        if move <= 2:
            faces = [0, 5, 2, 4, 0]  # Faces of the cube to be modified.
            self.transpose_cube()
            old_cube = copy.deepcopy(self.state)
            self.state[faces[1]][move] = old_cube[faces[0]][move]
            self.state[faces[2]][2 - move] = np.flip(old_cube[faces[1]][move])
            self.state[faces[3]][move] = np.flip(old_cube[faces[2]][2 - move])
            self.state[faces[4]][move] = old_cube[faces[3]][move]
            self.transpose_cube()
            if move == 0:
                self.state[3] = np.flip(self.state[3], axis=1).T
            elif move == 2:
                self.state[1] = np.flip(self.state[1].T, axis=1)
        elif move <= 5:
            faces = [0, 1, 2, 3, 0]
            old_cube = copy.deepcopy(self.state)
            for f in range(1, 5):
                self.state[faces[f]][move % 3] = old_cube[faces[f - 1]][move % 3]
            if move == 3:
                self.state[5] = np.flip(self.state[5], axis=1).T
            elif move == 5:
                self.state[4] = np.flip(self.state[4].T, axis=1)
        elif move <= 8:
            faces = [3, 5, 1, 4, 3]
            old_cube = copy.deepcopy(self.state)
            self.state[5][2 - move % 3] = np.flip(old_cube[3].T[2 - move % 3])
            self.state[4][move % 3] = np.flip(old_cube[1].T[move % 3])
            self.transpose_cube()
            self.state[1][move % 3] = old_cube[5][2 - move % 3]
            self.state[3][2 - move % 3] = old_cube[4][move % 3]
            self.transpose_cube()
            if move == 6:
                self.state[0] = np.flip(self.state[0].T, axis=1)
            elif move == 8:
                self.state[2] = np.flip(self.state[2], axis=1).T
        elif move <= 17:  # If move number is greater than 8, it becomes an inverse of that move number - 9
            self.rotate(move - 9)
            self.rotate(move - 9)
            self.rotate(move - 9)
        else:
            print("ERROR in rotate function.")
            sys.exit()

    def transpose_cube(self):  # Sorts the cube by columns rather than rows.
        for f in range(6):
            self.state[f] = self.state[f].T

    def scramble(self, rotations: int, return_hist=False):
        rot_hist = []
        for r in range(rotations):
            rot = np.random.randint(0, 18)
            rot_hist.append(rot)
            self.rotate(rot)
        if return_hist:
            return rot_hist

    def bot_cube(self):  # Flattens cube, one-hot encodes it, and converts it to a tensor for processing.
        return torch.flatten(torch.tensor([functions.one_hot(s, 6) for s in torch.flatten(torch.tensor(self.state))]))

    def reset(self):
        self.state = np.array([[[f] * 3 for i in range(3)] for f in range(6)])

    def pretty_print(self):
        print("Front:\n", self.state[0])
        print("Right:\n", self.state[1])
        print("Back:\n", self.state[2])
        print("Left:\n", self.state[3])
        print("Bottom:\n", self.state[4])
        print("Top:\n", self.state[5])

    def plot(self):  # Uses matplotlib to visualize a flattened cube. Some of these images are in the PDF of the project.
        plt.figure(facecolor='Black', figsize=(6, 6))
        ax = plt.axes()
        ax.set_facecolor("Black")
        plt.xlim(0, 12)
        self.state[2] = np.flip(np.flip(self.state[2], axis=1), axis=0)  # Bottom face is upsidedown because cube is unwrapped
        for f in range(len(self.state)):
            start = []
            if f == 0:  # Set corresponding starting positions for plotting based on face
                start = [4, 9]
            elif f == 1:
                start = [7, 9]
            elif f == 2:
                start = [4, 3]
            elif f == 3:
                start = [1, 9]
            elif f == 4:
                start = [4, 6]
            elif f == 5:
                start = [4, 12]
            else:
                print("Error in plot function.")
                sys.exit()
            pos = copy.deepcopy(start)
            for r in range(len(self.state[f])):
                for c in range(len(self.state[f][r])):
                    color = ""
                    square = self.state[f][r][c]
                    if square == 0:  # Set square color depending on what number it is
                        color = "red"
                    elif square == 1:
                        color = "blue"
                    elif square == 2:
                        color = "orange"
                    elif square == 3:
                        color = "green"
                    elif square == 4:
                        color = "yellow"
                    elif square == 5:
                        color = "white"
                    plt.scatter([pos[0]], [pos[1]], marker="s", color=color, s=[600])
                    pos[0] += 1
                pos[1] -= 1
                pos[0] = start[0]
        self.state[2] = np.flip(np.flip(self.state[2], axis=1), axis=0)  # Undo making bottom face upside down
        plt.show()

    def is_solved(self):
        for f in range(len(self.state)):
            for r in range(len(self.state[f])):
                for c in range(len(self.state[f])):
                    if not self.state[f][r][c] == self.state[f][0][0]:
                        return False
        return True
