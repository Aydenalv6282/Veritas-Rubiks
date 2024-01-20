import functions
import models
import environments
import sys
import copy

import torch


def trainer(model: models.MLP, optimizer_fn, learning_rate: float, loss_fn, x, y):
    optimizer = optimizer_fn(model.parameters(), lr=learning_rate)
    optimizer.zero_grad()
    model.train()
    for i in range(len(x)):
        # Make a prediction and find error
        y_hat = model(x[i])
        # print(y_hat.dtype)
        # print(y[i].dtype)
        # print(x[i])
        # print(y_hat, y[i])
        loss = loss_fn(y_hat, y[i])
        # print(loss)
        # Training
        loss.backward()
        # print("===")
    optimizer.step()
    optimizer.zero_grad()


'''
Trains network by scrambling the rubix cube and showing how to undo it, rather than reinforcement learning.
'''


def RubiksUnscramble(model: models.MLP, lr: float, episodes: int, scrambles: int, batch_size: int):
    rubiks = environments.Rubiks()
    unscramble_train_batch = []
    cube_hist_batch = []
    for e in range(episodes+1):
        rot_hist = rubiks.scramble(scrambles, return_hist=True)  # Scrambles cube and stores rotation history
        unscramble = [i+9 if i < 9 else i-9 for i in reversed(rot_hist)]  # Converts rotation history into a solution
        unscramble_train = [torch.tensor(functions.one_hot(u, 18)) for u in unscramble]  # Uses one-hot encoding on solution
        cube_hist = []

        for rot in unscramble:  # Look at how the cube was scrambled and reverse each move.
            cube_hist.append(rubiks.bot_cube())
            rubiks.rotate(rot)
        if not rubiks.is_solved():
            print("Error in RubiksUnscramble function.")
            sys.exit()
        # print(rot_hist)
        unscramble_train_batch += unscramble_train
        cube_hist_batch += cube_hist
        if (e / episodes) * 100 % 1 == 0:
            print("% Complete:", e / episodes)
            with torch.no_grad():
                print(torch.nn.MSELoss()(model(cube_hist[0]), unscramble_train[0]))
        if e % batch_size == 0:
            trainer(model=model, optimizer_fn=torch.optim.SGD, loss_fn=torch.nn.MSELoss(), learning_rate=lr,
                    x=cube_hist_batch, y=unscramble_train_batch)
            unscramble_train_batch = []
            cube_hist_batch = []