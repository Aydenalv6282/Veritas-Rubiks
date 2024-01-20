import models
import torch
import environments
import numpy as np
import optimizers
import functions

# Requires python 3.10
# Commands to run before running code:
'''
pip install matplotlib
pip install numpy

pip install torch 
OR, if you have CUDA, visit and run the output command:
https://pytorch.org/get-started/locally/ 
'''

'''
CubeMaster naming scheme:
Number before the dash: Episodes in millions
Number after the dash: Cube random rotations trained on
Number after B: Batch Size (if no B, then batch size = 1)
'''

# If your GPU is CUDA compatible
# torch.set_default_tensor_type(torch.cuda.FloatTensor)

# Create or load a pytorch multi layer perceptron
unscrambler = torch.load("RubiksNets/CubeMaster2-30", map_location=torch.device('cpu'))  # Remove map_location if you have CUDA
# unscrambler = models.MLP(shape=[324, 1024, 1024, 18], hidden_activation=torch.nn.ReLU, output_activation=torch.nn.Softmax)

# Optimize the model using the RubiksUnscramble algorithm
# Best results only really come in after 1 million episodes and under 64 batches, this may take a day, but 10000 should be 1 min
# optimizers.RubiksUnscramble(model=unscrambler, lr=.1, episodes=10000, scrambles=1, batch_size=1)

# Save the model for later use
# torch.save(unscrambler, "RubiksNets/Unscrambler")

# Visualize flattened cube and have the model play.
# Close the window that pops up every time you want the model to move.
functions.play(model=unscrambler, scrambles=7)

# Perform a more comprehensive test on the model, this may take a few minutes
# When the lists are printed, 0th index is equivalent to 1 scramble, 1 to 2, etc.
# functions.test_model("RubiksNets/Unscrambler")

# Graph the same graphs shown in the paper (it has its data loaded already)
# functions.graph_proportion_of_cubes_solved()
# functions.graph_average_moves()

