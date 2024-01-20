import numpy as np
from matplotlib import pyplot as plt
import torch
import environments


def one_hot(out: int, dim: int):  # Makes data easier to interpret for the network.
    ret = [0.0] * dim
    ret[out] = 1.0
    return ret


def play(model, scrambles):
    cube = environments.Rubiks()

    cube.scramble(scrambles)
    with torch.no_grad():
        for m in range(2*scrambles):
            cube.plot()
            move = model.forward(cube.bot_cube())
            cube.rotate(np.argmax(move))
            if cube.is_solved():
                cube.plot()
                print("Cube Solved!")
                return
        print("Cube not solved in time.")


def test_model(model):  # Will test model against 1000 games for each number of random rotations
    # This function generated the numbers used in the graphs and the functions:
    # functions.graph_proportion_of_cubes_solved()
    # functions.graph_average_moves()
    cube = environments.Rubiks()

    games = 1000

    p_won = []
    avg_moves = []

    for scr in range(1, 16):
        won = 0
        tot_moves = 0
        for g in range(games):
            cube.reset()
            cube.scramble(scr)

            moves = 0
            max_moves = scr * 2
            vic = False

            for m in range(max_moves):
                moves += 1
                with torch.no_grad():
                    cube.rotate(torch.argmax(model(cube.bot_cube())))
                if cube.is_solved():
                    won += 1
                    vic = True
                    break

            if vic:
                tot_moves += moves

        p_won.append(won / games)
        if won != 0:
            avg_moves.append(tot_moves / won)

    print("Portion Won:", p_won)
    print("Average # of Moves (Victorious Games)", avg_moves)


# Values below represent cube solving rate and average moves acquired from prior processing.

CM_E1_S10_P = [1.0, 1.0, 1.0, 0.999, 0.988, 0.941, 0.852, 0.708, 0.572, 0.439, 0.326, 0.242, 0.179, 0.132, 0.075]
CM_E1_S10_M = [1.0, 2.0, 2.758, 3.5950975487743873, 4.432068791097623, 5.2695615245779805, 6.08267855048669, 6.793366266760763, 7.442405173920643, 7.942649066909422, 8.604294478527608, 8.78099173553719, 9.324022346368714, 9.348484848484848, 8.946666666666667]

CM_E2_S10_P = [1.0, 1.0, 1.0, 1.0, 0.991, 0.966, 0.88, 0.752, 0.647, 0.461, 0.354, 0.247, 0.183, 0.133, 0.099]
CM_E2_S10_M = [1.0, 2.0, 2.728, 3.586, 4.434914228052472, 5.171842650103519, 6.0852272727272725, 6.6861702127659575, 7.457496136012365, 7.947939262472885, 8.6045197740113, 9.279352226720647, 9.371584699453551, 9.390977443609023, 10.181818181818182]

CM_E1_S10_B32_P = [1.0, 1.0, 1.0, 1.0, 0.982, 0.947, 0.86, 0.715, 0.572, 0.455, 0.328, 0.233, 0.157, 0.107, 0.079]
CM_E1_S10_B32_M = [1.0, 2.0, 2.726, 3.576, 4.413441955193482, 5.23336853220697, 6.1069767441860465, 6.8657342657342655, 7.346153846153846, 8.184615384615384, 8.5, 8.948497854077253, 9.45859872611465, 9.495327102803738, 10.531645569620252]

CM_E1_S10_B64_P = [1.0, 1.0, 1.0, 0.999, 0.987, 0.937, 0.827, 0.68, 0.585, 0.438, 0.324, 0.229, 0.167, 0.123, 0.074]
CM_E1_S10_B64_M = [1.0, 2.0, 2.764, 3.6096096096096097, 4.439716312056738, 5.2764140875133405, 6.029020556227327, 6.826470588235294, 7.177777777777778, 7.892694063926941, 8.570987654320987, 8.554585152838428, 9.449101796407186, 9.40650406504065, 9.986486486486486]

CM_E1_S10_B128_P = [0.049, 0.092, 0.025, 0.01, 0.008, 0.002, 0.0, 0.003, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
CM_E1_S10_B128_M = [1.0, 3.5543478260869565, 2.56, 3.2, 2.5, 3.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

CM_E1_S20_P = [1.0, 1.0, 1.0, 1.0, 0.99, 0.94, 0.851, 0.714, 0.592, 0.484, 0.369, 0.255, 0.184, 0.11, 0.087]
CM_E1_S20_M = [1.0, 2.0, 2.716, 3.623, 4.442424242424242, 5.327659574468085, 6.029377203290247, 6.8277310924369745, 7.295608108108108, 8.111570247933884, 8.639566395663957, 9.196078431372548, 9.353260869565217, 10.418181818181818, 9.89655172413793]

CM_E2_S20_P = [1.0, 1.0, 1.0, 1.0, 0.997, 0.968, 0.89, 0.793, 0.657, 0.544, 0.391, 0.296, 0.224, 0.171, 0.12]
CM_E2_S20_M = [1.0, 2.0, 2.73, 3.6, 4.422266800401204, 5.285123966942149, 6.0662921348314605, 6.802017654476671, 7.491628614916286, 8.121323529411764, 8.654731457800512, 9.304054054054054, 9.3125, 9.935672514619883, 10.75]

CM_E1_S30_P = [1.0, 1.0, 1.0, 0.999, 0.982, 0.951, 0.858, 0.723, 0.62, 0.464, 0.358, 0.273, 0.195, 0.142, 0.092]
CM_E1_S30_M = [1.0, 2.0, 2.756, 3.6166166166166165, 4.462321792260693, 5.299684542586751, 5.945221445221446, 6.8395573997233745, 7.52741935483871, 8.245689655172415, 8.723463687150838, 9.131868131868131, 9.430769230769231, 10.309859154929578, 11.173913043478262]

CM_E2_S30_P = [1.0, 1.0, 1.0, 1.0, 0.997, 0.972, 0.898, 0.79, 0.683, 0.527, 0.427, 0.322, 0.213, 0.188, 0.125]
CM_E2_S30_M = [1.0, 2.0, 2.766, 3.575, 4.433299899699097, 5.302469135802469, 6.001113585746102, 6.968354430379747, 7.486090775988287, 8.094876660341557, 8.78688524590164, 8.850931677018634, 9.845070422535212, 9.824468085106384, 10.344]

scr = [x for x in range(1, 16)]


def graph_proportion_of_cubes_solved():
    plt.figure(figsize=(11, 8), facecolor='#e5e5e5')
    plt.gca().set_facecolor('#f0f0f0')
    plt.grid(True, linestyle='--', linewidth=0.2)
    plt.tick_params(axis='both', which='major', labelsize=10)
    plt.xticks(scr)
    plt.yticks([.0, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1.0])
    plt.plot(scr, CM_E1_S10_P, label="1M Episodes, 10 Scrambles, Batch Size 1")
    plt.plot(scr, CM_E2_S10_P, label="2M Episodes, 10 Scrambles, Batch Size 1")
    plt.plot(scr, CM_E1_S10_B32_P, label="1M Episodes, 10 Scrambles, Batch Size 32")
    plt.plot(scr, CM_E1_S10_B64_P, label="1M Episodes, 10 Scrambles, Batch Size 64")
    plt.plot(scr, CM_E1_S10_B128_P, label="1M Episodes, 10 Scrambles, Batch Size 128")
    plt.plot(scr, CM_E1_S20_P, label="1M Episodes, 20 Scrambles, Batch Size 1")
    plt.plot(scr, CM_E2_S20_P, label="2M Episodes, 20 Scrambles, Batch Size 1")
    plt.plot(scr, CM_E1_S30_P, label="1M Episodes, 30 Scrambles, Batch Size 1")
    plt.plot(scr, CM_E2_S30_P, label="2M Episodes, 30 Scrambles, Batch Size 1")
    plt.legend(loc="upper right", title="Neural Networks", fontsize=7)
    plt.xlabel("Random Rotations When Scrambled", fontsize=13, )
    plt.ylabel("Portion out of 1000", fontsize=13)
    plt.title("Portion of Cubes Solved", fontsize=20, fontweight='bold')
    plt.show()


def graph_average_moves():
    plt.figure(figsize=(11, 8), facecolor='#e5e5e5')
    plt.gca().set_facecolor('#f0f0f0')
    plt.grid(True, linestyle='--', linewidth=0.2)  # Dashed grid lines
    plt.tick_params(axis='both', which='major', labelsize=10)  # Adjust tick label size
    plt.xticks(scr)
    plt.yticks([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13])
    plt.plot(scr, CM_E1_S10_M, label="1M Episodes, 10 Scrambles, Batch Size 1")
    plt.plot(scr, CM_E2_S10_M, label="2M Episodes, 10 Scrambles, Batch Size 1")
    plt.plot(scr, CM_E1_S10_B32_M, label="1M Episodes, 10 Scrambles, Batch Size 32")
    plt.plot(scr, CM_E1_S10_B64_M, label="1M Episodes, 10 Scrambles, Batch Size 64")
    plt.plot(scr, CM_E1_S10_B128_M, label="1M Episodes, 10 Scrambles, Batch Size 128")
    plt.plot(scr, CM_E1_S20_M, label="1M Episodes, 20 Scrambles, Batch Size 1")
    plt.plot(scr, CM_E2_S20_M, label="2M Episodes, 20 Scrambles, Batch Size 1")
    plt.plot(scr, CM_E1_S30_M, label="1M Episodes, 30 Scrambles, Batch Size 1")
    plt.plot(scr, CM_E2_S30_M, label="2M Episodes, 30 Scrambles, Batch Size 1")
    plt.legend(loc="upper left", title="Neural Networks", fontsize=7)
    plt.xlabel("Random Rotations When Scrambled", fontsize=13)
    plt.ylabel("Moves To Solve Cube", fontsize=13)
    plt.title("Average Number of Moves to Solve a Cube", fontsize=20, fontweight='bold')
    plt.show()