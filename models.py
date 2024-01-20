import torch


class MLP(torch.nn.Module):  # Multi Layer Perceptron using PyTorch
    def __init__(self, shape: list[int], hidden_activation, output_activation):
        super().__init__()
        self.shape = shape
        self.network = torch.nn.Sequential()
        for i in range(len(shape)-2):
            self.network.append(torch.nn.Linear(shape[i], shape[i+1]))
            self.network.append(hidden_activation())
        self.network.append(torch.nn.Linear(shape[-2], shape[-1]))
        if output_activation is not None:
            self.network.append(output_activation())

    def forward(self, x):
        return self.network(x)


