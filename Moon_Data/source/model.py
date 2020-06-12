import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleNet(nn.Module):
    
    def __init__(self, input_dim, hidden_dim, output_dim):
        '''Defines layers of a neural network.
           :param input_dim: Number of input features
           :param hidden_dim: Size of hidden layer(s)
           :param output_dim: Number of outputs
         '''
        super(SimpleNet, self).__init__()
        self.input_dim = input_dim
        # linear layer (input_dim -> hidden_dim)
        self.fc1 = nn.Linear(self.input_dim, hidden_dim)
        # linear layer (hidden_dim -> hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        # linear layer (hidden_dim -> output_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        # let's be a tryhard and add 33% dropout
        self.dropout = nn.Dropout(0.33)
        # sigmoid layer (output_dim -> class score)
        self.sig = nn.Sigmoid()
        

    def forward(self, x):
        '''Feedforward behavior of the net.
           :param x: A batch of input features
           :return: A single, sigmoid activated value
         '''
        # ensure input shape
        x = x.view(-1, self.input_dim)
        # sigmoid activation on first fully connected layer
        x = F.relu(self.fc1(x))
        # dropout layer
        x = self.dropout(x)
        # sigmoid activation on second fully connected layer
        x = F.relu(self.fc2(x))
        # dropout layer
        x = self.dropout(x)
        # output layer
        x = self.fc3(x)
        # convert output to class score
        return self.sig(x)