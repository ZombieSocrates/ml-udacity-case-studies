# torch imports
import torch.nn.functional as F
import torch.nn as nn


## TODO: Complete this classifier
class BinaryClassifier(nn.Module):
    """
    Define a neural network that performs binary classification.
    The network should accept your number of features as input, and produce 
    a single sigmoid value, that can be rounded to a label: 0 or 1, as output.
    
    Notes on training:
    To train a binary classifier in PyTorch, use BCELoss.
    BCELoss is binary cross entropy loss, documentation: https://pytorch.org/docs/stable/nn.html#torch.nn.BCELoss
    """

    def __init__(self, input_features, hidden_dim, output_dim):
        """
        Initialize the model by setting up linear layers.
        Use the input parameters to help define the layers of your model.
        :param input_features: the number of input features in your training/test data
        :param hidden_dim: helps define the number of nodes in the hidden layer(s)
        :param output_dim: the number of outputs you want to produce
        """
        super(BinaryClassifier, self).__init__()
        self.input_dim = input_features
        # linear layer (input dim -> hidden dim)
        self.fc1 = nn.Linear(self.input_dim, hidden_dim)
        # linear layer (hidden dim -> hidden dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        # linear layer (hidden dim -> output dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        # dropout of 10%, later bumped to 25%. Could have made this a kwarg...
        self.dropout = nn.Dropout(0.25)
        # sigmoid on the output to get a class score
        self.sig = nn.Sigmoid()
        

    
    ## TODO: Define the feedforward behavior of the network
    def forward(self, x):
        """
        Perform a forward pass of our model on input features, x.
        :param x: A batch of input features of size (batch_size, input_features)
        :return: A single, sigmoid-activated value as output
        """
        # ensure input shape
        x = x.view(-1, self.input_dim)
        # ReLU activations on first layer, with dropout
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        # Same on second layer
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        # Output layer
        x = self.fc3(x)
        # return class score
        return self.sig(x)
    