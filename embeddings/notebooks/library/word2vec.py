import torch
import torch.nn as nn
import torch.nn.functional as F

class SkipGram(nn.Module):

    def __init__(self, input_size, hidden_size):
        """Initializes the word2vec model

        Args:
            input_size (int): The size of the vocabulary.
            hidden_size (int): The word embedding size.

        """
        super(SkipGram, self).__init__()
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.h2o = nn.Linear(hidden_size, input_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input):
        """"Feed forward the model (used for training)

        Args:
            input (int): The index of the word in the embedding space.

        Returns:
            Tensor The distribution tensor.

        """
        output = self.embedding(input).view(1, -1)
        output = self.h2o(output)
        output = self.softmax(output)
        return output

    def get_vector(self, index):
        """Gets the corresponding embedding

        Args:
            index (int): The index of the word.

        Returns:
            Tensor: The word embedding.
        """
        return self.embedding(index)