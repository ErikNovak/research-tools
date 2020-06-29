import torch
import torch.nn as nn
import torch.nn.functional as F

class GloVe(nn.Module):

    def __init__(self, input_size, vector_size, alpha=.75, x_max=100):
        """Initializes the GloVe model
        
        Args:
            input_size (int): The size of the vocabulary.
            vector_size (int): The word embedding size.
        
        """
        super(GloVe, self).__init__()
        # word and context vector initialization
        self.w_vectors = nn.Embedding(input_size, vector_size)
        self.c_vectors = nn.Embedding(input_size, vector_size)
        # word and context bias initialization
        self.w_bias = torch.normal(0, 1, size=(input_size))
        self.c_bias = torch.normal(0, 1, size=(input_size))
        
        # model hyperparameters
        self.alpha = alpha
        self.x_max = x_max
        

    def forward(self, x, i, j):
        """Feed forward the model (used for training)
        
        Args:
            x (float): The co-occurrence of word i in context j.
            i (int): The index of the word i.
            j (int): The index of the context j.
            
        Returns:
            Tensor loss value.
        """
        # get the weight 
        f_x = (x / self.x_max)**self.alpha if x < self.x_max else 1

        # calculate w^T * c
        w_vector = self.w_vectors(i)
        c_vector = self.c_vectors(j)
        w_c = torch.matmul(w_vector, c_vector)
        
        # get the bias values
        w_bias = self.w_bias(i)
        c_bias = self.c_bias(j)
        
        # calculate the loss value
        loss = f_x * (w_c + w_bias + c_bias - torch.log(x))
        return loss
