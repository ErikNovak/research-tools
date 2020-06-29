import torch
import torch.nn as nn
import torch.nn.functional as F

class DM(nn.Module):
    """Distributed Memory version of Paragraph Vectors.

    Parameters:
        document_size (int): The size of the document matrix.
        vocabulary_size (int): The size of the vocabulary.
        hidden_size (int): The hidden layer size (Default: 300).
    """

    def __init__(self, document_size, vocabulary_size, window_size, hidden_size=300):
        super(DM, self).__init__()
        self.document_size = document_size
        self.hidden_size = hidden_size
        self.doc_embedding = nn.Embedding(self.document_size, hidden_size)
        self.voc_embedding = nn.Embedding(vocabulary_size, hidden_size, padding_idx=0)
        self.h2o = nn.Linear(hidden_size * (1 + window_size * 2), vocabulary_size)
        self.softmax = nn.LogSoftmax(dim=2)


    def forward(self, doc_input, words_input):
        """Do a forward propagation on the given document and words

        Parameters:
            doc_input (nn.Tensor): The tensor containing the document id.
            words_input (nn.Tensor): The tensor containing the document
                word ids.

        Returns:
            output (nn.Tensor): The tensor describing the probability of
                predicting the correct word.
        """

        doc_embedded = self.doc_embedding(doc_input)
        voc_embedded = self.voc_embedding(words_input)

        batch, window_size, values = voc_embedded.size()
        voc_concatenate = voc_embedded.view(batch, 1, -1)
        # concat and average the document and vocabulary embeddings
        output = torch.cat((doc_embedded, voc_concatenate), dim=2)
        output = self.h2o(output)
        output = self.softmax(output)
        return output


    def lock(self):
        """Locks all parameters, disabling gradient calculation"""

        for params in self.voc_embedding.parameters():
            params.requires_grad = False
        for params in self.h2o.parameters():
            params.requires_grad = False


    def unlock(self):
        """Unlocks all parameters for calculating the gradient"""

        for params in self.doc_embedding.parameters():
            params.requires_grad = True
        for params in self.voc_embedding.parameters():
            params.requires_grad = True
        for params in self.h2o.parameters():
            params.requires_grad = True


    def add_document_tensor(self, doc_tensor=None, device=None):
        """Add a new document embedding to the existing ones

        Parameters:
            doc_tensor (nn.Tensor): The tensor to be added to the document
                embedding space. If None, it adds a new random tensor
                of size (1, self.hidden_size) (Default: None).

            non_trainable (bool): If True, the new document embedding
                layer will not be able to compute the gradient
                (Default: False).

        Returns:
            document_id (int): The index number of the new document embedding.
        """

        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if doc_tensor is None:
            # assign the document tensor to be added to the doc embeddings
            doc_tensor = torch.Tensor(1, self.hidden_size).to(device)
            # set values from normal distribution
            doc_tensor.normal_()

        # add a new entry in the document embedding tensor
        doc_data = self.doc_embedding.weight.data
        self.doc_embedding = self.doc_embedding.from_pretrained(
            torch.cat((doc_data, doc_tensor), dim=0)
        )
        for params in self.doc_embedding.parameters():
            params.requires_grad = True

        # increase the document size
        document_id = self.document_size
        self.document_size += 1
        # return the document index
        return document_id


class DBOW(nn.Module):
    """Distributed Bag-of-Words Model.

    Parameters:
        document_size (int): The number of documents to embed.
        hidden_size (int): The document embedding size (Default: 300).
        voc_embedding (np.array): The vocabulary embedding weights (Default: None).
        voc_size (int): The size of vocabulary, used only when voc_embedding
            attribute is not present (Default: 100000).
    """
    def __init__(self, document_size, hidden_size=300, voc_embedding=None, voc_size=100000):
        super(DBOW, self).__init__()
        self.document_size = document_size
        self.hidden_size = hidden_size
        self.doc_embedding = nn.Embedding(document_size, hidden_size)

        if voc_embedding:
            self.set_voc_embedding(voc_embedding)
        else:
            self.voc_embedding = nn.Embedding(voc_size, self.hidden_size)


    def set_voc_embedding(self, weight_matrix: np.array) -> None:
        """Set the vocabulary embedding

        Parameters:
            weight_matrix: The numpy array containing the vocabulary
                embedding values.
        """
        vocabulary_size, _ = weight_matrix.shape
        self.voc_embedding = nn.Embedding(vocabulary_size, self.hidden_size)
        self.voc_embedding.load_state_dict({ "weight": weight_matrix })
        for params in self.voc_embedding.parameters():
            params.requires_grad = False


    def forward(self, document_ids, positive_ids, negative_ids):
        """Do a forward propagation on the given document

        Parameters:
            doc_input (nn.Tensor): The tensor containing the document id.

        Returns:
            loss (nn.Tensor): The loss value of the feed-forward.
        """
        doc_embedded = self.doc_embedding(document_ids)

        batch_size, _, _ = doc_embedded.size()
        # calculate the positive loss score
        pos_embedded = self.voc_embedding(positive_ids)
        pos_score = torch.bmm(doc_embedded, torch.transpose(pos_embedded, 1, 2))
        pos_score = torch.sum(pos_score, dim=1)
        pos_score = F.logsigmoid(pos_score)

        # calculate the negative loss score
        neg_embedded = self.voc_embedding(negative_ids)
        neg_score = torch.bmm(neg_embedded, torch.transpose(doc_embedded, 1, 2))
        neg_score = F.logsigmoid(-1*neg_score)
        neg_score = torch.sum(neg_score, dim=1)

        # sum up the positive and negative score
        loss = pos_score + neg_score

        # return the loss sum
        return -1 * loss.sum() / batch_size


    def lock(self) -> None:
        """Locks all parameters, disabling gradient calculation"""
        for params in self.doc_embedding.parameters():
            params.requires_grad = False


    def unlock(self) -> None:
        """Unlocks all parameters for calculating the gradient"""
        for params in self.doc_embedding.parameters():
            params.requires_grad = True


    def add_document_tensor(self, doc_tensor: torch.Tensor = None,
                            device: torch.device = torch.device("cpu")) -> int:
        """Add a new document embedding to the existing ones

        Args:
            doc_tensor: The tensor to be added to the document
                embedding space. If None, it adds a new random tensor
                of size (1, self.hidden_size) (Default: None).
            non_trainable: If True, the new document embedding
                layer will not be able to compute the gradient
                (Default: False).
            device: The device to which the model copies the
                new tensor to.

        Returns:
            The index number of the new document embedding.
        """
        if doc_tensor is None:
            # assign the document tensor to be added to the doc embeddings
            doc_tensor = torch.Tensor(1, self.hidden_size).to(device)
            # set values from normal distribution
            doc_tensor.normal_()

        # add a new entry in the document embedding tensor
        doc_data = self.doc_embedding.weight.data
        self.doc_embedding = self.doc_embedding.from_pretrained(
            torch.cat((doc_data, doc_tensor), dim=0)
        )
        for params in self.doc_embedding.parameters():
            params.requires_grad = True

        # increase the document size
        document_id = self.document_size
        self.document_size += 1

        # return the document index
        return document_id
