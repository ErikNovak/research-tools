import torch
import torch.nn as nn
import torch.optim as optim
# import the transformer model
from transformers import XLMRobertaForTokenClassification, XLMRobertaTokenizer
# python types
from typing import Dict, List, Tuple, Optional


class XLMRobertaNER(nn.Module):
    """The Named Entity Recognition model using XLM-RoBERTa

    The model is using Hugging Faces 'transformers' library
    (https://huggingface.co/transformers/) and is adapted
    to be trained on any named entity task.

    Args:
        n_labels (integer): The number of labels to predict.
        model (XLMRoberta): The XLM RoBERTa model modified for token classification.
        tokenizer (XLMRobertaTokenizer): The XLM RoBERTa tokenizer.

    """
    def __init__(self, config: Dict[str, str]):
        super(XLMRobertaNER, self).__init__()
        # set the placeholder for further assignments
        self.entities = config["entities"]

        self.n_labels = len(self.entities.keys())

        # prepare the model
        self.model = XLMRobertaForTokenClassification.from_pretrained("xlm-roberta-base", num_labels=self.n_labels)
        self.model.config.label2id = self.entities
        self.model.config.id2label = { value: key for key, value in self.entities.items() }

        # prepare the tokenizer
        self.tokenizer = XLMRobertaTokenizer.from_pretrained("xlm-roberta-base")


    def forward(self, tokens_tensor: torch.Tensor, attention_mask_tensor: torch.Tensor = None, labels_tensor: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor, List[Tuple[str, str]], List[str], torch.Tensor]:
        """Get the NER predictions

        Args:
            tokens_tensor (torch.Tensor): The tokens tensor.
            attention_mask_tensor (torch.Tensor): The attention mask tensor (Default: None).
            labels_tensor (torch.Tensor): The labels_tensor (Default: None).

        Returns:
            loss (torch.Tensor): The loss value. NOTE: If labels_tensor is NOT in the input arguments,
                this value is None.
            scores (torch.Tensor): The scores/predictions for each token. Size: (1, tokens_length, 768).
            entities (List[Tuple[str, str]]): The list of entity predictions for all tokens.
            tokens (List[str]): The list of all tokens. The first and last tokens are special
                XLM-RoBERTa tokens that can be ignored.
            labels (torch.Tensor): The IDs of the most likely labels for each token. It also
                contains the labels of the start (<s>) and end (</s>) special tokens, which
                can be ignored.
        """
        # get the input values
        output = self.model(
            input_ids=tokens_tensor,
            attention_mask=attention_mask_tensor,
            labels=labels_tensor
        )
        # get the outputs
        if len(output) > 1:
            loss, scores = output[0], output[1]
        else:
            loss, scores = None, output[0]

        # get tokens and labels
        tokens = self.tokenizer.convert_ids_to_tokens(tokens_tensor[0])
        labels = torch.argmax(scores, dim=2)[0]

        # get named entities
        entities = self.format_named_entities(tokens[1:-1], labels[1:-1])

        # convert the labels into actual entity tags
        labels = [self.model.config.id2label[label.item()] for label in labels]

        # return the (loss), scores, entities, tokens, labels
        return (loss, scores, entities, tokens, labels)


    def tokenization(self, text: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """Performs tokenization of the provided text (limit 512 tokens)

        Args:
            text (string): The text to be tokenized.

        Returns:
            tokens_tensors (Tensor): The tensor containing the indexed tokens.
            segments_tensors (Tensor): The tensor containing the segments ids.

        """
        # perform tokenization and segment IDs
        sequence_text = self.tokenizer.encode_plus(text, max_length=self.tokenizer.max_len)

        # create the tokens and attention mask tensors
        tokens_tensor = torch.tensor([sequence_text["input_ids"]])
        attention_mask_tensor = torch.tensor([sequence_text["attention_mask"]])

        return tokens_tensor, attention_mask_tensor


    def get_train_tokens_attention_labels_tensor(self, sentence: List[Dict[str, str]]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Converts the sentence tokens array into tensors

        Args:
            sentence (List[Dict[str, str]]): The list of labelled word in a sentence.
                The format is of each dict object is:
                    { 'word': str, 'pos': str, 'chunk': str, 'ner': str }

        Returns:
            tokens_tensor (torch.Tensor): The tokens tensor.
            attention_mask_tensor (torch.Tensor): The attention mask tensor.
            labels_tensor (torch.Tensor): The labels tensor.
        """
        tokens = []
        labels = []
        for entry in sentence:
            tmp_tokens = self.tokenizer.tokenize(entry["word"])
            labels += [self.model.config.label2id[entry["ner"]] for i in range(len(tmp_tokens))]
            tokens += tmp_tokens

        # create the tokens tensor
        tokens = ["<s>"] + tokens + ["</s>"]
        tokens = self.tokenizer.convert_tokens_to_ids(tokens)
        tokens_tensor = torch.tensor([tokens])

        # create the attentions mask tensor
        attention = [1] * len(tokens)
        attention_mask_tensor = torch.tensor([attention])

        # create the labels tensor
        labels = [0] + labels + [0]
        labels_tensor = torch.tensor(labels)

        # return the tokens, attention mask and labels tensors
        return tokens_tensor, attention_mask_tensor, labels_tensor


    def format_named_entities(self, tokens: List[str], labels: torch.Tensor) -> List[Tuple[str, str]]:
        """Formats and joins the tokens and labels

        Args:
            tokens (List[str]): The list of tokenized words.
            labels (torch.Tensor): The token labels extracted from the model.
                The labels are generated with the following function:
                `labels = torch.argmax(scores, dim=2)[0]` where `scores` are
                the label scores provided by `self.model`.

        Returns:
            ner (List[Tuple[str, str]]): The list of all Named Entity pairs.
        """

        def format_token(token: str) -> str:
            """Formats the token by removing the underscore

            Args:
                token (str): The token.

            Returns:
                str: The formatted string.

            """
            return token.replace("▁", "")

        # initialize the named entity array
        entities = []
        # initialize the variables with the first token and label
        tk = format_token(tokens[0])
        lb = self.model.config.id2label[labels[0].item()]

        # iterate through the tokens, labels pairs
        for token, label in zip(tokens[1:], labels[1:]):

            # get the current label
            l = self.model.config.id2label[label.item()]

            # if the token is a new word or if the label
            # has changed, then save the previous NER example
            # and reset the token variable
            if "▁" in token or l != lb:
                entities.append((tk.strip(), lb))
                tk = ""

            # merge and update the token and label, respectively
            tk += format_token(token)
            lb = l


        # add the last named entity pair
        entities.append((tk.strip(), lb))

        # return the list of named entities
        return entities