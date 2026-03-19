from typing import List, Union
import pytorch_lightning as pl

import torch.nn as nn
import os

import torch
from torch import Tensor
from torch.distributions.distribution import Distribution
from transformers import AutoTokenizer, AutoModel
from transformers import logging
from sentence_transformers import SentenceTransformer
from peft import get_peft_model, LoraConfig, TaskType
from transformers import CLIPProcessor, CLIPModel, T5Tokenizer, T5EncoderModel
import re

class DistilbertEncoderBase(pl.LightningModule):
    """
    This class is a base encoder for DistilBERT models.

    Attributes:
    - tokenizer: the tokenizer for the pre-trained DistilBERT model.
    - text_model: the pre-trained DistilBERT model.
    - text_encoded_dim: the dimension of the hidden state in the DistilBERT model.

    Methods:
    - __init__: initializes the DistilbertEncoderBase object with the given parameters.
    - train: sets the training mode for the model.
    """

    def __init__(self, modelpath: str, finetune: bool = False):
        """
        Initializes the DistilbertEncoderBase object with the given parameters.

        Inputs:
        - modelpath: the path to the pre-trained DistilBERT model.
        - finetune: a flag indicating whether to fine-tune the DistilBERT model.

        Outputs: None
        """
        super().__init__()
        logging.set_verbosity_error()

        # Tokenizer
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        self.text_model_name ='t5-base'
        self.text_tokenizer = T5Tokenizer.from_pretrained(self.text_model_name)
        self.text_model = T5EncoderModel.from_pretrained(self.text_model_name)

       
        # special_tokens_to_add = ['[LEFT]', '[RIGHT]', '[TWO_HANDS_RELATION]']
        # self.text_tokenizer.add_special_tokens({'additional_special_tokens': special_tokens_to_add})
        # self.text_model.resize_token_embeddings(len(self.text_tokenizer))
        
        # lora_config = LoraConfig(
        #     task_type=TaskType.FEATURE_EXTRACTION,
        #     r=8,
        #     lora_alpha=32,
        #     target_modules=["q", "v"],
        #     lora_dropout=0.1,
        #     bias="none",
        # )

        
        # self.tokenizer = AutoTokenizer.from_pretrained(modelpath)

        # # Text model
        # self.text_model = AutoModel.from_pretrained(modelpath)

        # Don't train the model
        # if not finetune:
        self.text_model.training = False
        for p in self.text_model.parameters():
            p.requires_grad = False

        # Then configure the model
        self.text_encoded_dim = self.text_model.config.hidden_size

    def train(self, mode: bool = True):
        """
        Sets the training mode for the model.

        Inputs:
        - mode: a flag indicating whether to set the model to training mode.

        Outputs: None
        """
        self.training = mode
        for module in self.children():
            # Don't put the model in
            if module == self.text_model and not self.hparams.finetune:
                continue
            module.train(mode)
        return self

    def get_last_hidden_state(self, texts: List[str], return_mask: bool = False):
        """
        Sets the training mode for the model.

        Inputs:
        - mode: a flag indicating whether to set the model to training mode.

        Outputs: None
        """
        text_inputs = self.text_tokenizer(
                texts, padding=True, truncation=True,
                return_tensors='pt'
            )
        text_inputs = {key: value.to(self.text_model.device) for key, value in text_inputs.items()}
        outputs = self.text_model(**text_inputs)
        return (outputs.last_hidden_state), text_inputs['attention_mask'].to(dtype=bool)
        
        # Tokenize the texts and convert them to tensors
        # encoded_inputs = self.tokenizer(texts, return_tensors="pt", padding=True)

        # # Pass the encoded inputs to the DistilBERT model
        # output = self.text_model(**encoded_inputs.to(self.text_model.device))

        # # If not returning the attention mask, return the last hidden state
        # if not return_mask:
        #     return output.last_hidden_state

        # # If returning the attention mask, return the last hidden state and the attention mask
        # return output.last_hidden_state, encoded_inputs.attention_mask.to(dtype=bool)
