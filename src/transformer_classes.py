# Retranscription of our model definitions for downstream use
import os
import json
import re

import torch
import torch.nn as nn
from transformers import PretrainedConfig, PreTrainedModel, PreTrainedTokenizer

# List of SMILES tokens that we're going to separate into lists
SMILES_REGEX = re.compile(
    r"(\%\d\d|Br|Cl|Si|Na|Ca|Li|@@?|=|#|\(|\)|\.|\[|\]|\/|\\|:|~|\+|\-|\d|[A-Za-z])"
)

# Returns a list containing each individual token
def tokenize_smiles(smiles: str) -> list[str]:
    return SMILES_REGEX.findall(smiles)

class CuteSmileyBERTConfig(PretrainedConfig):
    model_type = "CuteSmileyBERT"
    def __init__(self, vocab_size=100, d_model=256, nhead=8, num_layers=6, dim_feedforward=1024, dropout=0.1, max_len=150, **kwargs):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout
        self.max_len = max_len

class CuteSmileyBERT(PreTrainedModel):
    config_class = CuteSmileyBERTConfig
    def __init__(self, config):
        # Intialize the nn.Module parent class, needed for proper inheritance
        super().__init__(config)

        # Set the input max_len to a class attribute
        self.max_len = config.max_len

        # This defines a method for embeddings, which will be our input representation
        # It effectively creates a weight matrix of shape [vocab_size, d_model]
        # The weights are initialized at random and will be learnt over time
        self.embed = nn.Embedding(config.vocab_size, config.d_model)
        # We will also be using positional embeddings, which will be added to the tokens'
        # They will be learnt based solely on integer positions, with no information on the tokens
        self.pos_embed = nn.Embedding(config.max_len, config.d_model)

        # Defines a single transfomer layer, with attention and feedforward/MLP layers
        # Each later contains nhead attention heads, and a ff with dimensions dim_feedforward
        # Dropout prevents overfitting by randomly setting 10% of attention coefficients to 0
        # Batch-first means that the input shape is given as [batch, seq_len, d_model]
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model, nhead=config.nhead,
            dim_feedforward=config.dim_feedforward, dropout=config.dropout, batch_first=True
        )

        # This attribute defines the model's main structure, a sequence of num_layers encoder layers
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=config.num_layers)

        # This is the final layer, a simple linear MLP which maps the embeddings to token logits
        # The input dimension is d_model, and the output is vocab_size (1-hot encoded vector)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size)

    # This is a single training cycle. 
    def forward(
        self, 
        input_ids: list[int],
        return_embeddings: bool=False
    ):
        # Get batch size and sequence length, used for positional encodings
        batch_size, seq_len = input_ids.shape
        
        # Creates a 1D tensor which simply contains the int positions of each token
        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
        # Our positions tensor is embedded, without any information on the actual tokens
        pos_embeddings = self.pos_embed(positions)
        # The positional embeddings are then summed to the token embeddings
        x = self.embed(input_ids) + pos_embeddings

        # We pass the embeddings through the encoder block
        x = self.encoder(x)
        # The linear layer outputs logits
        logits = self.lm_head(x)
        if return_embeddings: 
            return x
        return logits
    
class SMILESTokenizer(PreTrainedTokenizer):
    def __init__(self, vocab, inv_vocab, **kwargs):
        self.vocab = vocab
        self.inv_vocab = inv_vocab
        self.pad_token = "<PAD>"
        self.unk_token = "<UNK>"
        self.bos_token = "<BOS>"
        self.eos_token = "<EOS>"
        self.mask_token = "<MASK>"
        self.special_tokens = [self.pad_token, self.unk_token, self.bos_token, self.eos_token, self.mask_token]
        super().__init__(**kwargs)
        self.add_special_tokens({
            "pad_token": self.pad_token,
            "unk_token": self.unk_token,
            "bos_token": self.bos_token,
            "eos_token": self.eos_token,
            "mask_token": self.mask_token
        })

    def get_vocab(self):
        return self.vocab

    @property
    def vocab_size(self):
        return len(self.vocab)

    def _tokenize(self, text):
        return tokenize_smiles(text)

    def _convert_token_to_id(self, token):
        return self.vocab.get(token, self.vocab[self.unk_token])

    def _convert_id_to_token(self, index):
        return self.inv_vocab.get(index, self.unk_token)

    def encode(self, text, add_special_tokens=True, max_length=150):
        tokens = self._tokenize(text)
        if add_special_tokens:
            tokens = [self.bos_token] + tokens + [self.eos_token]
        token_ids = [self._convert_token_to_id(tok) for tok in tokens]
        token_ids = token_ids[:max_length] + [self.vocab[self.pad_token]] * (max_length - len(token_ids))
        return token_ids

    def decode(self, token_ids, skip_special_tokens=True):
        tokens = [self._convert_id_to_token(i) for i in token_ids]
        if skip_special_tokens:
            tokens = [t for t in tokens if t not in self.special_tokens]
        return "".join(tokens)

    def save_vocabulary(self, save_directory, filename_prefix=None):
        vocab_file = os.path.join(save_directory, (filename_prefix or "") + "vocab.json")
        with open(vocab_file, "w") as f:
            json.dump(self.vocab, f)
        return (vocab_file,)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *inputs, **kwargs):
        vocab_file = os.path.join(pretrained_model_name_or_path, "vocab.json")
        if not os.path.exists(vocab_file):
            raise FileNotFoundError(
                f"vocab.json not found in {pretrained_model_name_or_path}. "
                "If loading from the Hub, ensure the tokenizer files were pushed to the repo, "
                "or pass a local directory containing vocab.json."
            )
        with open(vocab_file, "r") as f:
            vocab = json.load(f)
            inv_vocab = {i: tok for tok, i in vocab.items()}
        return cls(vocab, inv_vocab, **kwargs)