import torch
import torch.nn as nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from transformers.modeling_outputs import Seq2SeqLMOutput
from dataclasses import dataclass
import math

@dataclass
class ModelConfig:
    pad_token: int 
    eos_token: int 
    vocab_size : int 
    d_model: int = 128*2
    n_head:int = 4
    num_encoder_layers : int = 5
    num_decoder_layers : int = 1
    dim_feedforward : int = 128*4
    is_encoder_decoder: bool=True

class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.0, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)
    
class MyEncoderDecoderModelForSeq2SeqLM(nn.Module):    
    def _shift_right(self, input_ids):
        # shift inputs to the right
        shifted_input_ids = input_ids.new_zeros(input_ids.shape)
        shifted_input_ids[..., 1:] = input_ids[..., :-1].clone()
        shifted_input_ids[..., 0] = self.config.pad_token

        if self.config.pad_token is None:
            raise ValueError("self.model.config.pad_token_id has to be defined.")
        # replace possible -100 values in labels by `pad_token_id`
        shifted_input_ids.masked_fill_(shifted_input_ids == -100, self.config.pad_token)

        return shifted_input_ids

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.transformer = nn.Transformer(config.d_model, config.n_head, 
                                          config.num_encoder_layers, config.num_decoder_layers,
                                          config.dim_feedforward, batch_first=True)
        
        self.shared = nn.Embedding(config.vocab_size, config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.pe = PositionalEncoding(config.d_model)
        
    def forward(self, input_ids, decoder_input_ids=None, labels=None, attention_mask=None):
        input_embeds = self.shared(input_ids)
        input_embeds = self.pe(input_embeds)
        if decoder_input_ids is None:
            decoder_input_ids = self._shift_right(labels)
        decoder_input_embeds = self.shared(decoder_input_ids)
        
        mask = self.transformer.generate_square_subsequent_mask(
            decoder_input_ids.shape[1], 
            input_ids.device
        )
        sequence_output = self.transformer(input_embeds, decoder_input_embeds,
                               tgt_mask=mask,
                               tgt_is_causal=True)
        # print(decoder_input_embeds.shape, labels.shape, sequence_output.shape)
        lm_logits = self.lm_head(sequence_output)
        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss(ignore_index=-100)
            # move labels to correct device to enable PP
            labels = labels.to(lm_logits.device)
            loss = loss_fct(lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1))
            # TODO(thom): Add z_loss https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/layers.py#L666
        return Seq2SeqLMOutput(
            loss=loss,
            logits=lm_logits
        )
    
    @torch.no_grad()
    def generate(
        self,
        input_ids, 
        max_length=30
    ):
        decoder_input_ids = torch.tensor([self.config.pad_token]).to(input_ids.device)
        while decoder_input_ids[-1] != self.config.eos_token and len(decoder_input_ids)< max_length:
            res = self.forward(input_ids.unsqueeze(0), decoder_input_ids.unsqueeze(0)).logits[0, -1]
            selected_token = torch.argmax(res)
            decoder_input_ids = torch.tensor(decoder_input_ids.tolist() + [selected_token]).to(res.device)
            
        return decoder_input_ids