import torch
import torch.nn as nn
from .transformer import TransformerEncoder, TransformerDecoder, PositionEncoding


class MTranslate(nn.Module):
    def __init__(self, vocabulary_size, d_model=512, n_head=8, n_layers=6, d_ff=2048, drop_out=0.1):
        super(MTranslate, self).__init__()
        self.in_embedding = nn.Embedding(vocabulary_size, d_model)
        self.out_embedding = nn.Embedding(vocabulary_size, d_model)

        self.encoder = TransformerEncoder(d_model, n_head, d_ff, drop_out)
        self.decoder = TransformerDecoder(d_model, n_head, d_ff, drop_out)

        self.encoder_stk = nn.ModuleList([self.encoder for _ in range(n_layers)])
        self.decoder_stk = nn.ModuleList([self.decoder for _ in range(n_layers)])

        self.pe = PositionEncoding(d_model, drop_out=drop_out)

        self.generator = nn.Sequential(
            nn.Linear(d_model, vocabulary_size),
            nn.LogSoftmax(dim=-1)
        )

    def forward(self, src, trg, src_mask, trg_mask):
        # src_mask = (src == 0).unsqueeze(1).unsqueeze(2)
        # trg_mask = (trg == 0).unsqueeze(1).unsqueeze(2)
        src = self.pe(self.in_embedding(src))
        trg = self.pe(self.out_embedding(trg))

        for encoder in self.encoder_stk:
            src = encoder(src, src_mask)
        for decoder in self.decoder_stk:
            trg = decoder(trg, src, trg_mask, src_mask)

        return self.generator(trg)
