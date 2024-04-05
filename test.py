import torch
import torch.nn as nn
import sentencepiece as sp


model_path = r"./dataset/tokenizer.model"
model = sp.SentencePieceProcessor(model_path)

input = "hello world"
out = model.Encode(input, add_bos=True, add_eos=True)
print(out)
print(model.Decode(out))



