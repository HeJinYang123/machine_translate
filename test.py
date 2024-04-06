import torch
import torch.nn as nn
import sentencepiece as sp


model_path = r"./dataset/tokenizer.model"
model = sp.SentencePieceProcessor(model_path)

input = "I am a student. I love reading books."
ans = "我是学生，我爱看书。"

out = model.Encode(input, add_bos=True, add_eos=True)
out2 = model.Encode(ans, add_bos=True, add_eos=True)

print(f"input.encode: {out}")
print(f"ans.encode: {out2}")

a = [2, 3, 4]
print(f"234, {model.Decode(a)}")



