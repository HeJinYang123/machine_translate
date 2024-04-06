import torch

import os
import json
import sentencepiece as sp


def build_vocab(json_data_path, save_path, vocab_size=20000):
    english = []
    chinese = []
    max_sentence = 1000000
    cnt=0
    with open(json_data_path, "r") as fo:
        for line in fo:
            cnt+=1
            if(cnt>max_sentence):
                break
            data = json.loads(line)
            english.append(data["english"])
            chinese.append(data["chinese"])
    with open(os.path.join(save_path, "english.txt"), "w") as fo:
        for line in english:
            fo.write(line + "\n")
    with open(os.path.join(save_path, "chinese.txt"), "w") as fo:
        for line in chinese:
            fo.write(line + "\n")
    with open(os.path.join(save_path, "zh&en.txt"), "w") as fo:
        for line in english:
            fo.write(line + "\n")
        for line in chinese:
            fo.write(line + "\n")
    sp.SentencePieceTrainer.Train(
        input=r"./dataset/zh&en.txt",
        model_prefix=os.path.join(save_path, 'tokenizer'),
        vocab_size=vocab_size,
        model_type='bpe',
        pad_id=0, unk_id=1, bos_id=2, eos_id=3
    )
    # sp.SentencePieceTrainer.Train(
    #     input='../dataset/chinese.txt',
    #     model_prefix=os.path.join(save_path, 'chineseTokenizer'),
    #     vocab_size=ch_vocab_size,
    #     model_type='bpe',
    #     pad_id=0, unk_id=1, bos_id=2, eos_id=3
    # )

