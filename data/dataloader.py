from dataclasses import dataclass
from typing import List, Dict

import torch
import torch.nn as nn
import sentencepiece as sp

from tqdm import tqdm


class Tokenizer:
    def __init__(self, model_path):
        self.model = sp.SentencePieceProcessor(model_path)

    def encode(self, texts: List[str], add_bos=False, add_eos=False, return_tensor=False) -> List[int]:
        val = self.model.Encode(input=texts, out_type=int, add_bos=add_bos, add_eos=add_eos)
        if return_tensor:
            torch.LongTensor(val)
        return val

    def decode(self, text):
        return self.model.Decode(text)

    def get_vocab_size(self):
        return self.model.vocab_size()


class TranslationDataset(torch.utils.data.Dataset):
    def __init__(self, en_path, zh_path, tokenizer: Tokenizer):
        en = open(en_path, 'r').readlines()
        zh = open(zh_path, 'r').readlines()
        assert len(en) == len(zh), "en/zh size not match."
        self.length = len(en)

        self.items = []
        for i in tqdm(range(self.length)):
            # text1 = en[i].replace(' ', '').strip()
            text1 = en[i].strip()
            text2 = zh[i].strip()

            input = tokenizer.encode(text1)
            labels = tokenizer.encode(text2, add_bos=True, add_eos=True)
            item = {"input": input, "label": labels}
            self.items.append(item)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        return self.items[idx]


@dataclass
class DataCollator:
    """padding补全等"""
    tokenizer: Tokenizer
    padding = 0
    max_length: int = 64

    def __call__(self, features: List[Dict[str, List[int]]]) -> Dict[str, torch.Tensor]:
        batch_size = len(features)
        if batch_size == 0:
            return
        features_in_bucket = {}
        for item in features:
            for key, value in item.items():
                temp = features_in_bucket.get(key, [])
                temp.append(value)
                features_in_bucket[key] = temp
        batch = {}
        for key, value_list in features_in_bucket.items():
            batch[key] = padding(data_list=value_list, pad_id=self.padding, max_length=self.max_length,
                                 return_tensors="pt")
        return batch


def padding(data_list: List[List[int]], max_length=128, pad_id=0, return_tensors='pt') -> List[List[int]]:
    max_len = -1
    for token_id_list in data_list:
        if max_len < len(token_id_list):
            max_len = len(token_id_list)
    max_len = min(max_len, max_length)
    for ndx, token_id_list in enumerate(data_list):
        if len(token_id_list) < max_len:
            data_list[ndx].extend([pad_id] * (max_len - len(token_id_list)))
        elif len(token_id_list) > max_len:
            data_list[ndx] = data_list[ndx][0:max_len]
    if return_tensors == 'pt':
        return torch.Tensor(data_list).long()
    return data_list
