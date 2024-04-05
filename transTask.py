import os.path
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.nn.modules.module import T

from config import Config

from models.mtranslate import MTranslate
from data.dataloader import TranslationDataset, Tokenizer, DataCollator
from data.tokenizer import build_vocab


class TransTask:
    def __init__(self, config: Config, logger):
        super(TransTask, self).__init__()
        self.cfg = config
        self.logger = logger
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = MTranslate(vocabulary_size=config.vocab_size)

        if not os.path.exists(config.tokenizer):
            build_vocab(config.json_file_path, "./dataset")
        self.tokenizer = Tokenizer(config.tokenizer)
        train_dataset = TranslationDataset(config.en_path, config.zh_path, self.tokenizer)
        self.dataloader = torch.utils.data.DataLoader(
            train_dataset, config.batch_size, shuffle=True, collate_fn=DataCollator(self.tokenizer)
        )

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config.learning_rate)
        self.loss_func = nn.CrossEntropyLoss()

    def train(self):
        for epoch in range(self.cfg.epoch):
            self.model.train()
            for i, item in enumerate(tqdm(self.dataloader)):
                src = item["input"]
                trg = item["label"]
                print(f"src word: {self.tokenizer.decode(src[0].tolist())}")
                print(f"trg word: {self.tokenizer.decode(trg[0].tolist())}")
                output = self.model(src, trg)
                print(f"output word: {self.tokenizer.decode(output[0].argmax(dim=-1).tolist())}")
                loss = self.loss_func(output.view(-1, output.size(-1)), trg.view(-1))

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                if i % (len(self.dataloader) // 1000) == 0:
                    print(f"Epoch: {epoch}, Iter: {i}, Loss: {loss.item()}")

                if i % (len(self.dataloader) // 100) == 0:
                    self.translate("I am a student. I love reading books.")

    def translate(self, src):
        """src: str，输入英文句子，输出中文句子"""
        self.model.eval()
        self.logger.info(f"Test translate, input: {src}")
        src = torch.tensor(self.tokenizer.encode(src)).unsqueeze(0)
        trg = torch.tensor([[0]]).to(self.device)
        for i in range(10):
            output = self.model(src, trg)
            pred = output.argmax(dim=-1)[:, -1].unsqueeze(1)
            trg = torch.cat((trg, pred), dim=1)
            if pred[0, -1].item() == 1:
                break
        self.logger.info(f"Output: {self.tokenizer.decode(trg[0].tolist())}")
        self.model.train()
        return self.tokenizer.decode(trg[0].tolist())


def encoder_mask(src):
    return (src == 0).unsqueeze(1).unsqueeze(2)
