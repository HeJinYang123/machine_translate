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
        self.model = MTranslate(vocabulary_size=config.vocab_size).to(self.device)
        self.test_model = None

        if not os.path.exists(config.tokenizer):
            build_vocab(config.json_file_path, "./dataset", self.cfg.vocab_size)
        self.tokenizer = Tokenizer(config.tokenizer)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config.learning_rate)
        self.loss_func = nn.CrossEntropyLoss()
        self.start_epoch = 0
        if self.cfg.train_continue_path is not None:
            checkpoint = torch.load(self.cfg.train_continue_path)
            self.model.load_state_dict(checkpoint["state_dict"])
            self.optimizer.load_state_dict(checkpoint["optimizer"])
            self.start_epoch = checkpoint["epoch"]

    def train(self):
        train_dataset = TranslationDataset(self.cfg.en_path, self.cfg.zh_path, self.tokenizer)
        self.dataloader = torch.utils.data.DataLoader(
            train_dataset, self.cfg.batch_size, shuffle=True, collate_fn=DataCollator(self.tokenizer)
        )
        for epoch in range(self.start_epoch, self.cfg.epoch):
            self.model.train()
            for i, item in enumerate(tqdm(self.dataloader)):
                src = item["input"].to(self.device)
                trg = item["label"].to(self.device)
                # print(f"src word: {self.tokenizer.decode(src[0].tolist())}")
                # print(f"trg word: {self.tokenizer.decode(trg[0].tolist())}")

                # 训练任务，用src全体+trg前(len-1)个词，预测trg后(len-1)词
                src_mask = make_src_mask(src, self.tokenizer.model, self.device)
                trg_mask = make_trg_mask(trg[:, :-1], self.tokenizer.model, self.device)
                output = self.model(src, trg[:, :-1], src_mask, trg_mask)
                # print(f"output word: {self.tokenizer.decode(output[0].argmax(dim=-1).tolist())}")
                output = output.contiguous().view(-1, output.size(-1))
                trg = trg[:,1:].contiguous().view(-1)
                loss = self.loss_func(output, trg)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                if i % 100 == 0:
                    self.logger.info(f"Epoch: {epoch}, Iter: {i}, Loss: {loss.item()}")

                if i % (len(self.dataloader) // 10) == 0:
                    self.translate("He calls the Green Book, his book of teachings, “the new gospel.")

            self._save_checkpoint(epoch, loss.item())
            
            self.translate("I am a student. I love reading books.")
                

    def _save_checkpoint(self, epoch, loss, save_best=False):
        state = {
            'epoch': epoch + 1,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'config': self.cfg,
        }
        filename = str("./experience/" + f'checkpoint_{epoch}_{loss:.5f}.pt')
        torch.save(state, filename)
        self.logger.debug(f'Saving checkpoint: {filename} ...')

    def translate(self, src, mode="train"):
        """src: str, 输入英文句子, 输出中文句子"""
        if mode == "train":
            model = self.model.eval()
        elif self.test_model == None:
            model = MTranslate(vocabulary_size=self.cfg.vocab_size).to(self.device)
            checkpoint = torch.load(self.cfg.Test.checkpoint)
            model.load_state_dict(checkpoint["state_dict"])
            self.test_model = model
        else:
            model = self.test_model
        self.logger.info(f"Test translate, input: {src}")
        src = torch.tensor(self.tokenizer.encode(src, add_bos=True, add_eos=True)).unsqueeze(0).to(self.device)
        src_mask = make_src_mask(src, self.tokenizer.model, self.device)
        self.logger.info(f"src.encode: {src[0]}")
        trg = torch.tensor([[2]]).to(self.device)
        for i in range(self.cfg.max_len):
            trg_mask = make_trg_mask(trg, self.tokenizer.model, self.device)
            output = model(src, trg, src_mask, trg_mask)
            pred = output.argmax(dim=-1)[:, -1].unsqueeze(1)
            # print(f"pred: {output[0].argmax(dim=-1)}")
            trg = torch.cat((trg, pred), dim=1)
            # pred[0, -1] == <EOS>
            if pred[0, -1].item() == self.tokenizer.model.eos_id():
                break
        self.logger.info(f"Output: {self.tokenizer.decode(trg[0].tolist())}")
        self.model.train()
        return self.tokenizer.decode(trg[0].tolist())


def make_src_mask(src, tokenizer, device):
    src_mask = (src != tokenizer.pad_id()).unsqueeze(1).unsqueeze(2).to(device)
    return src_mask

def make_trg_mask(trg, tokenizer, device):
    # 1. mask掉 padding 占位符
    trg_pad_mask =  (trg != tokenizer.pad_id()).unsqueeze(1).unsqueeze(2)
    trg_len = trg.shape[1]
    trg_sub_mask = torch.tril(torch.ones((trg_len, trg_len), device=device)).bool()
    trg_mask = trg_pad_mask & trg_sub_mask
    return trg_mask
