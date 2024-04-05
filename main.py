import torch
import torch.nn as nn

from transTask import TransTask
from config import Config
from logger import Logger

if __name__ == '__main__':
    logger = Logger()
    logger.info("Start training...")
    task = TransTask(Config(), logger)
    task.train()