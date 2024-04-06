import torch
import torch.nn as nn

from transTask import TransTask
from config import Config
from logger import Logger

if __name__ == '__main__':
    logger = Logger()
    logger.info("Start Testing...")
    task = TransTask(Config(), logger)
    example_in = "Miaoxiang son know the devil tricks, white three quick deployment."
    task.translate(example_in, mode="test")
    while True:
        example_in = input(">  ")   
        if example_in.lower() == 'exit':
            print("退出程序。")
            break
        task.translate(example_in, mode="test")