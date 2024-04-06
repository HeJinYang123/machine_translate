import time


current_t = time.strftime("%m%d_%H")
class Config:
    json_file_path = f"./dataset/translation2019zh_train.json"
    en_path = f"./dataset/english.txt"
    zh_path = f"./dataset/chinese.txt"
    tokenizer = f"./dataset/tokenizer.model"
    vocab_size = 5000
    max_len = 100

    batch_size = 128
    learning_rate = 1e-4

    epoch = 10

    train_continue_path = r"experience/checkpoint_5_1.41370.pt"
    class Test:
        checkpoint = r"experience/checkpoint_9_1.26833.pt"

    experience_save_path = None