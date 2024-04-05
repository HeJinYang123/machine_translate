class Config:
    json_file_path = f"/Users/hjy/Downloads/translation2019zh/translation2019zh_valid.json"
    en_path = f"./dataset/english.txt"
    zh_path = f"./dataset/chinese.txt"
    tokenizer = f"./dataset/tokenizer.model"
    vocab_size = 5000
    max_len = 100

    batch_size = 4
    learning_rate = 1e-4

    epoch = 10