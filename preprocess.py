import sentencepiece as sp


import json

english = []
chinese = []
train_data = "/Users/hjy/Downloads/translation2019zh/translation2019zh_valid.json"
with open(train_data, "r") as fo:
    for line in fo:
        data = json.loads(line)
        print(data)
        break