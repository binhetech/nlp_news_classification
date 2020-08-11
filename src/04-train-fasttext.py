import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from sklearn.model_selection import train_test_split, StratifiedKFold
import fasttext


def gen_train_test_file(data, set):
    if set == "train":
        data["label"] = data["label"].map(lambda x: f"__label__{str(x)}")
        traind, devd = train_test_split(data, random_state=42, stratify=data["label"], test_size=0.1)
        traind.to_csv(f"../data/train-fasttext.txt", index=False, sep=" ")
        devd.to_csv(f"../data/dev-fasttext.txt", index=False, sep=" ")
        print("train shape={}".format(traind.shape))
        print("dev shape={}".format(devd.shape))
    else:
        data.to_csv(f"../data/{set}-fasttext.txt", index=False, sep=" ")


def evaluate(y_true, y_pred):
    return f1_score(y_true, y_pred, average='macro')


def write_file():
    trainData = pd.read_csv("../data/train_set.csv", sep="\t")
    print("trainData shape={}".format(trainData.shape))

    testAData = pd.read_csv("../data/test_a.csv", sep="\t")
    print("testAData shape={}".format(testAData.shape))

    gen_train_test_file(trainData, "train")
    gen_train_test_file(testAData, "test")


def train():
    lr = 0.5
    epoch = 20
    wordNgrams = 2
    model = fasttext.train_supervised(input="../data/train-fasttext.txt",
                                      autotuneValidationFile='../data/dev-fasttext.txt',
                                      autotuneDuration=1200)
    print("{} words".format(len(model.words)))
    print("{} labels={}".format(len(model.labels), model.labels))
    model.save_model("../models/fasttext-model.bin")

    X_test = pd.read_csv("../data/test-fasttext.txt")["text"]
    y_pred = [model.predict(i)[0][0][9:] for i in X_test]
    outs = pd.DataFrame({"label": y_pred})
    outs.to_csv(f"../data/test_a_predict-fasttext-{lr}-{epoch}-{wordNgrams}.csv", index=False)


def test():
    model = fasttext.load_model("../models/fasttext-model.bin")
    out = model.test("../data/dev-fasttext.txt")
    print(out)


write_file()
train()
test()
