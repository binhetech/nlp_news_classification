import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

import fasttext

trainData = pd.read_csv("../data/train_set.csv", sep="\t", nrows=1000)
print("trainData shape={}".format(trainData.shape))

testAData = pd.read_csv("../data/test_a.csv", sep="\t", nrows=100)
print("testAData shape={}".format(testAData.shape))


def gen_train_test_file(data, set):
    if set == "train":
        data["label"] = data["label"].map(lambda x: f"__label__{str(x)}")
    data.to_csv(f"../data/{set}-fasttext.txt")


def evaluate(y_true, y_pred):
    return f1_score(y_true, y_pred, average='macro')


gen_train_test_file(trainData, "train")
gen_train_test_file(testAData, "test")

lr = 0.5
epoch = 20
wordNgrams = 2
model = fasttext.train_supervised(input="../data/train-fasttext.txt", lr=lr, epoch=epoch, wordNgrams=wordNgrams)
print("model.labels={}".format(model.labels))
model.save_model("../models/fasttext-model.bin")

y_pred = model.test("../data/test-fasttext.txt")
outs = pd.DataFrame({"label": y_pred})
outs.to_csv(f"../data/test_a_predict-fasttext-{lr}-{epoch}-{wordNgrams}.csv", index=False)
