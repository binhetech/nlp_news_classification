#!/usr/bin/env python
# coding: utf-8


import numpy as np
import pandas as pd

from sklearn.externals import joblib
from sklearn.feature_extraction.text import TfidfVectorizer

trainData = pd.read_csv("../data/train_set.csv", sep="\t")
print("trainData shape={}".format(trainData.shape))


def train_tfidf(corpus, min_df, max_features):
    tfidfv = TfidfVectorizer(analyzer="word", min_df=min_df, max_features=max_features, use_idf=True)
    tfidfv.fit(corpus)
    print("feats dim={}".format(len(tfidfv.get_feature_names())))
    with open(f"../models/tfidf-model-{min_df}-{max_features}.jlb", "wb") as f:
        joblib.dump(tfidfv, f)
    return tfidfv


def get_tfidf_feats(min_df, max_features, with_test=True):
    corpus = trainData["text"].values
    if with_test:
        corpus = np.concatenate([corpus, pd.read_csv("../data/test_a.csv", sep="\t")["text"].values], axis=0)
    print("{} lines corpus to be used".format(len(corpus)))

    tfidfvec = train_tfidf(corpus, min_df, max_features)

    def extract_feats(X):
        return tfidfvec.transform(X).toarray()

    labelsData = trainData["label"].values
    featsData = extract_feats(trainData["text"].values)

    with open(f"../data/labelsData-{min_df}-{max_features}.jlb", "wb") as f:
        joblib.dump(labelsData, f)

    with open(f"../data/featsData-{min_df}-{max_features}.jlb", "wb") as f:
        joblib.dump(featsData, f)

    print("labelsData shape={}, featsData shape={}".format(labelsData.shape, featsData.shape))


min_df = 2
max_features = 9000
get_tfidf_feats(min_df, max_features)
