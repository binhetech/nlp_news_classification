#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd

import time

from sklearn.externals import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold, cross_val_score
from sklearn.model_selection import GridSearchCV

from sklearn.ensemble import RandomForestClassifier
import lightgbm as lgb
import xgboost as xgb

from sklearn.metrics import f1_score

min_df = 2
max_features = 9000
with open(f"../data/labelsData-{min_df}-{max_features}.jlb", "rb") as f:
    labelsData = joblib.load(f)
with open(f"../data/featsData-{min_df}-{max_features}.jlb", "rb") as f:
    featsData = joblib.load(f)
print("labelsData shape={}, featsData shape={}".format(labelsData.shape, featsData.shape))

with open(f"../models/tfidf-model-{min_df}-{max_features}.jlb", "rb") as f:
    tfidfvec = joblib.load(f)

testAData = pd.read_csv("../data/test_a.csv", sep="\t")


def extract_feats(X):
    return tfidfvec.transform(X).toarray()


X_test = extract_feats(testAData["text"].values)


def evaluate(y_true, y_pred):
    return f1_score(y_true, y_pred, average='macro')


def train_test(X_train, y_train, X_test, y_test, model):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return evaluate(y_test, y_pred)


def run_cross_validate():
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    model = RandomForestClassifier(n_estimators=300, n_jobs=16)
    results = []
    for train_index, test_index in skf.split(featsData, labelsData):
        X_train, y_train = featsData[train_index], labelsData[train_index]
        X_test, y_test = featsData[test_index], labelsData[test_index]
        print("X_train shape={}, y_train shape={}".format(X_train.shape, y_train.shape))
        print("X_test shape={}, y_test shape={}".format(X_test.shape, y_test.shape))
        f1 = train_test(X_train, y_train, X_test, y_test, model)
        results.append(f1)
    print("mean={}, f1_macro={}".format(np.mean(results), results))


def run_predict(model, name, min_df=3, max_features=5000):
    tStart = time.time()
    y_pred = model.predict(X_test)
    outs = pd.DataFrame({"label": y_pred})
    outs.to_csv(f"../data/test_a_predict-{name}-{min_df}-{max_features}.csv", index=False)
    print("{}: predict completed, time={}".format(name, time.time() - tStart))


def re_train(model, name, X_train, y_train):
    tStart = time.time()
    model.fit(X_train, y_train)
    print("{}: re-train completed, time={}".format(name, time.time() - tStart))
    return model


def run_cv(model, name, with_cv=True):
    if with_cv:
        tStart = time.time()
        cv = cross_val_score(model, featsData, labelsData, cv=skf, scoring="f1_macro", n_jobs=16)
        print("mean={}, f1_macro={}".format(np.mean(cv), cv))
        print("{}: cross-validate completed, time={:4f}".format(name, time.time() - tStart))

    # 重新在全部训练集中训练、拟合
    model = re_train(model, name, featsData, labelsData)
    # 在测试集上进行预测
    run_predict(model, name, min_df, max_features)


def run_gs_cv(model, name, paras, scoring, X_train, y_train):
    print("{}: grid search cv...".format(name))
    cv = GridSearchCV(estimator=model,
                      param_grid=paras,
                      scoring=scoring,
                      cv=skf, )
    cv.fit(X_train, y_train)
    print("cv best estimator: {}".format(cv.best_estimator_))
    print("cv best paras: {}".format(cv.best_params_))
    print("cv best score({})={}".format(scoring, cv.best_score_))

    # 重新在全部训练集中训练、拟合
    model = re_train(model, name, featsData, labelsData)
    # 在测试集上进行预测
    run_predict(model, name, min_df, max_features)
    return


skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# rfcls = RandomForestClassifier(n_estimators=400, n_jobs=16)
# run_cv(rfcls, "rf")

lgbmcls = lgb.LGBMClassifier(objective='multi-class', random_state=42, n_jobs=16)
paras = {
    'learning_rate': [0.001, 0.01, 0.05, 0.1],
    'boosting_type': ["gbdt", "rf"],
    'n_estimators': [100, 300, 500],
    'max_depth': [10, 15, 20, 25],
    'subsample': [0.2, 0.4, 0.6, 0.8, 1.0],
}
run_gs_cv(lgbmcls, "lgbm-gs", paras, "f1_macro", featsData, labelsData)
# run_cv(lgbmcls, "lgbm")
