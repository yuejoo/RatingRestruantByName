# -*- coding: utf-8 -*-


import pandas as pd
import numpy as np
import jieba


index = 0
words_dict = {}

pima = pd.read_csv("sample.csv")
# Remove the
pima["店铺名称"] = pima["店铺名称"].str.replace(r"\(.*\)","")
names = pima[["店铺名称", "评分"]]

print(names)
TRAIN_SPLIT = 0.8

def feature(name):
    seg_list = jieba.cut(name, cut_all=True)
    words_list = []
    for tk in seg_list:
        words_list.append(tk)
    return {
        'first-word': words_list[0], # First letter
        'first2-words': ''.join(words_list[0:2]), # First 2 letters
        'first3-words': ''.join(words_list[0:3]), # First 3 letters
        'last-word': ''.join(words_list[-1]),
        'last2-words': ''.join(words_list[-2:]),
        'last3-words': ''.join(words_list[-3:]),
    }

def valueFeatures(score):
    if score > 4.4:
        return "G"
    return "NG"

names = names.values
features = np.vectorize(feature)
valueFeatures = np.vectorize(valueFeatures)

X = features(names[:, 0])
y = valueFeatures(names[:, 1])

from sklearn.utils import shuffle
X, y = shuffle(X, y)
X_train, X_test = X[:int(TRAIN_SPLIT * len(X))], X[int(TRAIN_SPLIT * len(X)):]
y_train, y_test = y[:int(TRAIN_SPLIT * len(y))], y[int(TRAIN_SPLIT * len(y)):]

print(len(X_train), len(X_test), len(y_train), len(y_test))

from sklearn.feature_extraction import DictVectorizer
vec = DictVectorizer()

vec.fit(X_train)

from sklearn.tree import DecisionTreeClassifier

clf = DecisionTreeClassifier()
clf.fit(vec.transform(X_train), y_train)

print(clf.score(vec.transform(X_train), y_train))
print(clf.score(vec.transform(X_test), y_test))
print(clf.predict(vec.transform(feature("恶心的狗屎自助烤肉"))))