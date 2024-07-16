from taker import Model
import json

import numpy as np
import torch

import random

class LabeledTokenizer:
    def __init__(self, model):
        self.model: Model = model
        self.map = dict()
        self.cpt = dict()
        self.reverse_map = dict()
        self.num_labels = 0

    def get_label_id(self, label, n_tokens):
        # add label to map if not already present
        if label not in self.map:
            self.map[label] = self.num_labels
            self.reverse_map[self.num_labels] = label
            self.num_labels += 1
            self.cpt[label] = n_tokens
        else:
            self.cpt[label] += n_tokens
        return self.map[label]

    def tokenize(self, data, label_y):

        input_list = []
        label_list = []

        for prompt in range(len(data)):

            if data[prompt]['split_text'] is not None:

                for index in range(len(data[prompt]['split_text'])):

                    text = data[prompt]['split_text'][index]['text']
                    label = data[prompt]['split_text'][index][label_y]

                    residual = m.get_residual_stream(text).cpu()
                    n_tokens = residual.shape[1]

                    if n_tokens<token_max:

                        label_id = self.get_label_id(label, n_tokens)
                        label_idx = torch.full((n_tokens,), label_id)
                        input_list.append(residual)

                        label_list.append(label_idx)

        input_torch = torch.cat(input_list, dim=1)
        label_torch = torch.cat(label_list, dim=0)
        print(self.cpt)
        return input_torch, label_torch



m = Model("nickypro/tinyllama-15M")
#m = Model("roberta-large")
#m = Model("mistralai/Mistral-7B-Instruct-v0.2", dtype="int8")


labeled_tokenizer = LabeledTokenizer(m)

data_files=['mistral1.json', 'mistral2.json', 'mistral4.json']
data = list()
for fil in data_files:
    with open(fil, 'r') as infile:
        data.extend(json.load(infile))

random.shuffle(data)
idx = int(len(data)*0.8)
train_data = data[:idx]
test_data = data[idx:]

label_y = 'genre'

token_max = 512

x_train, y_train = labeled_tokenizer.tokenize(train_data, label_y)
x_test, y_test = labeled_tokenizer.tokenize(test_data, label_y)

device = 'cpu'

n_classes = torch.unique(y_train).numel()


save_x_train = x_train
save_x_test = x_test

save_y_train = y_train
save_y_test = y_test


from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import RidgeClassifier
from sklearn.linear_model import SGDClassifier


for layer in range(m.cfg.n_layers*2 + 1):
    #layer = 11

    clf = LogisticRegression(max_iter=100000)
    #clf = LinearDiscriminantAnalysis()
    #clf = RidgeClassifier(max_iter=100000)
    #clf = SGDClassifier(max_iter=1000000)

    x_train = np.array(save_x_train[layer].cpu())
    x_test = np.array(save_x_test[layer].cpu())

    y_train = np.array(save_y_train.cpu())
    y_test = np.array(save_y_test.cpu())

    pipe = make_pipeline(StandardScaler(), clf)

    pipe.fit(x_train, y_train)
    pred = pipe.predict(x_test)
    print('Model', clf, 'Layer', layer, 'Score:', f1_score(y_test, pred, average='macro'))


for layer in range(m.cfg.n_layers*2 + 1):
    #layer = 11

    #clf = LogisticRegression(max_iter=100000)
    clf = LinearDiscriminantAnalysis()
    #clf = RidgeClassifier(max_iter=100000)
    #clf = SGDClassifier(max_iter=1000000)

    x_train = np.array(save_x_train[layer].cpu())
    x_test = np.array(save_x_test[layer].cpu())

    y_train = np.array(save_y_train.cpu())
    y_test = np.array(save_y_test.cpu())

    pipe = make_pipeline(StandardScaler(), clf)

    pipe.fit(x_train, y_train)
    pred = pipe.predict(x_test)
    print('Model', clf, 'Layer', layer, 'Score:', f1_score(y_test, pred, average='macro'))


for layer in range(m.cfg.n_layers*2 + 1):
    #layer = 11

    #clf = LogisticRegression(max_iter=100000)
    #clf = LinearDiscriminantAnalysis()
    clf = RidgeClassifier(max_iter=100000)
    #clf = SGDClassifier(max_iter=1000000)

    x_train = np.array(save_x_train[layer].cpu())
    x_test = np.array(save_x_test[layer].cpu())

    y_train = np.array(save_y_train.cpu())
    y_test = np.array(save_y_test.cpu())

    pipe = make_pipeline(StandardScaler(), clf)

    pipe.fit(x_train, y_train)
    pred = pipe.predict(x_test)
    print('Model', clf, 'Layer', layer, 'Score:', f1_score(y_test, pred, average='macro'))


for layer in range(m.cfg.n_layers*2 + 1):
    #layer = 11

    #clf = LogisticRegression(max_iter=100000)
    #clf = LinearDiscriminantAnalysis()
    #clf = RidgeClassifier(max_iter=100000)
    clf = SGDClassifier(max_iter=1000000)

    x_train = np.array(save_x_train[layer].cpu())
    x_test = np.array(save_x_test[layer].cpu())

    y_train = np.array(save_y_train.cpu())
    y_test = np.array(save_y_test.cpu())

    pipe = make_pipeline(StandardScaler(), clf)

    pipe.fit(x_train, y_train)
    pred = pipe.predict(x_test)
    print('Model', clf, 'Layer', layer, 'Score:', f1_score(y_test, pred, average='macro'))
