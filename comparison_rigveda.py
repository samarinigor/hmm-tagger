import os
import numpy as np
import pandas as pd
from lib.hmm import hmm
from lib.data import data
from copy import deepcopy
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score


def dict_method(train_set: list[list[list[str, str]]], test_set: list[list[str]], test_y: list[str], labels: list[str]):
    model = hmm.HiddenMarkovModel('trigram')
    model.fit(train_set)

    words_population = model._HiddenMarkovModel__word_population

    pred_y = []
    for sentence in test_set:
        for word in sentence:
            if word in words_population.keys():
                tags = words_population[word]
                pred_y.append(max(tags, key=tags.get))
            else:
                pred_y.append('O')

    score = f1_score(test_y, pred_y, average=None, labels=labels)
    accuracy = accuracy_score(test_y, pred_y)

    return score, accuracy


def hmm_method(train_set: list[list[list[str, str]]], test_set: list[list[str]], test_y: list[str], labels: list[str]):
    model = hmm.HiddenMarkovModel('trigram')
    model.fit(train_set)

    pred_y = model.predict(test_set)
    pred_y = [tag for sentence in pred_y for tag in sentence]

    score = f1_score(test_y, pred_y, average=None, labels=labels)
    accuracy = accuracy_score(test_y, pred_y)

    return score, accuracy


if __name__ == '__main__':

    labels = ['ANIMAL', 'PER', 'O']
    data = data.parse_file(f'{os.getcwd()}/data/train/book1_f.train', True)

    n = 100
    table = pd.DataFrame(np.zeros((2, 3)), columns=labels, index=['f1_traditional', 'f1_hmm'])
    average_acc_trad = 0
    average_acc_hmm = 0
    for i in range(n):
        train_data, test_data = train_test_split(deepcopy(data), train_size=.8, test_size=.2)
        test_sentences = [[word for word, tag in sentence] for sentence in test_data]
        test_y = [tag for sentence in test_data for word, tag in sentence]

        f1_trad, acc_trad = dict_method(deepcopy(train_data), deepcopy(test_sentences), deepcopy(test_y), labels)
        table.loc['f1_traditional'] = table.loc['f1_traditional'] + f1_trad
        average_acc_trad = average_acc_trad + acc_trad

        f1_hmm, acc_hmm = hmm_method(deepcopy(train_data), deepcopy(test_sentences), deepcopy(test_y), labels)
        table.loc['f1_hmm'] = table.loc['f1_hmm'] + f1_hmm
        average_acc_hmm = average_acc_hmm + acc_hmm

    print(f'Accuracy_traditional: {average_acc_trad / n}; Accuracy_hmm: {average_acc_hmm / n}')
    print(table / n)
