import copy
import os

import numpy as np

from lib.hmm import hmm
from lib.data import data
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score


if __name__ == '__main__':
    """
    Тестирование на Ригведе
    """
    data_set = data.parse_file(f'{os.getcwd()}/data/train/book1_f.train', True)
    labels = ['ANIMAL', 'PER', 'O']

    n = 200
    average_score = np.zeros(3)
    average_accuracy = 0
    for i in range(n):
        train_set, test_set = train_test_split(copy.deepcopy(data_set), train_size=.8, test_size=.2)
        X_test = [[word for word, tag in sentence] for sentence in test_set]
        y_test = [tag for sentence in test_set for word, tag in sentence]

        model = hmm.HiddenMarkovModel('trigram')
        model.fit(train_set)
        y_pred = model.predict(X_test)
        y_pred = [tag for sentence in y_pred for tag in sentence]
        average_score = average_score + f1_score(y_test, y_pred, average=None, labels=labels)
        average_accuracy = average_accuracy + accuracy_score(y_test, y_pred)
    print(f'Average F1-score: {average_score / n}')
    print(f'Average accuracy: {average_accuracy / n}')