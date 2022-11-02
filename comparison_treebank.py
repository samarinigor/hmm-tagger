import pandas as pd
from lib.hmm import hmm
from copy import deepcopy
from random import choice
from sklearn.metrics import f1_score, accuracy_score
from nltk.corpus import treebank, brown


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
                pred_y.append(choice(labels))

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

    train_data = list(treebank.tagged_sents(tagset='universal'))[:500]
    test_data = list(brown.tagged_sents(tagset='universal'))[:1000]

    test_sentences = [[word for word, tag in sentence] for sentence in test_data]
    test_y = [tag for sentence in test_data for word, tag in sentence]
    labels = list(set(test_y))

    table = pd.DataFrame(columns=labels, index=['f1_traditional', 'f1_hmm'])

    f1_trad, acc_trad = dict_method(deepcopy(train_data), deepcopy(test_sentences), deepcopy(test_y), labels)
    f1_hmm, acc_hmm = hmm_method(deepcopy(train_data), deepcopy(test_sentences), deepcopy(test_y), labels)

    table.loc['f1_traditional'] = f1_trad
    table.loc['f1_hmm'] = f1_hmm
    print(f'Accuracy_traditional: {acc_trad}; Accuracy_hmm: {acc_hmm}')

    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        print(table)
