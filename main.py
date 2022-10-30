import os
import math
import numpy as np

from lib.hmm import hmm
from lib.data import data
from matplotlib import pyplot as plt


def labels_distribution(tags: list[str], hymns_beginnings: list[int], n: int):
    person_freq = np.zeros(n)
    animal_freq = np.zeros(n)
    current_hymn = 0
    for i in range(len(tags)):
        if i == hymns_beginnings[current_hymn+1]:
            current_hymn = current_hymn + 1
        l = hymns_beginnings[current_hymn+1] - hymns_beginnings[current_hymn]
        if tags[i] == 'PER':
            person_freq[math.floor((i - hymns_beginnings[current_hymn]) / l * n)] += 1
        elif tags[i] == 'ANIMAL':
            animal_freq[math.floor((i - hymns_beginnings[current_hymn]) / l * n)] += 1
    return person_freq, animal_freq


if __name__ == '__main__':
    train_set = data.parse_file(f'{os.getcwd()}/data/train/book1_f.train', True)
    model = hmm.HiddenMarkovModel('trigram')
    model.fit(train_set)

    test_set_raw = data.parse_file(f'{os.getcwd()}/data/test/raw/books2-10_raw.test', False)
    beginnings = data.get_beginnings(f'{os.getcwd()}/data/test/raw/books2-10_raw_beginnings.test')

    y_pred = model.predict(test_set_raw)
    data.write_output('output.txt', test_set_raw, y_pred)

    y_pred = [tag for sentence in y_pred for tag in sentence]

    n_intervals = 11
    p_freq, a_freq = labels_distribution(y_pred, beginnings, n_intervals)

    width = 1 / (n_intervals+1)
    ind = np.linspace(0, 1, n_intervals + 1)[:-1]
    fig = plt.figure(figsize=(15, 5))
    ax = fig.add_subplot(111)
    ax.bar(ind + 1 / (2 * n_intervals), np.array(p_freq), width, color='#ff7f0e', label='PER')
    ax.bar(ind + 1 / (2 * n_intervals), np.array(a_freq), width, color='#1f77b4', label='ANIMAL')
    plt.plot(ind + width / 2, p_freq, color='#1f77b4', linewidth=3, marker='o')
    plt.plot(ind + width / 2, a_freq, color='white', linewidth=3, marker='o')
    plt.xlabel('x*(100%)')
    plt.ylabel('frequency')
    plt.legend()
    plt.show()
