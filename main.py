import os
import numpy as np

from scipy import stats
from lib.hmm import hmm
from lib.data import data
from matplotlib import pyplot as plt


if __name__ == '__main__':
    train_set = data.parse_file(f'{os.getcwd()}/data/train/book1_f.train', True)
    model = hmm.HiddenMarkovModel('trigram')
    model.fit(train_set)

    test_set_raw = data.parse_file(f'{os.getcwd()}/data/test/raw/books2-10_raw.test', False)
    beginnings = data.get_beginnings(f'{os.getcwd()}/data/test/raw/books2-10_raw_beginnings.test')

    y_pred = model.predict(test_set_raw)
    data.write_output('output.txt', test_set_raw, y_pred)

    y_pred = [tag for sentence in y_pred for tag in sentence]

    n = 11
    person_counts, animal_counts = data.labels_counts(y_pred, beginnings, n)

    """ проверка гипотезы однородности """
    n1 = sum(person_counts)
    n2 = sum(animal_counts)

    chi_square = 0
    for i in range(n):
        chi_square = chi_square + (person_counts[i] / n1 - animal_counts[i] / n2) ** 2 / (
                    person_counts[i] + animal_counts[i])

    chi_square = n1 * n2 * chi_square
    print(f'{stats.chi2.ppf(1-0.05, n-1)} = X^2_кр < X^2_набл = {chi_square} => распределения различные')

    """ график распределения частот person/animal """
    person_frequency = person_counts / sum(person_counts)
    animal_frequency = animal_counts / sum(animal_counts)

    width = 1 / (n + 1)
    ind = np.linspace(0, 1, n + 1)[:-1]

    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot(111)
    ax.bar(ind + 1 / (2 * n), np.array(person_frequency), width, color='#ff7f0e', label='PER', alpha=.5)
    ax.bar(ind + 1 / (2 * n), np.array(animal_frequency), width, color='#1f77b4', label='ANIMAL', alpha=.5)
    plt.plot(ind + width / 2, person_frequency, color='#ff7f0e', linewidth=3, marker='o')
    plt.plot(ind + width / 2, animal_frequency, color='#1f77b4', linewidth=3, marker='o')
    plt.xlabel('x*(100%)')
    plt.ylabel('frequency')
    plt.legend()
    plt.show()
