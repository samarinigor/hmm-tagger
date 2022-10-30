import os
import string
import seaborn as sns

from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap


def parse_file(input_file: str, is_labeled: bool) -> list[list[tuple[str, str]]] or list[list[str]]:
    if not input_file or not os.path.isfile(input_file):
        raise Exception('Invalid file')
    sentence, sentences = [], []
    with open(input_file, 'r', encoding='utf-8') as f:
        while True:
            line = f.readline()
            if not line:
                break
            line = line.strip()
            if not line and sentence:
                sentences.append(sentence)
                sentence = []
            else:
                if is_labeled:
                    word, tag = line.rsplit(' ', 1)
                    if word in string.punctuation + '–':    # comment if you want text with punctuation marks
                        continue                            # comment if you want text with punctuation marks
                    sentence.append((word, tag))
                else:
                    sentence.append(line)
    return sentences


def get_beginnings(input_file: str) -> list[int]:
    if not input_file or not os.path.isfile(input_file):
        raise Exception('Invalid file')
    else:
        beginnings = []
        with open(input_file, 'r', encoding='utf-8') as f:
            while True:
                line = f.readline()
                if not line:
                    break
                else:
                    beginnings.append(int(line.strip()))
        return beginnings


def write_output(file_name: str, X_test: list[list[str]], y_pred: list[list[str]]) -> None:
    with open(file_name, 'w', encoding='utf-8') as f:
        for X_sent, y_sent in zip(X_test, y_pred):
            for X_word, y_tag in zip(X_sent, y_sent):
                f.write(f'{X_word}|{y_tag} ')
            f.write('\n')
    print('Файл записан!')


def plot_confusion_matrix(y_true: list[str], y_pred: list[str], labels: list[str]) -> None:
    plt.figure(figsize=(10, 10))
    ax = plt.subplot()
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    sns.heatmap(cm, annot=True, cmap=ListedColormap(['white']), cbar=False, linewidths=1, linecolor='#e2e2e2')
    plt.title('confusion matrix')
    plt.xlabel('prediction')
    plt.ylabel('ground truth')
    ax.xaxis.set_ticklabels(labels)
    ax.yaxis.set_ticklabels(labels)
    plt.show()


def metrics(y_true: list[str], y_pred: list[str], labels: list[str]) -> None:
    print(f'f1-score: {f1_score(y_true, y_pred, average=None, labels=labels)}')
    print(f'Accuracy: {accuracy_score(y_true, y_pred)}')
    plot_confusion_matrix(y_true, y_pred, labels)
