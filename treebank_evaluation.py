from lib.hmm import hmm
from lib.data import data
from sklearn.model_selection import train_test_split
from nltk.corpus import treebank


if __name__ == '__main__':
    """
    Тест модели на treebank sentences
    """
    nltk_data = list(treebank.tagged_sents(tagset='universal'))
    train_set, test_set = train_test_split(nltk_data, train_size=0.8, test_size=0.2)

    model = hmm.HiddenMarkovModel('trigram')
    model.fit(train_set)

    test_sentences = [[word for word, tag in sent] for sent in test_set]
    y_test = [tag for sent in test_set for word, tag in sent]

    y_pred = model.predict(test_sentences)
    y_pred = [tag for sentence in y_pred for tag in sentence]

    labels = list(set(y_test))
    data.metrics(y_test, y_pred, labels)