import numpy as np


class HiddenMarkovModel:

    __supported_models = {'trigram': 3}

    def __init__(self, ngram: str = 'trigram') -> None:
        if ngram in HiddenMarkovModel.__supported_models.keys():
            self.__ngram = HiddenMarkovModel.__supported_models[ngram]
        else:
            raise ValueError('Unsupported language model')
        self.__start_token = None
        self.__stop_token = None
        self.__train_set = None
        self.__n = 0
        """ useful dictionaries """
        self.__tags = set()
        self.__ngram_counts = dict()
        self.__word_population = dict()
        self.__prior_transition = dict()
        self.__prior_emission = dict()
        self.__singleton_transition = dict()
        self.__singleton_emission = dict()
        self.__one_count_transition = dict()
        self.__one_count_emission = dict()
        """ viterbi dictionaries """
        self.__pi = dict()
        self.__bp = dict()

    def fit(self, train_set: list[list[list[str, str]]], start_token: str = '_start_', stop_token: str = '_stop_') -> None:
        """
        Trains the model using the input data
        :param train_set: set of labeled sentences
        :param start_token: the padding token
        :param stop_token: the padding token
        :return: None
        """
        self.__train_set = train_set
        self.__start_token = start_token
        self.__stop_token = stop_token

        for sentence in self.__train_set:
            for _ in range(self.__ngram-1):
                sentence.insert(0, [self.__start_token, self.__start_token])
            sentence.append([self.__stop_token, self.__stop_token])
            tag_sentence = [tag for word, tag in sentence]
            for i in range(0, len(sentence)):
                if i >= (self.__ngram-1):
                    n_gram = tuple(tag_sentence[i+1-self.__ngram:i+1])
                    for _ in range(self.__ngram, 1, -1):
                        self.__ngram_counts[n_gram[:_]] = self.__ngram_counts.get(n_gram[:_], 0) + 1
                word, tag = sentence[i]
                self.__tags.add(tag)
                if word not in self.__word_population:
                    self.__word_population[word] = dict()
                self.__word_population[word][tag] = self.__word_population[word].get(tag, 0) + 1
                self.__ngram_counts[(tag,)] = self.__ngram_counts.get((tag,), 0) + 1
                self.__n = self.__n + 1
        self.__calc_prior_distribution()
        self.__one_count_probabilities()

    def __calc_prior_distribution(self) -> None:
        for tag in self.__tags:
            self.__prior_transition[tag] = (1 + self.__ngram_counts[(tag,)]) / (len(self.__tags) + self.__n)
        for word, population in self.__word_population.items():
            self.__prior_emission[word] = sum(population.values()) / self.__n

    def __calc_one_count_singletons(self) -> None:
        """
        Fill dictionaries with singletons i.e. |w_{i}: c(w_{i-n+1}^i) = 1|
        :return: None
        """
        for ngram, ngram_count in self.__ngram_counts.items():
            l = len(ngram)
            if ngram_count == 1 and l > 1:
                self.__singleton_transition[ngram[:l-1]] = self.__singleton_transition.get(ngram[:l-1], 0) + 1
        for word, population in self.__word_population.items():
            for tag, word_tag_count in population.items():
                if word_tag_count == 1:
                    self.__singleton_emission[tag] = self.__singleton_emission.get(tag, 0) + 1

    def __get_one_count_transition(self, tag_tuple: tuple) -> float:
        """
        One-count transition probability
        :param tag_tuple: (t_{i-n+1}^{i})
        :return: P(t_{i} | t_{i-n+1}^{i-1})
        """
        l = len(tag_tuple)
        if l == 1:
            return self.__prior_transition[tag_tuple[0]]
        else:
            alpha = 1 + self.__singleton_transition.get(tag_tuple[:l-1], 0)
            transition = ((self.__ngram_counts.get(tag_tuple, 0) + alpha * self.__get_one_count_transition(tag_tuple[-(l-1):])) /
                          (self.__ngram_counts.get(tag_tuple[:l-1], 0) + alpha))
        return transition

    def __get_one_count_emission(self, word: str, tag: str) -> float:
        """
        Smoothing emission probability
        :param word: w_{i}
        :param tag: t_{i}
        :return: P(w_{i} | t_{i})
        """
        beta = 1 + self.__singleton_emission.get(tag, 0)
        emission = ((self.__word_population.get(word, {}).get(tag, 0) + beta * self.__prior_emission.get(word, 1 / (self.__n + len(self.__tags)))) /
                    (self.__ngram_counts[(tag,)] + beta))
        return emission

    def __one_count_probabilities(self) -> None:
        """
       Fill dictionaries with one-count smoothing transition and emission probabilities
       :return: None
       """
        self.__calc_one_count_singletons()
        for ngram in self.__ngram_counts.keys():
            if len(ngram) == self.__ngram:
                self.__one_count_transition[ngram] = np.log(self.__get_one_count_transition(ngram))
        for word, population in self.__word_population.items():
            for tag in population.keys():
                self.__one_count_emission[(word, tag)] = np.log(self.__get_one_count_emission(word, tag))

    def __get_tags(self, word: str) -> list[str]:
        """
        Tags getter
        :param word: current word
        :return: list of tags
        """
        if word in self.__word_population:
            return self.__word_population[word].keys()
        else:
            return list(self.__tags)

    def __get_smoothed_transition(self, tag_tuple: tuple) -> float:
        """
        Smoothing transition probability getter
        :param tag_tuple: (t_{i-n+1}^{i})
        :return: P(t_{i} | t_{i-n+1}^{i-1})
        """
        if tag_tuple in self.__one_count_transition:
            return self.__one_count_transition[tag_tuple]
        else:
            return np.log(self.__get_one_count_transition(tag_tuple))

    def __get_smoothed_emission(self, word: str, tag: str) -> float:
        """
        Smoothing emission probability getter
        :param word: w_{i}
        :param tag: t_{i}
        :return: P(w_{i} | t_{i})
        """
        if (word, tag) in self.__one_count_emission:
            return self.__one_count_emission[(word, tag)]
        else:
            return np.log(self.__get_one_count_emission(word, tag))

    def predict(self, test_set: list[list[str]]) -> list[list[str]]:
        """
        Predicts tags sequences for a test data
        :param test_set: not tagged sentences
        :return: sequence of sequences of most likely tags
        """
        y_pred = []
        for sentence in test_set:
            y_pred.append(self.__trigram_decode(sentence))
        return y_pred

    def __trigram_decode(self, sentence: list[str]) -> list[str]:
        """
        Find the highest probable tag sequence (trigram language model)
        :param sentence: not tagged sequence of words
        :return: sequence of most likely tags
        """
        sentence = [self.__start_token, *sentence, self.__stop_token]
        for step, word in enumerate(sentence[1:], start=1):
            for tag_i in self.__get_tags(word):
                if step == 1:
                    transition_probability = self.__get_smoothed_transition((self.__start_token, self.__start_token, tag_i))
                    emission_probability = self.__get_smoothed_emission(word, tag_i)
                    self.__pi[(step, self.__start_token, tag_i)] = transition_probability + emission_probability
                    self.__bp[(step, self.__start_token, tag_i)] = (self.__start_token, self.__start_token)
                else:
                    max_score = float('-Inf')
                    bp = None
                    for tag_j in self.__get_tags(sentence[step-1]):
                        for tag_k in self.__get_tags(sentence[step-2]):
                            transition_probability = self.__get_smoothed_transition((tag_k, tag_j, tag_i))
                            emission_probability = self.__get_smoothed_emission(word, tag_i)
                            score = self.__pi[(step-1, tag_k, tag_j)] + transition_probability + emission_probability
                            if score > max_score:
                                max_score = score
                                bp = (tag_k, tag_j)
                        self.__pi[(step, tag_j, tag_i)] = max_score
                        self.__bp[(step, tag_j, tag_i)] = bp
        tag_sequence = self.__get_tag_sequence(sentence)
        return tag_sequence

    def __get_tag_sequence(self, sentence: list[str]):
        values = []
        for step in reversed(range(len(sentence))):
            if step == len(sentence)-1:
                max_score = float('-Inf')
                value = None
                for tag_i in self.__get_tags(sentence[step]):
                    for tag_j in self.__get_tags(sentence[step-1]):
                        if self.__pi[(step, tag_j, tag_i)] > max_score:
                            max_score = self.__pi[(step, tag_j, tag_i)]
                            value = (tag_j, tag_i)
                values.insert(0, value)
            else:
                values.insert(0, self.__bp[(step+1, *values[0])])
        tag_sequence = [value[-1] for value in values[1:-1]]
        return tag_sequence
