import os
import re
import string

from nltk.tokenize import sent_tokenize, word_tokenize


def punctuation_replace(s: str) -> str:
    s = re.sub(r'\d\d?', '', s).replace('...', '.')
    return s


def text_fragmentation(input_file: str) -> list[str]:
    current_hymn = ''
    hymns = []
    roman_nums = ['I', 'V', 'X']
    with open(input_file, 'r', encoding='utf-8') as f:
        while True:
            line = f.readline()
            if not line:
                break
            line = line.strip()
            if line:
                if line[0] in roman_nums:
                    if current_hymn:
                        hymns.append(punctuation_replace(current_hymn.rstrip()))
                        current_hymn = ''
                    else:
                        continue
                else:
                    current_hymn = current_hymn + line + ' '
    hymns.append(punctuation_replace(current_hymn.rstrip()))
    return hymns


def write_file(output_file: str, hymns: list[str], lowercase: bool, remove_brackets: bool) -> None:
    word_count = 0
    hymns_beginnings = [0]
    with open(output_file, 'w', encoding='utf-8') as f:
        for hymn in hymns:
            if lowercase:
                hymn = hymn.lower()
            sentences = sent_tokenize(hymn)
            for sentence in sentences:
                words = word_tokenize(sentence)
                bracket_open = False
                idx = None
                for i, word in enumerate(words):
                    # if word in string.punctuation + '–':      # uncomment if you want text without punctuation marks
                    #     continue
                    if word == '(':
                        bracket_open = True
                        idx = i + 1
                    elif word == ')':
                        bracket_open = False
                        if not remove_brackets:
                            f.write(f'({" ".join(words[idx:i])})\n')
                            word_count = word_count + 1
                    else:
                        if not bracket_open:
                            f.write(f'{word}\n')
                            word_count = word_count + 1
                f.write('\n')
            hymns_beginnings.append(word_count)
    path, extension = output_file.rsplit('.', 1)
    output_file = f'{path}_beginnings.{extension}'
    with open(output_file, 'w', encoding='utf-8') as f:
        for hymn_beginning in hymns_beginnings:
            f.write(f'{hymn_beginning}\n')


def book_structuring(input_file: str, output_file: str, lowercase: bool = False, remove_brackets: bool = False) -> None:
    if not input_file or not os.path.isfile(input_file):
        raise Exception('Invalid file')
    else:
        hymns = text_fragmentation(input_file)
        write_file(output_file, hymns, lowercase, remove_brackets)
