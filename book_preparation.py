import os

from lib.data import book_helper


if __name__ == '__main__':
    """
    Преобразует текстовый файл (книгу) в готовый для маркировки документ, где в отдельной строке записано отдельное слово.
    Также возвращает файл _beginnings с началами каждого из гимнов.
    """
    input_path = f'{os.getcwd()}/books/books2-10.txt'
    output_path = f'{os.getcwd()}/data/test/raw/books2-10_raw_p.test'
    book_helper.book_structuring(input_path, output_path, lowercase=True, remove_brackets=True)
