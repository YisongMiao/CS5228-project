# coding: utf-8
import pandas as pd
from multiprocessing import Queue, Pool
from rake_nltk import Rake
import pickle

FILE_PATH = 'all/test_v2.csv'
PICKLE_PATH = 'test_v2.pk'

PROCESS_NUM = 16


def worker(text):
    r = Rake()
    try:
        r.extract_keywords_from_text(text)
    except:
        return ''
    return ','.join(r.get_ranked_phrases())


if __name__ == '__main__':
    with open(PICKLE_PATH, 'rb') as fin:
        ctx = pickle.load(file=fin)
    with Pool(PROCESS_NUM) as p:
        result = p.map(worker, ctx)
    with open(FILE_PATH, 'r', encoding='utf-8') as fin:
        with open('output.csv', 'w', encoding='utf-8') as fout:
            first_line = fin.readline()
            fout.write(first_line)
            for line, words in zip(fin, result):
                newline = line.strip() + ',' + words + '\n'
                fout.write(newline)


