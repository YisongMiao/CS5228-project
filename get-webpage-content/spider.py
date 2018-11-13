# coding: utf-8
import pandas as pd
from multiprocessing import Queue, Pool
from goose3 import Goose
import pickle

FILE_PATH = 'all/test_v2.csv'
PROCESS_NUM = 40

def worker(url):
    g = Goose()
    try:
        article = g.extract(url=url)
        content = article.cleaned_text
    except:
        print('ERROR')
        return
    return content

if __name__ == '__main__':
    df = pd.read_csv(FILE_PATH)
    urls = []
    for idx, row in df.iterrows():
        urls.append(row['url'])
    print(len(urls), 'urls wait for crawling...')

    with Pool(PROCESS_NUM) as p:
        result = p.map(worker, urls)

    with open('ctx.pk', 'wb') as f:
        pickle.dump(result, f)


