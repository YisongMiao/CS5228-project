# This model is by title + webpage content(though only 55% is valid)

import pandas as pd
import regex as re
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDClassifier
import pickle
from sklearn.feature_extraction import stop_words
from nltk.stem.porter import *


train_df = pd.read_csv('data/train_v2.csv')
test_df = pd.read_csv('data/test_v2.csv')

print('train_df size: {}, test_df size: {}'.format(train_df.shape, test_df.shape))

print(len(stop_words.ENGLISH_STOP_WORDS))
my_stop_word = ['forty', 'much', 'whereafter', 'towards', 'our', 'during', 'done', 'one', 'please', 'whom', 'whereas', 'hers', 'thick', 'him', 'never', 'upon', 'are', 'whose', 'ten', 'except', 'than', 'those', 'whereupon', 'already', 'across', 're', 'somewhere', 'became', 'fire', 'six', 'without', 'although', 'third', 'eleven', 'amoungst', 'becomes', 'below', 'all', 'any', 'nothing', 'almost', 'how', 'in', 'he', 'over', 'seem', 'about', 'see', 'around', 'often', 'few', 'therefore', 'what', 'yourselves', 'both', 'keep', 'ours', 'always', 'once', 'others', 'seeming', 'none', 'perhaps', 'its', 'whether', 'whoever', 'be', 'etc', 'call', 'indeed', 'latter', 'further', 'last', 'three', 'eg', 'interest', 'other', 'but', 'your', 'here', 'per', 'wherein', 'against', 'sometime', 'within', 'name', 'former', 'may', 'when', 'will', 'whither', 'cry', 'beforehand', 'themselves', 'to', 'whereby', 'from', 'can', 'put', 'ie', 'bottom', 'which', 'have', 'not', 'seemed', 'until', 'somehow', 'was', 'cant', 'nobody', 'several', 'side', 'thru', 'now', 'four', 'there', 'neither', 'some', 'enough', 'otherwise', 'yet', 'into', 'still', 'anyone', 'made', 'via', 'everyone', 'give', 'take', 'thereby', 'amongst', 'thereafter', 'becoming', 'elsewhere', 'noone', 'no', 'onto', 'am', 'because', 'find', 'meanwhile', 'my', 'it', 'since', 'this', 'toward', 'anywhere', 'very', 'ever', 'another', 'formerly', 'her', 'hereby', 'me', 'an', 'himself', 'such', 'hereafter', 'might', 'afterwards', 'mine', 'nine', 'the', 'five', 'therein', 'where', 'why', 'anything', 'someone', 'that', 'after', 'fifteen', 'everywhere', 'herself', 'sometimes', 'nevertheless', 'mostly', 'show', 'a', 'part', 'before', 'together', 'either', 'of', 'whatever', 'de', 'eight', 'own', 'beyond', 'detail', 'even', 'were', 'cannot', 'fill', 'moreover', 'under', 'if', 'con', 'them', 'empty', 'get', 'system', 'nor', 'his', 'front', 'describe', 'must', 'then', 'between', 'been', 'bill', 'you', 'had', 'well', 'ourselves', 'alone', 'twelve', 'too', 'each', 'else', 'hundred', 'among', 'also', 'only', 'whole', 'less', 'hasnt', 'being', 'off', 'first', 'these', 'could', 'ltd', 'back', 'nowhere', 'with', 'everything', 'co', 'seems', 'sincere', 'myself', 'beside', 'go', 'every', 'rather', 'two', 'on', 'move', 'i', 'hence', 'who', 'due', 'amount', 'full', 'again', 'hereupon', 'fifty', 'whence', 'un', 'besides', 'latterly', 'mill', 'same', 'through', 'would', 'serious', 'sixty', 'top', 'most', 'we', 'anyway', 'couldnt', 'while', 'yourself', 'thus', 'become', 'their', 'thereupon', 'more', 'namely', 'above', 'wherever', 'found', 'itself', 'thin', 'thence', 'has', 'is', 'yours', 'by', 'for', 'though', 'along', 'out', 'she', 'next', 'twenty', 'however', 'something', 'herein', 'inc', 'at', 'do', 'throughout', 'as', 'anyhow', 'and', 'they', 'whenever', 'least', 'behind', 'or', 'should', 'so', 'many']
print(len(my_stop_word))  # here we remove 'up', 'down' as stop word, because it is important for some class.

stemmer = PorterStemmer()


def get_new_stopwords():
    manual_stopwords = ['obamacar', 'say', 'said', 'monday', 'tuesday',
                        'wednesday', 'thursday', 'friday', 'saturday', 'sunday',
                        'january', 'february', 'march', 'april', 'may', 'june',
                        'july', 'april', 'may', 'june', 'july', 'august', 'september',
                        'octerber', 'november', 'december']
    words = [stemmer.stem(word) for word in manual_stopwords]
    return words


manual_stopwords = get_new_stopwords()  # here we even remove more stop to denoise!


def my_tokenizer(s):
    words = re.findall(r'[A-Za-z]+', s)
    words = [word.lower() for word in words]
    words = [stemmer.stem(word) for word in words]
    words = [word for word in words if (len(word) > 1 and word not in manual_stopwords)]
    return words


train_raw = pickle.load(open('data/train_v2.pk', 'rb'))
test_raw = pickle.load(open('data/test_v2.pk', 'rb'))

custom_tokens = dict()

for i in range(train_df.shape[0]):
    #if i % 50 != 0:
    #    continue
    title_ = my_tokenizer(train_df.iloc[i]['title'])
    if train_raw[i] is None or len(train_raw[i]) < 500:  # means invalid webpage content, like 404 error
        custom_tokens[''.join(('train', str(i)))] = title_
        continue
    extend = my_tokenizer(train_raw[i])[: 1000]
    custom_tokens[''.join(('train-', str(i)))] = title_ + extend

for i in range(test_df.shape[0]):
    #if i % 50 != 0:
    #    continue
    title_ = my_tokenizer(test_df.iloc[i]['title'])
    if test_raw[i] is None or len(test_raw[i]) < 500:
        custom_tokens[''.join(('test', str(i)))] = title_
        continue
    extend = my_tokenizer(test_raw[i])[: 1000]
    custom_tokens[''.join(('test', str(i)))] = title_ + extend

print('Our custom tokenizer for countervectorizer is finished')

train_key = [k for k in custom_tokens.keys() if k[: 2] == 'tr']
test_key = [k for k in custom_tokens.keys() if k[: 2] == 'te']
print(len(train_key), len(test_key))

text_clf_svm = Pipeline([('vect', CountVectorizer(ngram_range=(1, 2), tokenizer=lambda key: custom_tokens[key], stop_words=my_stop_word)),
                         ('tfidf', TfidfTransformer(use_idf=True)),
                         ('clf-svm', SGDClassifier(alpha=0.001, loss='hinge', penalty='l2', warm_start=True, n_iter=6, random_state=42))])

y = train_df['category']
text_clf_svm = text_clf_svm.fit(train_key, y.values)
print('fit done')

result = text_clf_svm.predict(test_key)

write_file = open('result-model-2.txt', 'w+')
write_file.write('article_id,category\n')
i = 1
for item in result:
    write_file.write('{},{}\n'.format(i, item))
    i += 1
write_file.close()
