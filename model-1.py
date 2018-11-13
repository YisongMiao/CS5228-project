# This model is only by title

import pandas as pd
import regex as re
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDClassifier
from nltk.stem.porter import *

train_df = pd.read_csv('data/train_v2.csv')
test_df = pd.read_csv('data/test_v2.csv')

print('train_df size: {}, test_df size: {}'.format(train_df.shape, test_df.shape))


stemmer = PorterStemmer()

# treat obamacar as stopword!
manual_stopwords = ['obamacar']


def my_tokenizer(s):
    words = re.findall(r'[A-Za-z]+', s)
    words = [word.lower() for word in words]
    words = [stemmer.stem(word) for word in words]
    words = [word for word in words if (len(word) > 1 and word not in manual_stopwords)]
    return words


X = train_df['title']
y = train_df['category']
X_test = test_df['title']

text_clf_svm = Pipeline([('vect', CountVectorizer(ngram_range=(1, 2), tokenizer=my_tokenizer, stop_words='english')),
                         ('tfidf', TfidfTransformer(use_idf=True)),
                         ('clf-svm', SGDClassifier(alpha=0.001, loss='hinge', penalty='l2', warm_start=True, n_iter=6, random_state=42))])

text_clf_svm = text_clf_svm.fit(X.values.astype('U'), y.values)
print('fit done')

result = text_clf_svm.predict(X_test)

write_file = open('result-model-1.txt', 'w+')
write_file.write('article_id,category\n')
i = 1
for item in result:
    write_file.write('{},{}\n'.format(i, item))
    i += 1
write_file.close()
