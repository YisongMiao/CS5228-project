from sklearn.utils import shuffle
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline


# read train data
df = pd.read_csv('train_v2.csv', header=0)
X = df['title'] + '. From ' + df['publisher']
Y = df['category']

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.20)


parameters = {'vect__ngram_range': [(1, 1), (1, 2), (1, 3)],
               'tfidf__use_idf': (True, False),
               'clf__alpha': (1, 1e-1, 1e-2, 1e-3, 1e-4),
               'clf__fit_prior': (True, False),
              }

text_clf = Pipeline([('vect', CountVectorizer(stop_words='english', min_df=1e-3)),
                      ('tfidf', TfidfTransformer()),
                      ('clf', MultinomialNB()),])

# text_clf = text_clf.fit(X_train.values.astype('U'), y_train.values)

gs_clf = GridSearchCV(text_clf, parameters, n_jobs=-1)
gs_clf = gs_clf.fit(X_train.values.astype('U'), y_train.values)

print(gs_clf.best_score_)
print(gs_clf.best_params_)

# predicted = gs_clf.predict(X_test.values.astype('U'))
# print(np.mean(predicted == y_test))
# {'clf__alpha': 0.1, 'clf__fit_prior': True, 'tfidf__use_idf': False, 'vect__ngram_range': (1, 1)}
# 0.64

