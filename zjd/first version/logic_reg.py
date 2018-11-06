from sklearn.linear_model import LogisticRegression
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
               'clf__penalty': ('l1', 'l2'),
               'clf__tol': (1e-2, 1e-3, 1e-4),
               'clf__C': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1],
               'clf__fit_intercept': (True, False),
              }

text_clf = Pipeline([('vect', CountVectorizer(stop_words='english', min_df=1e-3)),
                      ('tfidf', TfidfTransformer()),
                      ('clf', LogisticRegression()),])

# text_clf = text_clf.fit(X_train.values.astype('U'), y_train.values)

gs_clf = GridSearchCV(text_clf, parameters, n_jobs=-1)
gs_clf = gs_clf.fit(X_train.values.astype('U'), y_train.values)

print(gs_clf.best_score_)
print(gs_clf.best_params_)

# predicted = gs_clf.predict(X_test.values.astype('U'))
# print(np.mean(predicted == y_test))
# {'clf__C': 1, 'clf__fit_intercept': True, 'clf__penalty': 'l2', 'clf__tol': 0.01, 'tfidf__use_idf': True, 'vect__ngram_range': (1, 2)}
# 0.65
