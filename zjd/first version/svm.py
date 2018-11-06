from sklearn.utils import shuffle
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline


# read train data
df = pd.read_csv('train_v2.csv', header=0)
X = df['title'] + '. From ' + df['publisher']
Y = df['category']

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.20)

# 'clf-svm__'
parameters_svm = {'vect__ngram_range': [(1, 1), (1, 2)],
                  'tfidf__use_idf': (True, False),
                  'clf-svm__alpha': (1e-2, 1e-3),
                  'clf-svm__loss': ['hinge', 'log', 'modified_huber', 'squared_hinge', 'perceptron', 'squared_loss',
                                    'huber', 'epsilon_insensitive', 'squared_epsilon_insensitive'],
                  'clf-svm__penalty': ['l1', 'l2', 'elasticnet'],
                  # 'clf-svm__learning_rate': ['constant', 'optimal', 'invscaling', 'adaptive'],

                  'clf-svm__warm_start': (True, False),
                  # 'clf-svm__eta0': (1e-2, 1e-3, 1e-4, 1e-5),
                  }

text_clf_svm = Pipeline([('vect', CountVectorizer()),
                      ('tfidf', TfidfTransformer()),
                      ('clf-svm', SGDClassifier(n_iter=5, random_state=42)),])

# text_clf = text_clf.fit(X_train.values.astype('U'), y_train.values)

gs_clf = GridSearchCV(text_clf_svm, parameters_svm, n_jobs=-1)
gs_clf = gs_clf.fit(X_train.values.astype('U'), y_train.values)

print(gs_clf.best_score_)
print(gs_clf.best_params_)

# predicted = gs_clf.predict(X_test.values.astype('U'))
# print(np.mean(predicted == y_test))
# {'clf-svm__alpha': 0.001, 'clf-svm__loss': 'hinge', 'clf-svm__penalty': 'l2', 'clf-svm__warm_start': True, 'tfidf__use_idf': True, 'vect__ngram_range': (1, 1)}
# 0.6637627048330222

