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

df['title'] = df['title'].str.lower()
df['publisher'] = df['publisher'].str.lower()

X = df['title'] + '. from ' + df['publisher']
Y = df['category']

# X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.20)

# 'clf-svm__'


text_clf_svm = Pipeline([('vect', CountVectorizer(ngram_range=(1, 1))),
                      ('tfidf', TfidfTransformer(use_idf=True)),
                      ('clf-svm', SGDClassifier(alpha=0.001, loss='hinge', penalty='l2', warm_start=True, n_iter=6, random_state=42)),])

text_clf_svm = text_clf_svm.fit(X.values.astype('U'), Y.values)

df2 = pd.read_csv('test_v2.csv', header=0)
print(type(df2))

X_test = df2['title'] + '. from ' + df2['publisher']

predicted = text_clf_svm.predict(X_test.values.astype('U'))

write_file = open('result-11-06-v1.txt', 'w+')
write_file.write('article_id,category\n')
i = 1
for item in predicted:
    write_file.write('{},{}\n'.format(i, item))
    i += 1
write_file.close()
# print(data)
'''
pd.DataFrame(data=data[1:,1:],    # values
              index=data[1:,0],    # 1st column as index
              columns=['article_id', 'category'])
'''
# print(np.mean(predicted == y_test))
# {'clf-svm__alpha': 0.001, 'clf-svm__loss': 'hinge', 'clf-svm__penalty': 'l2', 'clf-svm__warm_start': True, 'tfidf__use_idf': True, 'vect__ngram_range': (1, 1)}
# 0.6637627048330222

