from nltk.stem.porter import *
stemmer = PorterStemmer()

import matplotlib.pyplot as plt
%matplotlib inline


def my_tokenizer(s):
    words = re.findall(r'[A-Za-z]+', s)
    words = [word.lower() for word in words]
    words = [stemmer.stem(word) for word in words]
    return words

# below are Junda's work


# for submit
X_test = test_df['title']

text_clf_svm = Pipeline([('vect', CountVectorizer(ngram_range=(1, 0), tokenizer=my_tokenizer, stop_words='english')),
                      ('tfidf', TfidfTransformer(use_idf=True)),
                      ('clf-svm', SGDClassifier(alpha=0.001, loss='hinge', penalty='l2', warm_start=True, n_iter=6, random_state=42)),])

text_clf_svm = text_clf_svm.fit(X.values.astype('U'), y.values)
print('fit done')

result = text_clf_svm.predict(X_test)

write_file = open('result-11-06-3.txt', 'w+')
write_file.write('article_id,category\n')
i = 1
for item in result:
    write_file.write('{},{}\n'.format(i, item))
    i += 1
write_file.close()

plt.hist(result, normed=True, bins=5)
plt.ylabel('Probability')
plt.show()

# ngram_range(1, 0): submission result: 0.85886
# ngram_range(1, 2): submission result: 0.87349
# ngram_range(1, 3): submission result: 0.86879
