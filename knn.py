from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
import string
import re
import numpy as np
from collections import Counter
import pandas as pd
from sklearn.model_selection import train_test_split

stop = set(stopwords.words('english'))
exclude = set(string.punctuation)
lemma = WordNetLemmatizer()


# Cleaning the text sentences so that punctuation marks, stop words & digits are removed
def clean(doc):
	stop_free = " ".join([i for i in doc.lower().split() if i not in stop])
	punc_free = ''.join(ch for ch in stop_free if ch not in exclude)
	normalized = " ".join(lemma.lemmatize(word) for word in punc_free.split())
	processed = re.sub(r"\d+", "", normalized)
	y = processed.split()
	return y

df = pd.read_csv('train_v2.csv', header=0)
# 0 = Ratings downgrade, 1 = Sanctions, 2 = Growth into new markets, 3 = New product coverage, 4 = Others
# remove category
df = df[(df.category > 1) & (df.category != 3)]
# now we change 0-3 to 0, 4 to 1 in category
df.category.replace([2, 4], [1, 0], inplace=True)

X = df['title']
Y = df['category']
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.20)

train_clean_sentences = []
'''
fp = open(path, 'r')
for line in fp:
	line = line.strip()
	cleaned = clean(line)
	cleaned = ' '.join(cleaned)
	train_clean_sentences.append(cleaned)
'''

for each in X_train:
	each = str(each).strip()
	cleaned = clean(each)
	cleaned = ' '.join(cleaned)
	train_clean_sentences.append(cleaned)

vectorizer = TfidfVectorizer(stop_words='english')
X_process = vectorizer.fit_transform(train_clean_sentences)

# Clustering the document with KNN classifier
modelknn = KNeighborsClassifier(n_neighbors=7, weights='distance')
modelknn.fit(X_process, y_train)

# Clustering the training 30 sentences with K-means technique
# modelkmeans = KMeans(n_clusters=5, init='k-means++', max_iter=200, n_init=100)
# modelkmeans.fit(X_process)

test_clean_sentence = []
for each in X_test:
	cleaned_test = clean(str(each))
	cleaned = ' '.join(cleaned_test)
	cleaned = re.sub(r"\d+", "", cleaned)
	test_clean_sentence.append(cleaned)

Test = vectorizer.transform(test_clean_sentence)

predicted_labels_knn = modelknn.predict(Test)
# predicted_labels_kmeans = modelkmeans.predict(Test)

print(np.mean(predicted_labels_knn == y_test))
# print(np.mean(predicted_labels_kmeans == y_test))