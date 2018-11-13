# CS5228 Final Project - News Article Classification

Project webpage: https://www.kaggle.com/c/cs5228-article-category-prediction

Project instructor:

- Prof. Wang Wei (https://www.comp.nus.edu.sg/~wangwei/)
- PhD Candidate Qi Panpan
- PhD Candidate Zhu Lei

## Replicate out experiment

- `python model-1.py` 

  Use this to train model 1 and make prediction. (with only title)

- `python model-2.py`

  Use this to train model 2 and make prediction. (with title and webpage content)

- `python ensemble-rule-based.py`

  Our rule based ensemble method

- `python check-replication.py`

  You can use this to check if you(also we) can replicate our best work on Kaggle, the answer is YES!



## Dataset to contribute

We contribute our obtained webpage content, you can use this to push your model's performance to a higher level! Our team is really excited about that!

Usage: It is a pickle file, after load from pickle, it is a list, the index is in the same order as that in the original dataset.

Link:

 https://github.com/YisongMiao/CS5228-project/blob/master/data/train_v2.pk

https://github.com/YisongMiao/CS5228-project/blob/master/data/test_v2.pk



## Study purpose

- https://github.com/YisongMiao/CS5228-project/blob/master/zjd/first%20version/svm.py

  It is code by Junda to use grid search to find best parameter

- https://github.com/YisongMiao/CS5228-project/blob/master/get-webpage-content/spider.py

  It is code by Chen Song to obtain webpage content


# Give us a star on this git repo if you like it!