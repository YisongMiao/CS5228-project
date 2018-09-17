# Idea

目前大体来说有3条线，我们可以都实现，之后再集成(ensemble)在一起



## 第一条线 - Association Rule

Intuition 是：如果一则新闻的Publisher 是 `Forbus`, 而且Title里出现了 `Stocks` 和 `$`的两个字眼， 那么它就是category 1 的.

我们要做的就是，挖掘出这些Association Rule。

这个做法的优势是利用Rule来分类，从经验主义来说Precision可能比较高，劣势是这些Rule的数量可能有限，以及我们如何挑选Rule也需要有很好的标准。



## 第二条线 - 传统的分类模型

Title, URL 代表的是word feature, 用 bag-of-word 或者 N-gram 模型，应该就有不错的效果了！

Publisher, Timestamp这些feature 也集成进去.



## 第三条线 - 深度学习

这个最大的优势是可以使用word embedding

word embedding可以让training 里没有出现的word, 在test中也可以被正确的理解，所以当training 和 test 的差异比较大的时候，这个方法就显得很有帮助

Yisong本科的时候做过RNN的文本分类，这个套他以前的代码很快就可以搞定