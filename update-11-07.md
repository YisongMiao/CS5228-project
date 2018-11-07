Hi 俊达



我今天取得了两组提高：



## 提高一

0.87349 -> 0.87401 其实是昨天 的 my_tokenizer 有问题，stupid mistake: 出来很多长度为1 的字母，忘删掉了；



## 提高二

0.87401 -> 0.87506 这个就非常有趣了，把obamacare 当作 stop word… 立马提高 了 0.1

我觉得这有很大的启示意义，说明 许多东西产生的噪音是很大的，我们可以从这个地方入手来解决，我提供一个思路：

看看有哪些关键词，他们的document frequency 很高，但是在各个category 都有分布。

例如 obamacare 在5个 category中的分布大概是： [0.1, 0.1, 0.2, 0.1, 0.4]。我忘了，你可以自己plot 出来看看。



这样，我们就有一组新的stopword!



同时，default 里的 stopword 其实也可以做一些 个性化的修改，我举个例子：

在某一组里，似乎情绪都比较消极，会有 decline, drop 等词，这时 会有`down` 这个词，这时其实需要把down 保留下来，因为是比较 make sense的，`down` 其实可以保留下来。

我觉得这个方法，甚至可以得到 1%-2% 的提高！



## 新方向1

我其实发现，许多动词 都很有效，我们观察一下就会发现，句子中最核心的意思，要么是由2-3个关键的名词来表达的，要么是由`1个关键的动词来表达的`

我准备把 train 和 test 中的动词都提取出来，看看哪些test中的动词（非常有意义），在train中没出现，我们直接在train中 mini-batch update, 把这些动词埋进去，说不定也会有提高



## 新方向2

可以用陈松的那个 55%可以用的爬虫，其实确实可以用。

目前排名最高的组，其实也只有一半的网页有扩充文本，他们从84 提高到 88

这个说明了两个问题：

- 那一半的扩充文本就够了，足以获得提高（即使不够均匀）
- 这个提高可能会有巨大的提高

我们需要对陈松的那个扩充文本有好的清洗



## 文件：

- 扩充文本：

https://github.com/YisongMiao/CS5228-project/blob/master/data/train_v2.pk

https://github.com/YisongMiao/CS5228-project/blob/master/data/test_v2.pk

每个pickle里蕴含了一个 list

list[i] 就是 编号为 i 的网页 的内容，我觉得质量还可以



- 我今天的代码：

https://github.com/YisongMiao/CS5228-project/blob/master/11-07-best.ipynb

里面有些我自己尝试、打印的东西，你可以忽略～