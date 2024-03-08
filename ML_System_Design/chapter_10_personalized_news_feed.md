## Chapter 10: Personalized News Feed

### 0. Clarification
- the system retrieves unseen posts or posts with unseen comments
- ranks them based on how engaging they are to user
- take no longer than 200 ms
- objective of the system is to increase user engagement
    - engagement includes, impression, click, like, share, comments...

### 1. Frame as ML task
1.1 Define the ML objective
- maximize the num of specific implicit reactions, such as dwell time or user clicks
- maximize the num of specific explicit reactions, such as like or share
- maximize a weightd score based on both implicit and explicit reactions

| Reaction | click | like | comment | share | friendship request | hide | block|
|----------|----------|----------|----------|----------|----------|----------|----------|
| Weight   | 1 | 5 | 10 | 20 | 30| -20 | -30|


1.2 Specify input and output: user => personalized news feed system => a ranked list of posts based on engagement

1.3 Choose the right ML category:
- Pointwise Learning to Rank (LTR) to rank posts by engagement scores (between a user and a post)
- A binary classifier to predict the prob of various reactions for a <user, post> pair



### Reference
- [x] [News Feed Ranking in Facebook](https://engineering.fb.com/2021/01/26/ml-applications/news-feed-ranking/)
- [x] [Twitter's news feed system](https://blog.x.com/engineering/en_us/topics/insights/2017/using-deep-learning-at-scale-in-twitters-timelines#:~:text=Using%20Deep%20Learning%20at%20Scale%20in%20Twitter%E2%80%99s%20Timelines,companies.%20...%203%20Impact%20...%204%20Acknowledgements%20)
- [x] [LinkedIn's News Feed system](https://www.linkedin.com/blog/engineering/feed/understanding-feed-dwell-time)
- [ ] [BERT paper]
- [x] [ResNet model]
- [x] [CLIP model: learns visual concepts from natural language supervision](https://openai.com/research/clip)
- [ ] [Viterbi algorithm for hashtag segmentation](https://en.wikipedia.org/wiki/Viterbi_algorithm)
- [x] [TF-IDF: a measure of importance of a word to a document in a collection or corpus](https://en.wikipedia.org/wiki/Tf%E2%80%93idf)
- [x] [Word2Vec](https://en.wikipedia.org/wiki/Word2vec)
- [x] [Serving a billion personalized new feed](https://www.youtube.com/watch?v=Xpx5RYNTQvg)
- [x] [MSE loss](https://en.wikipedia.org/wiki/Mean_squared_error)
- [x] [MAE loss](https://en.wikipedia.org/wiki/Mean_absolute_error)
- [x] [Huber loss](https://en.wikipedia.org/wiki/Huber_loss)
- [x] [Binary cross entropy loss](https://en.wikipedia.org/wiki/Cross-entropy)
- [ ] [A news feed system design](https://liuzhenglaichn.gitbook.io/system-design/news-feed/design-a-news-feed-system)
- [ ] [Predict viral tweets](https://towardsdatascience.com/using-data-science-to-predict-viral-tweets-615b0acc2e1e)
- [ ] [Cold start problem in recommendation systems](https://en.wikipedia.org/wiki/Cold_start_(recommender_systems))
- [ ] [Positional bias](https://eugeneyan.com/writing/position-bias/)
- [x] [Determine retraining frequency](https://huyenchip.com/2022/01/02/real-time-machine-learning-challenges-and-solutions.html#continual-learning)